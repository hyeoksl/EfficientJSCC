import argparse
import ast
import inspect
import math
import os
import random
import re
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import MultiStepLR

from model import *
from utils import getUsableGPUs, load_ckpt, logger_configuration, progressMeter, save_ckpt, load_dataloaders
from torchinfo import summary


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


class PrefixWriter:
    def __init__(self, writer, prefix):
        self.writer = writer
        self.prefix = prefix
        self.metric_names = {
            "loss": "loss",
            "MSE": "mse",
            "PSNR": "psnr",
        }

    def add_scalar(self, name, value, step):
        metric_name = self.metric_names.get(name, name.lower())
        self.writer.add_scalar(f"{self.prefix}/{metric_name}", value, step)


@dataclass
class EpochMetrics:
    loss: float
    mse: float
    psnr: float


def dict_to_namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [dict_to_namespace(v) for v in value]
    return value


def namespace_to_dict(value):
    if isinstance(value, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(value).items()}
    if isinstance(value, list):
        return [namespace_to_dict(v) for v in value]
    return value


def load_config(config_path, overrides=None):
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if overrides:
        i = 0
        while i < len(overrides):
            arg = overrides[i]
            if arg.startswith("--"):
                key = arg[2:]
                # '=' 기호가 포함된 경우 처리 (--model.type=Swin_JSCC)
                if "=" in key:
                    key, val_str = key.split("=", 1)
                else:
                    # 다음 인자가 값인 경우 처리 (--model.type Swin_JSCC)
                    if i + 1 < len(overrides) and not overrides[i + 1].startswith("--"):
                        val_str = overrides[i + 1]
                        i += 1
                    else:
                        val_str = "True"

                # float, int, list 등 적절한 자료형으로 변환 시도
                try:
                    val = ast.literal_eval(val_str)
                except (ValueError, SyntaxError):
                    val = val_str  # 변환 실패 시 문자열로 유지

                # 중첩 딕셔너리 경로를 따라가서 값 업데이트
                keys = key.split(".")
                d = config
                for k in keys[:-1]:
                    if k not in d:
                        d[k] = {}
                    d = d[k]
                d[keys[-1]] = val
            i += 1

    return dict_to_namespace(config)


def save_config(config, save_path):
    with open(save_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(namespace_to_dict(config), handle, sort_keys=False)


def find_latest_checkpoint(log_dir):
    if not log_dir or not os.path.isdir(log_dir):
        return None

    latest_checkpoint = None
    latest_epoch = -1
    pattern = re.compile(r"latest_ckpt_ep(\d+)\.pth\.tar$")
    for filename in os.listdir(log_dir):
        match = pattern.match(filename)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch > latest_epoch:
            latest_epoch = epoch
            latest_checkpoint = os.path.join(log_dir, filename)

    return latest_checkpoint


def resolve_pretrain_ckpt(config, config_path):
    pretrain_ckpt = getattr(config.logging, "pretrain_ckpt", None)
    if pretrain_ckpt:
        return pretrain_ckpt

    config_dir = os.path.dirname(os.path.abspath(config_path))
    if os.path.basename(config_path) == "config.yaml" and os.path.basename(os.path.dirname(config_path)) == config.experiment_name:
        return find_latest_checkpoint(config_dir)

    return None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_device():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        gpu_id = getUsableGPUs()
    except Exception:
        gpu_id = None
    if gpu_id is None:
        gpu_id = 0
    return torch.device(f"cuda:{gpu_id}")


def build_model(config, device):
    model_cls = globals().get(config.model.type)
    if model_cls is None:
        raise ValueError(f"Unsupported model type: {config.model.type}")

    model_kwargs = vars(config.model).copy()
    model_kwargs.pop("type")
    model = model_cls(**model_kwargs).to(device)
    return model


def build_optimizer(config, model):
    lr = float(config.optimizer.learning_rate)
    opt_type = config.optimizer.type.lower()

    if opt_type == "adam":
        return Adam(model.parameters(), lr=lr)
    if opt_type == "adamw":
        return AdamW(model.parameters(), lr=lr)
    if opt_type == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9)
    raise ValueError(f"Unsupported optimizer type: {config.optimizer.type}")


def build_scheduler(config, optimizer):
    milestones = list(getattr(config.optimizer, "lr_milestone_epoch", []))
    gamma = float(getattr(config.optimizer, "lr_gamma", 0.1))
    return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


def build_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        return NullWriter()
    return SummaryWriter(log_dir=log_dir)


def resolve_patch_size(config):
    img_shape = getattr(config.model, "img_shape", None)
    if img_shape is None:
        return (256, 256)
    return tuple(img_shape)


def resolve_snr_bounds(config):
    for attr_name in ("train_snr_range", "snr_range"):
        snr_range = getattr(config.model, attr_name, None)
        if snr_range is not None:
            if len(snr_range) != 2:
                raise ValueError(f"{attr_name} must contain exactly two values")
            return float(snr_range[0]), float(snr_range[1])
    snr = float(config.model.SNRdB)
    return snr, snr


def sample_snrs(batch_size, device, snr_bounds):
    snr_min, snr_max = snr_bounds
    if math.isclose(snr_min, snr_max):
        return torch.full((batch_size, 1), snr_min, device=device)
    return torch.empty((batch_size, 1), device=device).uniform_(snr_min, snr_max)


def resolve_spp_bounds(config):
    spp_max = float(config.model.spp)
    spp_min = 0.03
    if spp_max < spp_min:
        raise ValueError(f"model.spp must be >= {spp_min}, got {spp_max}")
    return spp_min, spp_max


def sample_spp(spp_bounds):
    spp_min, spp_max = spp_bounds
    if math.isclose(spp_min, spp_max):
        return spp_min
    return random.uniform(spp_min, spp_max)


def call_model(model, images, **runtime_kwargs):
    forward_params = inspect.signature(model.forward).parameters
    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in forward_params.values()
    )
    if accepts_kwargs:
        return model(images, **runtime_kwargs)

    supported_kwargs = {
        name: value
        for name, value in runtime_kwargs.items()
        if name in forward_params
    }
    return model(images, **supported_kwargs)


def psnr_from_mse(mse):
    mse = max(mse, 1e-12)
    return 10.0 * math.log10(1.0 / mse)


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    snr_bounds,
    spp_bounds,
    writer,
    logger,
    epoch,
    clip_max_norm=None,
    train=True,
    on_step_end=None,
):
    model.train(train)

    desc = "train" if train else "valid"
    meter = progressMeter(
        desc=desc,
        writer=PrefixWriter(writer, desc),
        logger=logger,
        total_data_len=len(loader.dataset),
        epoch=epoch + 1,
        use_pbar=True,
    )
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step_idx, images in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            batch_size = images.shape[0]
            snrs = sample_snrs(batch_size, device, snr_bounds)
            spp = sample_spp(spp_bounds)

            if train:
                optimizer.zero_grad(set_to_none=True)

            outputs = call_model(model, images, snrs=snrs, spp=spp)
            losses = model.loss(outputs, images)
            loss = losses["loss"]
            mse = losses.get("MSE", loss)

            if train:
                loss.backward()
                if clip_max_norm is not None and clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
                optimizer.step()

            meter.update(
                {
                    "batch_size": batch_size,
                    "loss": loss,
                    "MSE": mse,
                    "PSNR": psnr_from_mse(mse.detach().item()),
                }
            )
            meter.verbose_states()

            if train and on_step_end is not None:
                on_step_end(step_idx, loss.detach().item())

    avg_loss = meter.write_summary()
    avg_mse = meter.states_sum["MSE"] / max(meter.states_sum["batch_size"], 1)
    return EpochMetrics(loss=avg_loss, mse=avg_mse, psnr=psnr_from_mse(avg_mse))


def main():
    parser = argparse.ArgumentParser(description="Train DeepJSCEC models")
    parser.add_argument("--config", default="config.yaml", help="Path to a YAML config file")
    # args는 --config를, unknown_args는 그 외의 인자들을 리스트로 받습니다.
    args, unknown_args = parser.parse_known_args()

    # load_config에 unknown_args 전달
    config = load_config(args.config, overrides=unknown_args)
    set_seed(int(config.seed))

    device = build_device()
    logger = logger_configuration(config.logging, config.experiment_name)
    writer = build_writer(os.path.join(config.logging.save_path, "tensorboard"))
    save_config(config, os.path.join(config.logging.save_path, "config.yaml"))
    logger.info("Using device: %s", device)

    model = build_model(config, device)
    logger.info("\n%s", summary(model, (1, 3, 256, 256), depth=2, verbose=0))
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)
    train_loader, valid_loader = load_dataloaders(
        train_dir=config.dataset.train_dataset,
        valid_dir=config.dataset.valid_dataset,
        train_batch_size=int(config.dataset.train_batch_size),
        valid_batch_size=int(config.dataset.valid_batch_size),
        num_workers=int(config.dataset.num_workers),
        patch_size=resolve_patch_size(config),
    )

    start_epoch, best_valid_loss = load_ckpt(
        resolve_pretrain_ckpt(config, args.config),
        model,
        optimizer,
        scheduler,
        logger,
        device=device,
    )

    snr_bounds = (0.0, 20.0)
    spp_bounds = resolve_spp_bounds(config)
    clip_max_norm = getattr(config, "clip_max_norm", None)
    save_every_train_steps = int(getattr(config.logging, "save_train_checkpoint_every_steps", 500))

    for epoch in range(start_epoch, int(config.epochs)):
        logger.info("Epoch %d/%d", epoch + 1, int(config.epochs))

        def maybe_save_train_checkpoint(step_idx, loss_value):
            if save_every_train_steps <= 0 or step_idx % save_every_train_steps != 0:
                return
            save_ckpt(
                config.logging,
                epoch=epoch,
                step=step_idx,
                loss=loss_value,
                is_best=False,
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                logger=logger,
                best_valid_loss=best_valid_loss,
            )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            snr_bounds=snr_bounds,
            spp_bounds=spp_bounds,
            writer=writer,
            logger=logger,
            epoch=epoch,
            clip_max_norm=clip_max_norm,
            train=True,
            on_step_end=maybe_save_train_checkpoint,
        )
        valid_metrics = run_epoch(
            model=model,
            loader=valid_loader,
            optimizer=optimizer,
            device=device,
            snr_bounds=snr_bounds,
            spp_bounds=spp_bounds,
            writer=writer,
            logger=logger,
            epoch=epoch,
            train=False,
        )
        scheduler.step()

        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

        logger.info(
            "train loss=%.6f mse=%.6f psnr=%.2f | valid loss=%.6f mse=%.6f psnr=%.2f",
            train_metrics.loss,
            train_metrics.mse,
            train_metrics.psnr,
            valid_metrics.loss,
            valid_metrics.mse,
            valid_metrics.psnr,
        )

        is_best = valid_metrics.loss < best_valid_loss
        if is_best:
            best_valid_loss = valid_metrics.loss

        if getattr(config.logging, "save_model", True):
            save_ckpt(
                config.logging,
                epoch=epoch,
                loss=valid_metrics.loss,
                is_best=is_best,
                model=model,
                optimizer=optimizer,
                lr_scheduler=scheduler,
                logger=logger,
                best_valid_loss=best_valid_loss,
            )

    writer.close()


if __name__ == "__main__":
    main()
