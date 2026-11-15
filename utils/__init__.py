from utils.dataset import *
import os, logging, sys
import pynvml
import torch
from tqdm import tqdm

def getUsableGPUs():  # GPU 뭐 쓸 수 있는지 확인
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    
    best_gpu_id = None
    max_free_mem = 0
    
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        # GPU 사용률이 0이고, 가용 메모리가 가장 큰 GPU 탐색
        if utilization.gpu == 0 and mem_info.free > max_free_mem:
            max_free_mem = mem_info.free
            best_gpu_id = i
            
    pynvml.nvmlShutdown()
    return best_gpu_id

def logger_configuration(config, experiment_name):  # 로그 찍기
    logger = logging.getLogger("DeepJSCEC")
    logger.handlers.clear()
    config.save_path = os.path.join(config.save_dir, experiment_name)
    log_path = config.save_path + '/logging.log'
        
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                "%Y-%m-%d %H:%M:%S")
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    filehandler = logging.FileHandler(log_path)
    filehandler.setLevel(logging.DEBUG)
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    logger.setLevel(logging.DEBUG)
    
    return logger


def load_ckpt(ckpt, model, optimizer, lr_scheduler, logger, device=None):  # 학습한 거 불러오기
    if ckpt is not None:
        try:
            if device is None:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            checkpoint = torch.load(ckpt, map_location=device)
            model_state = checkpoint.get("state_dict", checkpoint.get("model"))
            scheduler_state = checkpoint.get("lr_scheduler", checkpoint.get("scheduler"))

            if model_state is None or scheduler_state is None:
                raise KeyError("Checkpoint is missing model or scheduler state")

            model.load_state_dict(model_state)
            start_epoch = checkpoint["epoch"] + 1
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(scheduler_state)
            logger.info("Loaded from" + ckpt)
            best_valid_loss = checkpoint.get("best_valid_loss", float("inf"))
        except:
            start_epoch = 0
            best_valid_loss = float("inf")
            logger.info("Failed to load state from" + ckpt + "Training from scratch...")
    else:
        start_epoch = 0
        best_valid_loss = float("inf")
        logger.info("Training from scratch...")
        
    return start_epoch, best_valid_loss

def save_ckpt(
    config,
    epoch,
    loss,
    is_best,
    model,
    optimizer,
    lr_scheduler,
    logger,
    best_valid_loss=None,
    step=None,
):
    # 학습한 거 저장하기
    state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "loss": loss,
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
    }
    if step is not None:
        state["step"] = step
    if best_valid_loss is not None:
        state["best_valid_loss"] = best_valid_loss

    epoch_id = epoch + 1
    latest_epoch_path = os.path.join(config.save_path, f"latest_ckpt_ep{epoch_id}.pth.tar")
    torch.save(state, latest_epoch_path)
    for fname in os.listdir(config.save_path):
        if fname.startswith("latest_ckpt_ep") and fname != os.path.basename(latest_epoch_path):
            os.remove(os.path.join(config.save_path, fname))

    if step is None:
        logger.info(f"Saved latest model at epoch {epoch_id}")
    else:
        logger.info(f"Saved latest model at epoch {epoch_id}, step {step}")

    # Save best only after validation / epoch completion.
    if is_best and step is None:
        best_epoch_path = os.path.join(config.save_path, f"best_ckpt_ep{epoch_id}.pth.tar")
        torch.save(state, best_epoch_path)
        for fname in os.listdir(config.save_path):
            if fname.startswith("best_ckpt_ep") and fname != os.path.basename(best_epoch_path):
                os.remove(os.path.join(config.save_path, fname))
        logger.info(f"Saved best model at epoch {epoch_id}")
    

class progressMeter:
    def __init__(self, desc, writer, logger, total_data_len, epoch, use_pbar):
        # 텐서 대신 일반 float 누적 딕셔너리 사용
        self.states_sum = {'batch_size': 0.0}
        
        self.writer = writer
        self.logger = logger
        self.use_pbar = use_pbar
        self.epoch = epoch
        self.desc = f'Epoch {epoch} | {desc}:'
        self.total_data_len = total_data_len
        
        if use_pbar:
            # ncols를 제거하고 {bar:10}를 사용하여 막대 길이만 고정
            # 괄호 외부에 {postfix}를 배치하여 긴 지표 문자열이 잘리지 않도록 설정
            custom_format = '{desc} {percentage:3.0f}%|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]{postfix}'
            self.pbar = tqdm(total=total_data_len, file=sys.stdout, desc=self.desc, bar_format=custom_format)
        
    def update(self, state_dict):
        assert 'batch_size' in state_dict.keys(), "Batch size should be provided in states! (as 'batch_size')"
        assert 'loss' in state_dict.keys(), "Loss should be provided in states! (as 'loss')"
        
        self.state_size = state_dict.pop('batch_size')
        self.states_sum['batch_size'] += self.state_size
        
        for name, val in state_dict.items():
            if torch.is_tensor(val):
                val = val.detach().item() # GPU 텐서를 CPU float로 변환
                
            if name not in self.states_sum:
                self.states_sum[name] = 0.0
            self.states_sum[name] += val * self.state_size
    
    def verbose_states(self, extra_state_dict=None):
        total_batch = self.states_sum['batch_size']
        avg_states = {name: val / total_batch for name, val in self.states_sum.items() if name != 'batch_size'}
        
        if self.use_pbar:
            self.pbar.update(int(self.state_size))
            
            # 원하는 형식의 문자열로 직접 조립
            out_str = ""
            for name, val in avg_states.items():
                if name=='PSNR':
                    out_str += f" | {name}:{val:.2f}"
                else:
                    out_str += f" | {name}:{val:.5f}"

            if extra_state_dict is not None:
                for name, val in extra_state_dict.items():
                    if isinstance(val, tuple):
                        out_str += f" | {name}:{val[0]:.2f}-{val[1]:.2f}"
                    elif isinstance(val, float):
                        out_str += f" | {name}:{val:.4f}"
                    else:
                        out_str += f" | {name}:{val}"
                
            # 조립된 문자열을 postfix로 전달
            self.pbar.set_postfix_str(out_str[1:])
        else:
            out = f"{self.desc}"
            out_str = ""
            for name, val in avg_states.items():
                if name=='PSNR':
                    out_str += f" | {name}:{val:.2f}"
                else:
                    out_str += f" | {name}:{val:.5f}"

            if extra_state_dict is not None:
                for name, val in extra_state_dict.items():
                    if isinstance(val, tuple):
                        out_str += f" | {name}:{val[0]:.2f}-{val[1]:.2f}"
                    elif isinstance(val, float):
                        out_str += f" | {name}:{val:.4f}"
                    else:
                        out_str += f" | {name}:{val}"
            print(out + out_str)
    
    def write_summary(self):
        total_batch = self.states_sum['batch_size']
        avg_states = {name: val / total_batch for name, val in self.states_sum.items() if name != 'batch_size'}
        
        out = ''
        for name, val in avg_states.items():
            self.writer.add_scalar(name, val, self.epoch)
            out += f"|{name}: {val:.5f}"
            
        epoch_prog = f"({int(total_batch):4d}/{self.total_data_len:4d})"
        self.logger.debug(f'Epoch Summary|{self.desc} {epoch_prog}{out[1:]}')
        
        if self.use_pbar:
            self.pbar.close()
            
        return avg_states['loss'] # Average Loss
