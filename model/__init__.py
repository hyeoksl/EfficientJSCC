from model.layers import *
from einops import rearrange
__all__=["DeepJSCC_test", "DeepJSCC_main", "AWGNChannel"]

class AWGNChannel(nn.Module):
    """SISO AWGN Channel with proper complex noise variance."""
    def __init__(self, SNRdB=None):
        super().__init__()
        self.default_snr_linear = 10 ** (SNRdB / 10) if SNRdB is not None else None

    def forward(self, x, snr_db=None):
        assert x.dtype == torch.complex64, f"Expected complex64, got {x.dtype}"

        # Power normalization (unit average power per symbol)
        pwr = torch.mean(torch.abs(x) ** 2, dim=-1, keepdim=True)
        x = x / (torch.sqrt(pwr) + 1e-9)

        # Resolve SNR
        if snr_db is not None:
            snr_linear = 10 ** (snr_db / 10.0)
        elif self.default_snr_linear is not None:
            snr_linear = self.default_snr_linear
        else:
            raise ValueError("SNR must be provided either at init or in forward()")

        # Complex AWGN: split noise power equally between I and Q
        noise_std = (1.0 / (2.0 * snr_linear)) ** 0.5
        noise = torch.complex(
            torch.randn(x.shape, device=x.device) * noise_std,
            torch.randn(x.shape, device=x.device) * noise_std,
        )

        return x + noise

class DeepJSCC_test(nn.Module):
    '''
    backbone이 ResNet임.
    '''
    def __init__(self, 
                 dims:list=[256,256,256],        # 내부 backbone에서 쓸 dimension
                 spp:float=0.5,      # 최대 전송 symbol의 배수 (H*W*spp)
                 SNRdB:int=10,       # 훈련 때 쓸 기본 SNR[dB]
                 depth:list=[2,2,4],        # depth별 ResNetBlock 수
                 resgroup:list=[1,1,1],     # ResNetBlcok 그룹 개수
                 itgroup: list=[1,1,1],     # us/ds layer 그룹 개수
                 **kwargs):
        super().__init__()
        self.max_spp=spp
        self.ds_factor=2**len(depth)  # downsampling 횟수
        self.snr=SNRdB
        self.symbol_dim=int(self.max_spp*(self.ds_factor**2))*2   # max_spp가 0.5고 3번 downsampling이면 총 64 channel

        encoder=[PatchEmbedding(dims[0], patch_size=2)]
        decoder=[PatchReconstruct(dims[0], patch_size=2)]

        for step, i in enumerate(depth):
            for _ in range(i):
                encoder.append(ResNetBlock(dims[step], groups=resgroup[step]))
                decoder.insert(0, ResNetBlock(dims[step], groups=resgroup[step]))
            encoder.append(PatchMerging(dims[step], dims[step+1] if step+1<len(depth) else self.symbol_dim, groups=itgroup[step]))
            decoder.insert(0, PatchReverse(dims[step+1] if step+1<len(depth) else self.symbol_dim, dims[step], groups=itgroup[step]))

        encoder[-1]=nn.Conv2d(dims[-1], self.symbol_dim, kernel_size=1, groups=itgroup[-1])
        self.encoder=nn.Sequential(*encoder)

        decoder[0]=nn.Conv2d(self.symbol_dim, dims[-1], kernel_size=1, groups=itgroup[-1])
        self.decoder=nn.Sequential(*decoder)

        self.channel=AWGNChannel(self.snr)

    def get_parser(parser):
        # Union of all arguments used across models
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dims', type=list, default=[256,256,256])
        parser.add_argument('--model.depth', type=list, default=[2,2,4])
        parser.add_argument('--model.resgroup', type=list, default=[1,1,1])
        parser.add_argument('--model.itgroup', type=list, default=[1,1,1])
        return parser
        
    def forward(self,x:torch.Tensor, snrs:torch.Tensor=None):
        B = x.shape[0]
        if snrs is None:
            snrs=torch.ones(B,1,device=x.device)*self.snr

        z=self.encoder(x)   # (B,3,H,W)->(B,dim,H//4,W//4)
        _,_,H,W=z.shape

        y=rearrange(z, 'b (n_c iq) h w -> b (n_c h w) iq', iq=2)  # complex symbol로 reshaping
        y=torch.complex(y[...,0],y[...,1])
        power=torch.sqrt(torch.mean(torch.abs(y)**2,dim=1,keepdim=True)+1e-6)
        y=y/power # Complex symbol powernorm

        y_hat=self.channel(y, snrs)
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        z_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H, w=W)

        # decoding
        x_hat=self.decoder(z_hat)
        x_hat=torch.clamp(x_hat, 0, 1)
        
        return x_hat
    
    def loss(self, output, target):
        out={}
        out['MSE']=torch.nn.functional.mse_loss(output, target)
        out['loss']=out['MSE']
        return out
    
class DeepJSCC_main(nn.Module):
    '''
    backbone이 ConvNeXt임.
    '''
    def __init__(self, 
                 dims:list=[256,256,256],        # 내부 backbone에서 쓸 dimension
                 spp:float=0.5,      # 최대 전송 symbol의 배수 (H*W*spp)
                 SNRdB:int=10,       # 훈련 때 쓸 기본 SNR[dB]
                 depth:list=[2,2,4],        # depth별 ConvNeXtBlock 수
                 resgroup:list=[1,1,1],     # ConvNeXtBlock 그룹 개수
                 itgroup: list=[1,1,1],     # us/ds layer 그룹 개수,
                 shuffle:bool=True,     # ConvNeXtBlock 내에서 채널 shuffle 여부
                 normgroup:bool=False,       # ConvNeXtBlock 내에서 그룹별 LayerNorm 여부
                 **kwargs):
        super().__init__()
        self.max_spp=spp
        self.ds_factor=2**len(depth)  # downsampling 횟수
        self.snr=SNRdB
        self.symbol_dim=int(self.max_spp*(self.ds_factor**2))*2   # max_spp가 0.5고 3번 downsampling이면 총 64 channel

        encoder=[PatchEmbedding(dims[0], patch_size=2)]
        decoder=[PatchReconstruct(dims[0], patch_size=2)]

        for step, i in enumerate(depth):
            for _ in range(i):
                encoder.append(GroupedConvNeXtBlock(dims[step], groups=resgroup[step], shuffle=shuffle, normgroup=normgroup))
                decoder.insert(0, GroupedConvNeXtBlock(dims[step], groups=resgroup[step], shuffle=shuffle, normgroup=normgroup))
            encoder.append(PatchMerging(dims[step], dims[step+1] if step+1<len(depth) else self.symbol_dim, groups=itgroup[step]))
            decoder.insert(0, PatchReverse(dims[step+1] if step+1<len(depth) else self.symbol_dim, dims[step], groups=itgroup[step]))

        encoder[-1]=nn.Conv2d(dims[-1], self.symbol_dim, kernel_size=1, groups=itgroup[-1])
        self.encoder=nn.Sequential(*encoder)

        decoder[0]=nn.Conv2d(self.symbol_dim, dims[-1], kernel_size=1, groups=itgroup[-1])
        self.decoder=nn.Sequential(*decoder)

        self.channel=AWGNChannel(self.snr)

    def get_parser(parser):
        # Union of all arguments used across models
        parser.add_argument('--model.SNRdB', type=float)
        parser.add_argument('--model.spp', type=float)
        parser.add_argument('--model.dims', type=list, default=[256,256,256])
        parser.add_argument('--model.depth', type=list, default=[2,2,4])
        parser.add_argument('--model.resgroup', type=list, default=[1,1,1])
        parser.add_argument('--model.itgroup', type=list, default=[1,1,1])
        parser.add_argument('--model.shuffle', action='store_true')
        parser.add_argument('--model.normgroup', action='store_false')
        return parser
        
    def forward(self,x:torch.Tensor, snrs:torch.Tensor=None):
        B = x.shape[0]
        if snrs is None:
            snrs=torch.ones(B,1,device=x.device)*self.snr

        z=self.encoder(x)   # (B,3,H,W)->(B,dim,H//4,W//4)
        _,_,H,W=z.shape

        y=rearrange(z, 'b (n_c iq) h w -> b (n_c h w) iq', iq=2)  # complex symbol로 reshaping
        y=torch.complex(y[...,0],y[...,1])
        power=torch.sqrt(torch.mean(torch.abs(y)**2,dim=1,keepdim=True)+1e-6)
        y=y/power # Complex symbol powernorm

        y_hat=self.channel(y, snrs)
        y_hat = torch.stack((y_hat.real, y_hat.imag), dim=-1)
        z_hat = rearrange(y_hat, "b (n_c h w) iq -> b (n_c iq) h w", h=H, w=W)

        # decoding
        x_hat=self.decoder(z_hat)
        x_hat=torch.clamp(x_hat, 0, 1)
        
        return x_hat
    
    def loss(self, output, target):
        out={}
        out['MSE']=torch.nn.functional.mse_loss(output, target)
        out['loss']=out['MSE']
        return out