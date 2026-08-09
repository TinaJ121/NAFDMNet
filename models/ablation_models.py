import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__(); self.drop_prob = drop_prob
    def forward(self, x): return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False), nn.ReLU(),
            nn.Conv2d(mid, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention()
    def forward(self, x): return x * self.sa(x * self.ca(x))

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
    return x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H//window_size, W//window_size, window_size, window_size, -1)
    return x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.window_size = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2*window_size[0]-1)*(2*window_size[1]-1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = (coords_flatten[:,:,None] - coords_flatten[:,None,:])
        relative_coords = relative_coords.permute(1,2,0).contiguous()
        relative_coords[:,:,0] += window_size[0] - 1
        relative_coords[:,:,1] += window_size[1] - 1
        relative_coords[:,:,0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2,-1)
        rel_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0]*self.window_size[1],
            self.window_size[0]*self.window_size[1], -1)
        attn = attn + rel_bias.permute(2,0,1).contiguous().unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B_//nW, nW, self.num_heads, N, N) +
                    mask.unsqueeze(1).unsqueeze(0))
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., drop=0., attn_drop=0., drop_path_rate=0.):
        super().__init__()
        self.dim = dim; self.window_size = window_size; self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = WindowAttention(dim, (window_size,window_size), num_heads,
                                     attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = Mlp(dim, int(dim*mlp_ratio), drop=drop)
    def _get_attn_mask(self, Hp, Wp, device):
        if self.shift_size > 0:
            img_mask = torch.zeros((1,Hp,Wp,1), device=device)
            h_slices = (slice(0,-self.window_size), slice(-self.window_size,-self.shift_size), slice(-self.shift_size,None))
            w_slices = (slice(0,-self.window_size), slice(-self.window_size,-self.shift_size), slice(-self.shift_size,None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:,h,w,:] = cnt; cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size**2)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            return attn_mask.masked_fill(attn_mask!=0,-100.).masked_fill(attn_mask==0,0.)
        return None
    def forward(self, x):
        B, C, H, W = x.shape; shortcut = x
        x = x.permute(0,2,3,1)
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0,0,0,pad_r,0,pad_b))
        _, Hp, Wp, _ = x.shape
        shifted_x = torch.roll(x,(-self.shift_size,-self.shift_size),dims=(1,2)) if self.shift_size>0 else x
        attn_mask = self._get_attn_mask(Hp, Wp, x.device)
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size**2, C)
        attn_windows = self.attn(self.norm1(x_windows), mask=attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        x = torch.roll(shifted_x,(self.shift_size,self.shift_size),dims=(1,2)) if self.shift_size>0 else shifted_x
        if pad_r>0 or pad_b>0: x = x[:,:H,:W,:].contiguous()
        x = x.permute(0,3,1,2)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0,2,3,1))).permute(0,3,1,2))
        return x

def estimate_noise_mad(x):
    lap = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],
                       dtype=x.dtype, device=x.device).view(1,1,3,3)
    noise_maps, sigmas = [], []
    for i in range(x.shape[0]):
        xi = x[i:i+1]
        filtered = F.conv2d(xi, lap, padding=1)
        med = filtered.median()
        mad = torch.median(torch.abs(filtered - med))
        sigma = torch.clamp(mad / 0.6745, min=1e-6, max=1.0)
        sigma_norm = torch.clamp(sigma / 0.25, 0.0, 1.0)
        noise_maps.append(torch.ones_like(xi) * sigma_norm)
        sigmas.append(sigma_norm.view(1,1))
    return torch.cat(noise_maps), torch.cat(sigmas)


def _res_conv(dim):
    return nn.Sequential(
        nn.Conv2d(dim, dim, 3,1,1), nn.BatchNorm2d(dim), nn.GELU(),
        nn.Conv2d(dim, dim, 3,1,1), nn.BatchNorm2d(dim))

def make_decoder(hidden_dim, use_skip=False):
    if use_skip:
        return SkipDecoderFull(hidden_dim)
    else:
        return SimpleDecoder(hidden_dim)

class SimpleDecoder(nn.Module):
    def __init__(self, hidden_dim=160):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim//2, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim//2, hidden_dim//4, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim//4, hidden_dim//8, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim//8, 1, 3,1,1))
        init.constant_(self.dec[-1].weight, 0)
        init.constant_(self.dec[-1].bias,   0)
    def forward(self, feat, skip=None):
        return self.dec(feat)

class SkipDecoderFull(nn.Module):
    def __init__(self, hidden_dim=160):
        super().__init__()
        self.dec1      = nn.Sequential(nn.Conv2d(hidden_dim, hidden_dim, 3,1,1), nn.GELU())
        self.skip_fuse = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, 1), nn.GELU())
        self.dec2      = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim//2, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim//2, hidden_dim//4, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim//4, hidden_dim//8, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim//8, 1, 3,1,1))
        init.constant_(self.dec2[-1].weight, 0)
        init.constant_(self.dec2[-1].bias,   0)
    def forward(self, feat, skip):
        x = self.dec1(feat)
        x = self.skip_fuse(torch.cat([x, skip], dim=1))
        return self.dec2(x)


class NAFDMNet_A1_Baseline(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=160):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim//2, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, 3,1,1), nn.BatchNorm2d(hidden_dim))
        self.cnn_blocks = nn.ModuleList([_res_conv(hidden_dim) for _ in range(6)])
        self.decoder    = SimpleDecoder(hidden_dim)

    def forward(self, x):
        feat = self.in_conv(x)
        skip = feat
        for blk in self.cnn_blocks:
            feat = blk(feat) + feat
        R = self.decoder(feat)
        out = torch.clamp(x + R, 0, 1)
        return out, {}


class NAFDMNet_A2_PlusSwin(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=160, drop_path_rate=0.1):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim//2, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, 3,1,1), nn.BatchNorm2d(hidden_dim))
        self.cnn_blocks  = nn.ModuleList([_res_conv(hidden_dim) for _ in range(6)])
        dpr = [drop_path_rate * i / 3 for i in range(4)]
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(hidden_dim, 8, 8, shift_size=0 if i%2==0 else 4,
                                 drop_path_rate=dpr[i], attn_drop=0.05, drop=0.05)
            for i in range(4)])
        self.fuse_conv = nn.Conv2d(hidden_dim*2, hidden_dim, 1)
        self.norm      = nn.LayerNorm(hidden_dim)
        self.decoder   = SimpleDecoder(hidden_dim)

    def forward(self, x):
        xs = self.in_conv(x)
        cf = xs
        for blk in self.cnn_blocks: cf = blk(cf) + cf
        sf = xs
        for blk in self.swin_blocks: sf = blk(sf)
        sf   = sf + xs
        fuse = self.fuse_conv(torch.cat([cf, sf], dim=1))
        fuse = self.norm(fuse.permute(0,2,3,1)).permute(0,3,1,2)
        R    = self.decoder(fuse)
        return torch.clamp(x+R, 0, 1), {}


class NAFDMNet_A3_PlusCBAM(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=160, drop_path_rate=0.1):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim//2, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, 3,1,1), nn.BatchNorm2d(hidden_dim))
        self.cnn_blocks  = nn.ModuleList([_res_conv(hidden_dim) for _ in range(6)])
        dpr = [drop_path_rate * i / 3 for i in range(4)]
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(hidden_dim, 8, 8, shift_size=0 if i%2==0 else 4,
                                 drop_path_rate=dpr[i], attn_drop=0.05, drop=0.05)
            for i in range(4)])
        self.cbam      = CBAM(hidden_dim*2, reduction=16)
        self.fuse_conv = nn.Conv2d(hidden_dim*2, hidden_dim, 1)
        self.norm      = nn.LayerNorm(hidden_dim)
        self.decoder   = SimpleDecoder(hidden_dim)

    def forward(self, x):
        xs = self.in_conv(x)
        cf = xs
        for blk in self.cnn_blocks: cf = blk(cf) + cf
        sf = xs
        for blk in self.swin_blocks: sf = blk(sf)
        sf   = sf + xs
        fuse = self.cbam(torch.cat([cf, sf], dim=1))
        fuse = self.fuse_conv(fuse)
        fuse = self.norm(fuse.permute(0,2,3,1)).permute(0,3,1,2)
        R    = self.decoder(fuse)
        return torch.clamp(x+R, 0, 1), {}


class NAFD_NoAdaptive(nn.Module):
    def __init__(self, in_dim, **kwargs):
        super().__init__()
        self.low_enhance  = self._make_branch(in_dim)
        self.mid_enhance  = self._make_branch(in_dim)
        self.high_enhance = self._make_branch(in_dim)
        self.low_scale  = nn.Parameter(torch.zeros(1))
        self.mid_scale  = nn.Parameter(torch.zeros(1))
        self.high_scale = nn.Parameter(torch.zeros(1))
        self.out_norm   = nn.BatchNorm2d(in_dim)

    def _make_branch(self, dim):
        branch = nn.Sequential(
            nn.Conv2d(dim, dim, 3,1,1, groups=dim), nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1), nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3,1,1), nn.BatchNorm2d(dim))
        init.constant_(branch[-2].weight, 0); init.constant_(branch[-2].bias, 0)
        init.constant_(branch[-1].weight, 0); init.constant_(branch[-1].bias, 0)
        return branch

    def _freq_masks(self, H, W, device):
        cy, cx = H//2, W//2
        y = torch.arange(H, device=device).float() - cy
        x = torch.arange(W, device=device).float() - cx
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        dist = torch.sqrt(xx**2 + yy**2)
        R = min(H,W)/2; r_low, r_mid = R*0.15, R*0.40
        low  = torch.sigmoid((r_low-dist)*0.8)
        mid  = torch.sigmoid((dist-r_low)*0.8)*torch.sigmoid((r_mid-dist)*0.8)
        high = torch.sigmoid((dist-r_mid)*0.8)
        total = low+mid+high+1e-8
        return (low/total).view(1,1,H,W),(mid/total).view(1,1,H,W),(high/total).view(1,1,H,W)

    def _freq_split(self, x):
        H, W = x.shape[-2], x.shape[-1]
        xf   = torch.fft.rfft2(x, norm='ortho')
        xf_s = torch.fft.fftshift(xf, dim=-2)
        lm, mm, hm = self._freq_masks(H, W//2+1, x.device)
        def _back(f): return torch.fft.irfft2(torch.fft.ifftshift(f,dim=-2),s=(H,W),norm='ortho')
        return _back(xf_s*lm), _back(xf_s*mm), _back(xf_s*hm)

    def forward(self, x, noise_map=None, z_sigma=None):
        low_s, mid_s, high_s = self._freq_split(x)
        dl = self.low_enhance(low_s)   * torch.tanh(self.low_scale)
        dm = self.mid_enhance(mid_s)   * torch.tanh(self.mid_scale)
        dh = self.high_enhance(high_s) * torch.tanh(self.high_scale)
        delta = (dl + dm + dh) / 3.0
        return self.out_norm(x + delta), None


class NAFDMNet_A4_NAFDnoAdapt(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=160, drop_path_rate=0.1):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim//2, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, 3,1,1), nn.BatchNorm2d(hidden_dim))
        self.cnn_blocks  = nn.ModuleList([_res_conv(hidden_dim) for _ in range(6)])
        dpr = [drop_path_rate * i / 3 for i in range(4)]
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(hidden_dim, 8, 8, shift_size=0 if i%2==0 else 4,
                                 drop_path_rate=dpr[i], attn_drop=0.05, drop=0.05)
            for i in range(4)])
        self.cbam      = CBAM(hidden_dim*2, reduction=16)
        self.fuse_conv = nn.Conv2d(hidden_dim*2, hidden_dim, 1)
        self.norm      = nn.LayerNorm(hidden_dim)
        self.nafd      = NAFD_NoAdaptive(hidden_dim)
        self.decoder   = SimpleDecoder(hidden_dim)

    def forward(self, x):
        xs = self.in_conv(x)
        cf = xs
        for blk in self.cnn_blocks: cf = blk(cf) + cf
        sf = xs
        for blk in self.swin_blocks: sf = blk(sf)
        sf   = sf + xs
        fuse = self.cbam(torch.cat([cf, sf], dim=1))
        fuse = self.fuse_conv(fuse)
        fuse = self.norm(fuse.permute(0,2,3,1)).permute(0,3,1,2)
        mod_feat, _ = self.nafd(fuse)
        R = self.decoder(mod_feat)
        return torch.clamp(x+R, 0, 1), {}


class MultiScaleFrequencyModulation(nn.Module):
    def __init__(self, in_dim, latent_dim=256, drop=0.1):
        super().__init__()
        self.noise_encoder = nn.Sequential(
            nn.Conv2d(1, in_dim//4, 3,1,1), nn.GELU(),
            nn.Conv2d(in_dim//4, in_dim//2, 3,1,1), nn.GELU(),
            nn.Conv2d(in_dim//2, in_dim, 3,1,1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(in_dim, latent_dim),
            nn.LayerNorm(latent_dim), nn.GELU(), nn.Dropout(drop))
        self.sigma_encoder = nn.Sequential(
            nn.Linear(1, 64), nn.GELU(),
            nn.Linear(64, latent_dim//2),
            nn.LayerNorm(latent_dim//2))
        self.freq_weight_head = nn.Sequential(
            nn.Linear(latent_dim+latent_dim//2, latent_dim//2),
            nn.GELU(), nn.Linear(latent_dim//2, 3), nn.Softmax(dim=1))
        self.low_enhance  = self._make_branch(in_dim)
        self.mid_enhance  = self._make_branch(in_dim)
        self.high_enhance = self._make_branch(in_dim)
        self.low_scale  = nn.Parameter(torch.zeros(1))
        self.mid_scale  = nn.Parameter(torch.zeros(1))
        self.high_scale = nn.Parameter(torch.zeros(1))
        self.out_norm   = nn.BatchNorm2d(in_dim)

    def _make_branch(self, dim):
        branch = nn.Sequential(
            nn.Conv2d(dim,dim,3,1,1,groups=dim), nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim,dim,1), nn.BatchNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim,dim,3,1,1), nn.BatchNorm2d(dim))
        init.constant_(branch[-2].weight,0); init.constant_(branch[-2].bias,0)
        init.constant_(branch[-1].weight,0); init.constant_(branch[-1].bias,0)
        return branch

    def _freq_masks(self, H, W, device):
        cy,cx = H//2,W//2
        y = torch.arange(H,device=device).float()-cy
        x = torch.arange(W,device=device).float()-cx
        yy,xx = torch.meshgrid(y,x,indexing='ij')
        dist = torch.sqrt(xx**2+yy**2)
        R = min(H,W)/2; r_low,r_mid = R*0.15,R*0.40
        low  = torch.sigmoid((r_low-dist)*0.8)
        mid  = torch.sigmoid((dist-r_low)*0.8)*torch.sigmoid((r_mid-dist)*0.8)
        high = torch.sigmoid((dist-r_mid)*0.8)
        total = low+mid+high+1e-8
        return (low/total).view(1,1,H,W),(mid/total).view(1,1,H,W),(high/total).view(1,1,H,W)

    def _freq_split(self, x):
        H,W = x.shape[-2],x.shape[-1]
        xf   = torch.fft.rfft2(x,norm='ortho')
        xf_s = torch.fft.fftshift(xf,dim=-2)
        lm,mm,hm = self._freq_masks(H,W//2+1,x.device)
        def _back(f): return torch.fft.irfft2(torch.fft.ifftshift(f,dim=-2),s=(H,W),norm='ortho')
        return _back(xf_s*lm),_back(xf_s*mm),_back(xf_s*hm)

    def forward(self, x, noise_map, z_sigma):
        B = x.shape[0]
        z_map  = self.noise_encoder(noise_map)
        z_sig  = self.sigma_encoder(z_sigma)
        freq_w = self.freq_weight_head(torch.cat([z_map,z_sig],dim=1))
        low_s,mid_s,high_s = self._freq_split(x)
        dl = self.low_enhance(low_s)   * torch.tanh(self.low_scale)
        dm = self.mid_enhance(mid_s)   * torch.tanh(self.mid_scale)
        dh = self.high_enhance(high_s) * torch.tanh(self.high_scale)
        fw = freq_w.view(B,3,1,1)
        delta = fw[:,0:1]*dl + fw[:,1:2]*dm + fw[:,2:3]*dh
        return self.out_norm(x+delta), freq_w.view(B,3,1,1)


class NAFDMNet_A5_NAFDFull(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=160, latent_dim=320, drop_path_rate=0.1):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim//2, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, 3,1,1), nn.BatchNorm2d(hidden_dim))
        self.cnn_blocks  = nn.ModuleList([_res_conv(hidden_dim) for _ in range(6)])
        dpr = [drop_path_rate*i/3 for i in range(4)]
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(hidden_dim, 8, 8, shift_size=0 if i%2==0 else 4,
                                 drop_path_rate=dpr[i], attn_drop=0.05, drop=0.05)
            for i in range(4)])
        self.cbam      = CBAM(hidden_dim*2, reduction=16)
        self.fuse_conv = nn.Conv2d(hidden_dim*2, hidden_dim, 1)
        self.norm      = nn.LayerNorm(hidden_dim)
        self.freq_mod  = MultiScaleFrequencyModulation(hidden_dim, latent_dim)
        self.decoder   = SimpleDecoder(hidden_dim)

    def forward(self, x):
        xs = self.in_conv(x)
        cf = xs
        for blk in self.cnn_blocks: cf = blk(cf) + cf
        sf = xs
        for blk in self.swin_blocks: sf = blk(sf)
        sf   = sf + xs
        fuse = self.cbam(torch.cat([cf, sf], dim=1))
        fuse = self.fuse_conv(fuse)
        fuse = self.norm(fuse.permute(0,2,3,1)).permute(0,3,1,2)
        noise_map, z_sigma = estimate_noise_mad(x)
        mod_feat, fw = self.freq_mod(fuse, noise_map, z_sigma)
        R = self.decoder(mod_feat)
        return torch.clamp(x+R, 0, 1), {'freq_weights': fw}


class NAFDMNet_A6_Full(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=160, latent_dim=320, drop_path_rate=0.1):
        super().__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim//2, 3,1,1), nn.GELU(),
            nn.Conv2d(hidden_dim//2, hidden_dim, 3,1,1), nn.BatchNorm2d(hidden_dim))
        self.cnn_blocks  = nn.ModuleList([_res_conv(hidden_dim) for _ in range(6)])
        dpr = [drop_path_rate*i/3 for i in range(4)]
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(hidden_dim, 8, 8, shift_size=0 if i%2==0 else 4,
                                 drop_path_rate=dpr[i], attn_drop=0.05, drop=0.05)
            for i in range(4)])
        self.cbam      = CBAM(hidden_dim*2, reduction=16)
        self.fuse_conv = nn.Conv2d(hidden_dim*2, hidden_dim, 1)
        self.norm      = nn.LayerNorm(hidden_dim)
        self.freq_mod  = MultiScaleFrequencyModulation(hidden_dim, latent_dim)
        self.decoder   = SkipDecoderFull(hidden_dim)

    def forward(self, x):
        xs = self.in_conv(x)
        cf = xs
        for blk in self.cnn_blocks: cf = blk(cf) + cf
        sf = xs
        for blk in self.swin_blocks: sf = blk(sf)
        sf   = sf + xs
        fuse = self.cbam(torch.cat([cf, sf], dim=1))
        fuse = self.fuse_conv(fuse)
        fuse = self.norm(fuse.permute(0,2,3,1)).permute(0,3,1,2)
        noise_map, z_sigma = estimate_noise_mad(x)
        mod_feat, fw = self.freq_mod(fuse, noise_map, z_sigma)
        R = self.decoder(mod_feat, xs)
        out = torch.clamp(x+R, 0, 1)
        return out, {'freq_weights': fw}


ABLATION_MODELS = {
    'A1_Baseline':      NAFDMNet_A1_Baseline,
    'A2_PlusSwin':      NAFDMNet_A2_PlusSwin,
    'A3_PlusCBAM':      NAFDMNet_A3_PlusCBAM,
    'A4_NAFDnoAdapt':   NAFDMNet_A4_NAFDnoAdapt,
    'A5_NAFDFull':      NAFDMNet_A5_NAFDFull,
    'A6_Full':          NAFDMNet_A6_Full,
}

ABLATION_DESCRIPTIONS = {
    'A1_Baseline':    'CNN×6 only, no Swin, no CBAM, no NAFD',
    'A2_PlusSwin':    'A1 + Swin-T×4 (hybrid backbone)',
    'A3_PlusCBAM':    'A2 + CBAM attention fusion',
    'A4_NAFDnoAdapt': 'A3 + NAFD (fixed uniform freq weights, no noise-adaptive)',
    'A5_NAFDFull':    'A3 + NAFD (full noise-adaptive MAD weights)',
    'A6_Full':        'A5 + SkipDecoder (full NAFDMNet v2)',
}


def build_ablation_model(name: str, **kwargs):
    assert name in ABLATION_MODELS, \
        "Unknown ablation: {}. Choose from: {}".format(name, list(ABLATION_MODELS.keys()))
    return ABLATION_MODELS[name](**kwargs)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(2, 1, 256, 256).to(device)

    print("=" * 60)
    print("Ablation Model Summary")
    print("=" * 60)
    for name, desc in ABLATION_DESCRIPTIONS.items():
        model = build_ablation_model(name).to(device)
        with torch.no_grad():
            out, _ = model(x)
        n = count_params(model)
        print("[{}] {:.2f}M params | out: {} | {}".format(
            name, n/1e6, out.shape, desc))
    print("=" * 60)