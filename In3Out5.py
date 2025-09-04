import torch
import os
import spintorch
import numpy as np
from spintorch.utils import tic, toc, stat_cuda
from spintorch.plot import wave_integrated, wave_snapshot, plot_output
from spintorch.solver1 import MMSolver
from numpy import pi
import warnings
warnings.filterwarnings("ignore", message=".*Casting complex values to real.*")
import pandas as pd
import torch.nn.functional as F


# -------------------- Parameters --------------------
dx, dy, dz = 50e-9, 50e-9, 20e-9
B0_theta = pi/2
nx, ny = 210, 200
Ms, B0, Bt = 140e3, 60e-3, 1e-3
dt = 20e-12
f1, f2 = 4.726e9, 4.726e9 #4e9, 5e9
f3, f4 = 4.204e9, 4.204e9
f5, f6 = 5.091e9, 5.091e9
timesteps = 1400
device = torch.device('cuda')

# -------------------- Directories --------------------
basedir = 'In3Out5/'
plotdir = 'plots/' + basedir
savedir = 'models/' + basedir
os.makedirs(plotdir, exist_ok=True)
os.makedirs(savedir, exist_ok=True)


Ms_CoPt = 723e3 # saturation magnetization of the nanomagnets (A/m)

r0_x, r0_y, dr, dm, z_off = 10, 10, 2, 2, 10  # starting pos, period, magnet size, z distance
rx, ry = int((nx-2*r0_x)/dr), int((ny-2*r0_y)/dr+1)

rho = torch.zeros((rx, ry))  # Design parameter array

## spinwave propagation allowed region shape defining here
mask = np.zeros((nx, ny))+ 0.25  # boundaries

S = 200 / 288          #  rescale factor
sx = lambda x: int(round(x*S))
sy = lambda y: int(round(y*S))

def tube(x0, x1, y, width):
    w = max(1, int(round(width*S/2)))
    y0, y1 = sy(y) - w, sy(y) + w
    x0, x1 = sx(x0), sx(x1)
    mask[x0:x1+1, y0:y1+1] = 1            

def split(x0, x1, y, width, theta=np.pi/4):
    w = max(1, int(round(width*S/2)))
    x0_, x1_ = sx(x0), sx(x1)
    y_  = sy(y)
    for x in range(x0_, x1_+1):
        off = int(np.tan(theta) * (x - x0_))
        mask[x, y_-off-w:y_-off+w+1] = 1    
        mask[x, y_+off-w:y_+off+w+1] = 1    
        
def up_split(x0, x1, y, width, theta=np.pi/4):
    w = max(1, int(round(width*S/2)))
    x0_, x1_ = sx(x0), sx(x1)
    y_  = sy(y)
    for x in range(x0_, x1_+1):
        off = int(np.tan(theta) * (x - x0_))
        mask[x, y_-off-w:y_-off+w+1] = 1    
        
def down_split(x0, x1, y, width, theta=np.pi/4):
    w = max(1, int(round(width*S/2)))
    x0_, x1_ = sx(x0), sx(x1)
    y_  = sy(y)
    for x in range(x0_, x1_+1):
        off = int(np.tan(theta) * (x - x0_))
        mask[x, y_+off-w:y_+off+w+1] = 1    



# Construction of physical platform
shift = 45
SHIFT = 15              
tube(55-shift, 95-shift,  76+SHIFT, 20)
tube(55-shift, 95-shift, 124+SHIFT, 20)
tube(55-shift, 95-shift, 172+SHIFT, 20)
split(95-shift,120-shift,  76+SHIFT, 20)
split(95-shift,120-shift, 124+SHIFT, 20)
split(95-shift,120-shift, 172+SHIFT, 20)

tube(120-shift,160-shift, 100+SHIFT, 20)
tube(120-shift,160-shift, 147+SHIFT, 20)
tube(120-shift,160-shift,  53+SHIFT, 20)
tube(120-shift,160-shift, 195+SHIFT, 20)
split(160-shift,185-shift,  100+SHIFT, 20)
split(160-shift,185-shift, 147+SHIFT, 20)
split(160-shift,185-shift,  53+SHIFT, 20)
split(160-shift,185-shift, 195+SHIFT, 20)

tube(185-shift,225-shift, 28+SHIFT, 20)
tube(185-shift,225-shift, 76+SHIFT, 20)
tube(185-shift,225-shift,  124+SHIFT, 20)
tube(185-shift,225-shift, 172+SHIFT, 20)
tube(185-shift,225-shift, 218+SHIFT, 20)
split(225-shift,250-shift, 28+SHIFT, 20)
split(225-shift,250-shift,  76+SHIFT, 20)
split(225-shift,250-shift, 124+SHIFT, 20)
split(225-shift,250-shift, 172+SHIFT, 20)
split(225-shift,250-shift, 218+SHIFT, 20)

tube(250-shift,290-shift, 52+SHIFT, 20)
tube(250-shift,290-shift, 100+SHIFT, 20)
tube(250-shift,290-shift,  148+SHIFT, 20)
tube(250-shift,290-shift,  5+SHIFT, 20)
tube(250-shift,290-shift,  195+SHIFT, 20)
tube(250-shift,290-shift,  242+SHIFT, 20)
split(290-shift,315-shift, 52+SHIFT, 20)
split(290-shift,315-shift, 100+SHIFT, 20)
split(290-shift,315-shift,  148+SHIFT, 20)
split(290-shift,315-shift,  195+SHIFT, 20)
down_split(290-shift,315-shift,  5+SHIFT, 20)
up_split(290-shift,315-shift,  242+SHIFT, 20)

tube(315-shift,328-shift, 28+SHIFT, 20)
tube(315-shift,328-shift,  76+SHIFT, 20)
tube(315-shift,328-shift, 124+SHIFT, 20)
tube(315-shift,328-shift, 172+SHIFT, 20)
tube(315-shift,328-shift, 218+SHIFT, 20)

# Apply the M=0 mask to the entire plate
Msat = Ms*mask
Msat = Msat.astype(np.float32)
geom = spintorch.geom.WaveGeometryArray_Ms(rho, (nx, ny), (dx, dy, dz), Msat, B0, B0_theta, 
                                    r0_x, r0_y, dr, dm, z_off, rx, ry, Ms_CoPt,mask)
# Define the source and probes
src1 = spintorch.WaveLineSource(10, 57, 10, 71, dim=2)   
src2 = spintorch.WaveLineSource(10, 90, 10, 104, dim=2)  
src3 = spintorch.WaveLineSource(10, 123, 10, 137, dim=2)  

probes = []
probe_positions = [
    (190, 30, 6),
    (190, 63, 6),
    (190, 97, 6),
    (190, 130, 6),
    (190, 162, 6)
    ]

probes = [spintorch.WaveIntensityProbeDisk(*pos) for pos in probe_positions]

model = MMSolver(geom, dt, [src1, src2, src3] , probes).to(device)
model.retain_history = True

# 3. Data
# 1) Audio siganl and label
X = torch.load(r"INPUT_5_900silence_extract_new.pt", map_location=device)   # (N, 1600, 1)
y = torch.load(r"OUTPUT_5_900silence_extract_new.pt", map_location=device)  # (N,)

# 2) reference signal
N, T, _ = X.shape
dt = 20e-12                                
f  = 0.5e9                                    # Hz
Bt = 1e-3                                   
t  = torch.arange(T, device=device, dtype=X.dtype) * dt   
sig2 = Bt * torch.sin(2 * pi * f * t)                
sig2 = sig2.unsqueeze(0).expand(N, -1).unsqueeze(-1)  

INPUT = torch.cat([sig2, X, sig2], dim=2)      
OUTPUTS = y
# -------------------- Loss & Optimizer --------------------
def selective_loss_debug(output, target_index, eps=1e-12):
    batch_size = output.shape[0]
    target = output[torch.arange(batch_size), target_index]
    nontarget = output.clone()
    nontarget[torch.arange(batch_size), target_index] = 0
    leak = nontarget.sum(dim=1)
    ratio = leak / (target + eps)
    log_ratio = ratio.log10()
    for i in range(batch_size):
        print(f"[Signal {i}] Target: {target[i].item():.3e}, Leak: {leak[i].item():.3e}, Ratio: {ratio[i].item():.3f}, log10: {log_ratio[i].item():.3f}")
    return log_ratio.mean()


def cross_entropy_from_ratio(outputs, labels,
                             alpha: float = 1.5,
                             eps: float = 1e-12,
                             verbose: bool = False):

    B, C = outputs.shape
    idx = torch.arange(B, device=outputs.device)

    # —— Target / Leak ————————————————————————————————
    target = outputs[idx, labels]                             # [B]
    leak   = outputs.sum(dim=1) - target                      # [B]
    ratio  = leak / (target + eps)                            # [B]
    log_ratio = torch.log10(ratio + eps)                      # [B]

    if verbose:
        for b in range(B):
            print(f"[Signal {b}] Target: {target[b]:.3e}, "
                  f"Leak: {leak[b]:.3e}, "
                  f"Ratio: {ratio[b]:.3f}, "
                  f"log10: {log_ratio[b]:.3f}")

    logits = -alpha * torch.log10(
        (leak.unsqueeze(1) - outputs + target.unsqueeze(1) + eps)
        / (outputs + eps)
    )      

    loss = F.cross_entropy(logits, labels)
    return loss

# 1. optimizer & scheduler 
init_lr   = 3e-3
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',         
    factor=0.5,        
    patience=8,        
    threshold=1e-3,   
    verbose=True
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
     optimizer, T_max=80, eta_min=3e-5, verbose=True)

# 2. checkpoint 
ckpt_ep   = 100          # -1 means run from the start
start_ep  = 0
loss_iter = []

if ckpt_ep >= 0:
    ckpt = torch.load(os.path.join(savedir, f'model_e{ckpt_ep}.pt'), map_location='cpu')
	#ckpt = torch.load(savedir / f'model_e{ckpt_ep}.pt', map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    loss_iter = ckpt['loss_iter']
    start_ep  = ckpt['epoch'] + 1
    print(f"✔  Resumed from epoch {ckpt['epoch']}  (LR={optimizer.param_groups[0]['lr']:.2e})")

# Training loop
clip_norm    = 1.0
max_epoch    = 140
for epoch in range(start_ep, max_epoch + 1):
    tic()

    optimizer.zero_grad()

    output_all = model(INPUT)           # (B, T, N_probe)
    u          = output_all.sum(dim=1)  # (B, N_probe)

    loss = cross_entropy_from_ratio(u, OUTPUTS, verbose=True)
    loss_iter.append(loss.item())

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)   
    optimizer.step()
    scheduler.step(loss.item() if isinstance(scheduler,
                         torch.optim.lr_scheduler.ReduceLROnPlateau) else None)

    spintorch.plot.plot_loss(loss_iter, plotdir)
    stat_cuda(f'epoch {epoch:>3d}')

    lr_now = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch:3d} | Loss {loss:7.4f} | LR {lr_now:.2e}")
    toc()

    # Saving
    torch.save({
        'epoch'               : epoch,
        'loss_iter'           : loss_iter,
        'model_state_dict'    : model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, os.path.join(savedir, f'model_e{epoch}.pt'))

    # Final Epoch Visualization
    if model.retain_history and epoch == 0 or epoch ==140:
        num_signals = INPUT.shape[0]
        for i in range(num_signals):
                _ = model(INPUT[i:i+1])  # only run signal[i] to populate m_history
                mz = torch.stack(model.m_history, 1)[0, :, 2, :] - model.m0[0, 2, :].unsqueeze(0).cpu()
                wave_snapshot(model, mz[int(1200/2)-1], f"{plotdir}/snapshot_signal{i+1}_mid_epoch{epoch}.png", r"$m_z$")
                wave_snapshot(model, mz[-1], f"{plotdir}/snapshot_signal{i+1}_end_epoch{epoch}.png", r"$m_z$")
                wave_integrated(model, mz, f"{plotdir}/integrated_signal{i+1}_epoch{epoch}.png")
                plot_output(u[i].detach(), p=len(probes), epoch=epoch, plotdir=plotdir, suffix=f"_signal{i+1}")