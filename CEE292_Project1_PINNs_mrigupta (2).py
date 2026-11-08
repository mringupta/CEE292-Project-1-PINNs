#!/usr/bin/env python
# coding: utf-8

# # Synthetic Dataset
# 
# Dataset was generated using the Crank-Nicolson numerical solver. The parameters of the solver from the SI of the paper were passed to Claude, which produced the code to generate the training and testing datasets. The values for T and R were 5 and 2, respectively, and inferred from the main paper text (p. 13).

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set up plot styles
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 14, 'figure.figsize': (10, 6)})

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for available device (CUDA, MPS, or CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print("Using Apple Metal Performance Shaders (MPS)")
else:
    device = torch.device('cpu')
    print("Using CPU")
print(f"Using device: {device}")


# In[2]:


import os
save_dir = '/Users/mrinalinigupta/Library/CloudStorage/GoogleDrive-mrigupta@stanford.edu/My Drive/1. Classes/1. AU26/CEE 292/CEE292_Project1_PINNs_CoLab'
print(os.path.exists(save_dir))   # should print True


# In[3]:


## Crank–Nicolson reference solution (Zhang & Chen 2026, Supplementary S1)
## Claude: written with Claude; T_D = 5, R = 2 assumed (not stated in paper; consistent with Fig. S2 ranges)
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, identity
from scipy.sparse.linalg import splu

# Parameters
T_D   = 5.0      # dimensionless advection number (vL/D_x)
R     = 2.0      # retardation factor (linear adsorption)
nx    = 400      # 400 uniform cells -> dx = 0.0025
dx    = 1.0 / nx
dt    = 2e-4     # dimensionless time step
t_end = 0.6

x  = np.linspace(0, 1, nx + 1)          # 401 nodes
nt = int(round(t_end / dt))             # 3000 time steps
t  = np.linspace(0, t_end, nt + 1)

# Spatial operator for R*C_t = C_xx - T_D*C_x
# dispersion: central difference; advection: first-order upwind
a = 1/dx**2 + T_D/dx      # coefficient of C_{i-1}
b = -2/dx**2 - T_D/dx     # coefficient of C_i
c = 1/dx**2               # coefficient of C_{i+1}

# Unknowns are nodes 1..nx (node 0 is the inlet, C = 1)
lower = a * np.ones(nx - 1)
main  = b * np.ones(nx)
upper = c * np.ones(nx - 1)
lower[-1] = a + c         # outlet dC/dx = 0: ghost node C_{nx+1} = C_{nx-1}
L = diags([lower, main, upper], [-1, 0, 1], format='csc') / R

bc = np.zeros(nx)
bc[0] = a / R * 1.0       # inlet C = 1 enters the first interior equation

# Crank–Nicolson: (I - dt/2 L) C^{n+1} = (I + dt/2 L) C^n + dt*bc
I  = identity(nx, format='csc')
A  = (I - 0.5 * dt * L).tocsc()
B  = (I + 0.5 * dt * L).tocsc()
lu = splu(A)              # factor once, reuse every step

# Time stepping (IC: C = 0 everywhere at t = 0)
C_all = np.zeros((nt + 1, nx + 1))
C_int = np.zeros(nx)
for k in range(1, nt + 1):
    C_int = lu.solve(B @ C_int + dt * bc)
    C_all[k, 0]  = 1.0
    C_all[k, 1:] = C_int

def idx(tq):
    return int(round(tq / dt))

# Training observations: tD = 0.05..0.30, 21 uniform points in x
t_obs     = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
x_obs_idx = np.arange(0, nx + 1, 20)          # every 20th node -> 21 points
x_obs     = x[x_obs_idx]
C_obs     = np.array([C_all[idx(tq), x_obs_idx] for tq in t_obs])   # (6, 21)

# Test data: tD = 0.4, 0.5, 0.6 at all 401 points
t_test = np.array([0.4, 0.5, 0.6])
C_test = np.array([C_all[idx(tq), :] for tq in t_test])             # (3, 401)

print("C_obs:", C_obs.shape, " C_test:", C_test.shape)
print("Min C at test times:", C_test.min(axis=1).round(3))   # expect ~[0.749, 0.860, 0.923]

np.savez(f'{save_dir}/cn_data.npz', x=x, t=t, C_all=C_all,
         t_obs=t_obs, x_obs=x_obs, C_obs=C_obs, t_test=t_test, C_test=C_test)
# Plot
plt.figure(figsize=(7, 4))
for tq in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    plt.plot(x, C_all[idx(tq)], label=f"$t_D$={tq}")
plt.scatter(np.tile(x_obs, len(t_obs)), C_obs.ravel(), s=10, c='k', zorder=3, label='observations')
plt.xlabel('$x_D$'); plt.ylabel('$C_D$'); plt.title('Crank–Nicolson reference solution')
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.show()


# # Soft-Boundary PINNs Model
# 
# The soft-boundary model from the paper was replicated in order to facilitate direct comparison and limit initialization-biased conclusions. The base code comes from HW01, for which the code was originally used from the lecture SciML notebook. Claude was used to assist in code generation (commented in code) and debugging. The soft-boundary PINNs model does not exactly replicate the paper's soft-boundary PINNs model as the lambda for each loss in the paper was assigned through conducting several trials. For simplicity, and as the objective of the project is to compare the hard and soft implementations of the model, lambda = 1.0 was assigned to all losses.

# In[4]:


# Generate test points for plotting. From the paper, 401 points were used for spatial evaluation at each time step.
x_test_np = np.linspace(0, 1, 401)
# changed with Claude's help to test at 3 different t_ds (t_d = 0.4, 0.5, 0.6)
t_test = [0.4, 0.5, 0.6]


# In[5]:


class PINN(nn.Module):
    """Standard Feedforward Neural Network for PINN"""
    def __init__(self, input_size=2, hidden_size=64, output_size=1, num_layers=4):
        super(PINN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# physics, data, and boundary losses translated from paper with Claude's assistance

# Physical parameters (Section 3.4: T_D = 5, R = 2)
T_D = 5.0
R   = 2.0

def pde_loss(model, xt_interior):
    """Residual of C_xx - T_D*C_x - R*C_t = 0. Input columns: [x, t]."""
    xt_interior.requires_grad_(True)
    C = model(xt_interior)

    # Gradient w.r.t. both inputs: column 0 = dC/dx, column 1 = dC/dt
    dC = torch.autograd.grad(C, xt_interior, torch.ones_like(C), create_graph=True)[0]
    C_x = dC[:, 0:1]
    C_t = dC[:, 1:2]
    # Second derivative in x: differentiate C_x again, keep the x column
    C_xx = torch.autograd.grad(C_x, xt_interior, torch.ones_like(C_x), create_graph=True)[0][:, 0:1]

    pde_residual = C_xx - T_D * C_x - R * C_t
    return torch.mean(pde_residual**2)

def ic_loss(model, xt_ic):
    """Initial condition: C = 0 at t = 0"""
    C = model(xt_ic)
    return torch.mean(C**2)

def inlet_loss(model, xt_inlet):
    """Inlet: C = 1 at x = 0 (t > 0)"""
    C = model(xt_inlet)
    return torch.mean((C - 1.0)**2)

def outlet_loss(model, xt_outlet):
    """Outlet: dC/dx = 0 at x = 1"""
    xt_outlet.requires_grad_(True)
    C = model(xt_outlet)
    C_x = torch.autograd.grad(C, xt_outlet, torch.ones_like(C), create_graph=True)[0][:, 0:1]
    return torch.mean(C_x**2)

def data_loss(model, xt_obs, C_obs_t):
    """Match the Crank–Nicolson observations"""
    C = model(xt_obs)
    return torch.mean((C - C_obs_t)**2)


# In[6]:


## Claude: training points following the paper (Sec. 2.3): 40,000 PDE, 5,000 IC, 5,000 per boundary
t_max = 0.6

def make_training_points(seed=42):
    g = torch.Generator(device='cpu').manual_seed(seed)
    # Interior (PDE) points: random (x, t) in [0,1] x [0, 0.6]
    xt_interior = torch.rand(40000, 2, generator=g) * torch.tensor([1.0, t_max])
    # Initial condition: t = 0, random x
    x_ic = torch.rand(5000, 1, generator=g)
    xt_ic = torch.cat([x_ic, torch.zeros_like(x_ic)], dim=1)
    # Inlet: x = 0, random t
    t_in = torch.rand(5000, 1, generator=g) * t_max
    xt_inlet = torch.cat([torch.zeros_like(t_in), t_in], dim=1)
    # Outlet: x = 1, random t
    t_out = torch.rand(5000, 1, generator=g) * t_max
    xt_outlet = torch.cat([torch.ones_like(t_out), t_out], dim=1)
    return [p.to(device) for p in (xt_interior, xt_ic, xt_inlet, xt_outlet)]

xt_interior, xt_ic, xt_inlet, xt_outlet = make_training_points()

# Observation points from the Crank–Nicolson solver: 6 times x 21 locations = 126 points
X_obs, T_obs = np.meshgrid(x_obs, t_obs)                       # each (6, 21)
xt_obs  = torch.tensor(np.stack([X_obs.ravel(), T_obs.ravel()], axis=1), dtype=torch.float32).to(device)  # (126, 2)
C_obs_t = torch.tensor(C_obs.ravel().reshape(-1, 1), dtype=torch.float32).to(device)                     # (126, 1)
print(xt_interior.shape, xt_ic.shape, xt_inlet.shape, xt_outlet.shape, xt_obs.shape, C_obs_t.shape)


# In[7]:


import time

def train_pinn_soft(model, epochs=8000, lr=0.001, lambda_pde=1.0, lambda_ic=1.0,
                    lambda_inlet=1.0, lambda_outlet=1.0, lambda_data=1.0):
    ## changed epochs and lambda; points/timing adapted with Claude
    """Trains the PINN using the soft constraint method."""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- Training Soft-Constraint PINN (lambdas: pde={lambda_pde}, ic={lambda_ic}, "
          f"inlet={lambda_inlet}, outlet={lambda_outlet}, data={lambda_data}) ---")
    losses = {'total': [], 'pde': [], 'ic': [], 'inlet': [], 'outlet': [], 'data': []}

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    pbar = tqdm(range(epochs), desc="Training Soft-Constraint PINN", ncols=100)
    for epoch in pbar:
        optimizer.zero_grad()

        # Compute losses
        loss_pde    = pde_loss(model, xt_interior)
        loss_ic     = ic_loss(model, xt_ic)
        loss_inlet  = inlet_loss(model, xt_inlet)
        loss_outlet = outlet_loss(model, xt_outlet)
        loss_data   = data_loss(model, xt_obs, C_obs_t)

        # Total loss
        total_loss = (lambda_pde * loss_pde + lambda_ic * loss_ic + lambda_inlet * loss_inlet
                      + lambda_outlet * loss_outlet + lambda_data * loss_data)

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Store and display losses
        losses['total'].append(total_loss.item())
        losses['pde'].append(loss_pde.item())
        losses['ic'].append(loss_ic.item())
        losses['inlet'].append(loss_inlet.item())
        losses['outlet'].append(loss_outlet.item())
        losses['data'].append(loss_data.item())
        if epoch % 100 == 0:   # updating every epoch slows training
            pbar.set_postfix({'Loss': f'{total_loss.item():.2e}', 'PDE': f'{loss_pde.item():.2e}',
                              'data': f'{loss_data.item():.2e}'})

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t_start
    pbar.close()
    print(f"Training time: {train_time:.1f} s")
    return model, losses, train_time


# In[8]:


# Instantiate and train the soft-constraint model (paper: 4 hidden layers x 64, tanh)
torch.manual_seed(42)
pinn_soft = PINN(input_size=2, hidden_size=64, output_size=1, num_layers=4).to(device)
pinn_soft_trained, losses_soft, time_soft = train_pinn_soft(pinn_soft)

# Predictions at t = 0.4, 0.5, 0.6
pinn_soft_trained.eval()
preds_soft = []
with torch.no_grad():
    for tq in t_test:
        t_np = np.full_like(x_test_np, tq)
        xt = torch.tensor(np.stack([x_test_np, t_np], axis=1), dtype=torch.float32).to(device)
        preds_soft.append(pinn_soft_trained(xt).cpu().numpy().flatten())
preds_soft = np.array(preds_soft)   # (3, 401)

# Plot: PINN vs Crank–Nicolson at each test time
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for i, tq in enumerate(t_test):
    axes[i].plot(x_test_np, C_test[i], 'k-', linewidth=3, label='Crank–Nicolson')
    axes[i].plot(x_test_np, preds_soft[i], 'r--', linewidth=2.5, label='PINN (soft)')
    axes[i].set_title(f'$t_D$ = {tq}'); axes[i].set_xlabel('$x_D$'); axes[i].set_ylabel('$C_D$')
    axes[i].legend(); axes[i].grid(True, alpha=0.5)
plt.tight_layout(); plt.show()

# Check boundary values: inlet should be ~1
for i, tq in enumerate(t_test):
    print(f"t = {tq}: C(0) = {preds_soft[i][0]:.4f}, C(1) = {preds_soft[i][-1]:.4f}")


# # The Hard Constraint Implementation
# 
# Base code from SciML notebook (Soft vs. Hard constraint) and updated with Claude.

# In[9]:


class PINNHardConstraint(nn.Module):
    """PINN that enforces C(x,0)=0, C(0,t)=g(t), dC/dx(1,t)=0 by construction.
    Adapted from HW1 hard-constraint class with Claude."""
    def __init__(self, base_network, eps=0.005):
        super(PINNHardConstraint, self).__init__()
        self.base_network = base_network
        self.eps = eps                       # smoothing time for the inlet step

    def forward(self, xt):
        x = xt[:, 0:1]
        t = xt[:, 1:2]

        # psi(x) = x(2 - x): psi(0) = 0, and dpsi/dx = 0 at x = 1
        psi = x * (2.0 - x)

        # Smoothed inlet: g(0) = 0, g(t) -> 1 quickly
        g = 1.0 - torch.exp(-t / self.eps)

        # Network sees (psi, t), so the output depends on x only through psi -> zero slope at x = 1
        nn_output = self.base_network(torch.cat([psi, t], dim=1))

        # Trial function: C = g(t) * (1 + psi * NN)
        C_trial = g * (1.0 + psi * nn_output)
        return C_trial


# In[10]:


## Written by Claude: predictions at t = 0.4, 0.5, 0.6 (same order as C_test rows)
def predict(model):
    model.eval()
    preds = []
    with torch.no_grad():                       # no gradients needed for prediction
        for tq in t_test:
            t_np = np.full_like(x_test_np, tq)  # all points at this time
            xt = torch.tensor(np.stack([x_test_np, t_np], axis=1), dtype=torch.float32).to(device)
            preds.append(model(xt).cpu().numpy().flatten())
    return np.array(preds)                      # shape (3, 401)


# In[11]:


## Written by Claude
def train_pinn_hard(model, epochs=8000, lr=0.001, lambda_pde=1.0, lambda_data=1.0):
    """Trains the PINN using the hard constraint method.
    IC, inlet, and outlet are built into the model, so only PDE + data losses remain.
    Uses the same training points as the soft model (Cell A)."""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n--- Training Hard-Constraint PINN (lambdas: pde={lambda_pde}, data={lambda_data}) ---")
    losses = {'total': [], 'pde': [], 'data': []}

    if device.type == 'cuda':
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    pbar = tqdm(range(epochs), desc="Training Hard-Constraint PINN", ncols=100)
    for epoch in pbar:
        optimizer.zero_grad()

        # Compute losses
        loss_pde  = pde_loss(model, xt_interior)
        loss_data = data_loss(model, xt_obs, C_obs_t)

        # Total loss
        total_loss = lambda_pde * loss_pde + lambda_data * loss_data

        # Backpropagation
        total_loss.backward()
        optimizer.step()

        # Store and display losses
        losses['total'].append(total_loss.item())
        losses['pde'].append(loss_pde.item())
        losses['data'].append(loss_data.item())
        if epoch % 100 == 0:   # updating every epoch slows training
            pbar.set_postfix({'Loss': f'{total_loss.item():.2e}', 'PDE': f'{loss_pde.item():.2e}',
                              'data': f'{loss_data.item():.2e}'})

    if device.type == 'cuda':
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t_start
    pbar.close()
    print(f"Training time: {train_time:.1f} s")
    return model, losses, train_time


# In[12]:


## Written by Claude
torch.manual_seed(42)
base_pinn = PINN(input_size=2, hidden_size=64, output_size=1, num_layers=4).to(device)
pinn_hard = PINNHardConstraint(base_pinn).to(device)
pinn_hard_trained, losses_hard, time_hard = train_pinn_hard(pinn_hard)

preds_hard = predict(pinn_hard_trained)   # (3, 401), same order as C_test

# Check conditions: C(0) should equal g(t) ~ 1 exactly
for i, tq in enumerate(t_test):
    print(f"t = {tq}: C(0) = {preds_hard[i][0]:.6f}, C(1) = {preds_hard[i][-1]:.4f}")


# In[13]:


## Written by Claude
def r2_rmse(pred, true):
    rmse = np.sqrt(np.mean((pred - true)**2))
    r2 = 1 - np.sum((pred - true)**2) / np.sum((true - true.mean())**2)
    return r2, rmse

for name, preds, t_train in [('Soft', preds_soft, time_soft), ('Hard', preds_hard, time_hard)]:
    print(f"{name} PINN vs Crank–Nicolson (training time {t_train:.1f} s):")
    for i, tq in enumerate(t_test):
        r2, rmse = r2_rmse(preds[i], C_test[i])
        print(f"  t = {tq}: R² = {r2:.4f}, RMSE = {rmse:.5f}")
    r2_all, rmse_all = r2_rmse(preds.ravel(), C_test.ravel())
    print(f"  pooled:  R² = {r2_all:.4f}, RMSE = {rmse_all:.5f}")

# How well each model satisfies the conditions (evaluated on the training condition points)
# Soft: these are its trained penalty terms; Hard: should be ~0 by construction
print("\nCondition errors (mean squared):")
for name, m in [('Soft', pinn_soft_trained), ('Hard', pinn_hard_trained)]:
    m.eval()
    ic  = ic_loss(m, xt_ic).item()
    out = outlet_loss(m, xt_outlet.clone()).item()
    # inlet: compare to 1 for both models (hard enforces g(t), which is ~1 except for very small t)
    inl = inlet_loss(m, xt_inlet).item()
    print(f"  {name}: IC = {ic:.2e}, inlet (vs 1) = {inl:.2e}, outlet slope = {out:.2e}")


# In[14]:


## Written by Claude
fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True)
for i, tq in enumerate(t_test):
    ax = axes[0, i]
    ax.plot(x_test_np, C_test[i], 'k-', linewidth=3, label='Crank–Nicolson')
    ax.plot(x_test_np, preds_soft[i], 'r--', linewidth=2.2, label='PINN (soft)')
    ax.plot(x_test_np, preds_hard[i], 'b:', linewidth=2.6, label='PINN (hard)')
    ax.set_title(f'$t_D$ = {tq}'); ax.set_ylabel('$C_D$'); ax.grid(True, alpha=0.5)
    if i == 0: ax.legend()

    ax = axes[1, i]
    ax.plot(x_test_np, preds_soft[i] - C_test[i], 'r-', linewidth=1.8, label='soft − CN')
    ax.plot(x_test_np, preds_hard[i] - C_test[i], 'b-', linewidth=1.8, label='hard − CN')
    ax.axhline(0, color='k', linewidth=0.8)
    ax.set_xlabel('$x_D$'); ax.set_ylabel('error'); ax.grid(True, alpha=0.5)
    if i == 0: ax.legend()
plt.tight_layout()
plt.savefig(f'{save_dir}/soft_vs_hard_profiles.png', dpi=200, bbox_inches='tight')
plt.show()

# Training loss histories
plt.figure(figsize=(10, 5))
plt.semilogy(losses_soft['total'], 'r-', alpha=0.8, label='Soft: total')
plt.semilogy(losses_soft['pde'], 'r:', alpha=0.8, label='Soft: PDE')
plt.semilogy(losses_hard['total'], 'b-', alpha=0.8, label='Hard: total')
plt.semilogy(losses_hard['pde'], 'b:', alpha=0.8, label='Hard: PDE')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training loss')
plt.legend(); plt.grid(True, alpha=0.5)
plt.savefig(f'{save_dir}/soft_vs_hard_losses.png', dpi=200, bbox_inches='tight')
plt.show()

# Save single-seed results
np.savez(f'{save_dir}/results_seed42.npz', preds_soft=preds_soft, preds_hard=preds_hard,
         time_soft=time_soft, time_hard=time_hard, C_test=C_test, x_test=x_test_np, t_test=np.array(t_test))


# In[ ]:


