# %% [markdown]
# Null Geodesics Calculation
# This notebook calculates null geodesics in a perturbed FLRW metric with a Newtonian potential.
# It computes the photon trajectory, checks the null condition, and visualizes the results.

# %% [markdown]
# Import Libraries
# Import required libraries for numerical calculations, integration, and plotting.

import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D

# %% [markdown]
# Define Potential and Derivatives
# Define the Newtonian potential and its spatial derivatives with safeguards.

m = 0.06
r_min = 1e-4  # Minimum radius to avoid singularities

def phi(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r = max(r, r_min)
    return -m / r

def dphi_dx(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r = max(r, r_min)
    return m * x / (r**3)

def dphi_dy(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r = max(r, r_min)
    return m * y / (r**3)

def dphi_dz(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    r = max(r, r_min)
    return m * z / (r**3)

# %% [markdown]
# Cosmological Parameters and Scale Factor
# Define cosmological parameters and scale factor derivative with bounds.

wm = 1.0  # Omega_m = matter density
wd = 0.3  # Dark energy for stability
H0 = 0.01  # Reduced Hubble constant
a_min = 1e-4  # Minimum scale factor
a_max = 1e4   # Maximum scale factor

def dadtau(a_val):
    a_val = np.clip(a_val, a_min, a_max)
    return H0 * a_val**2 * np.sqrt(wm/a_val**3 + wd)

# %% [markdown]
# Christoffel Symbols
# Calculate Christoffel symbols for the metric ds² = -a²(1+2φ)dτ² + a²(1-2φ)(dx² + dy² + dz²).

def calculate_christoffel(tau, x, y, z, a_val, ktau, kx, ky, kz):
    """Calculate Christoffel symbols for the given metric"""
    a_val = np.clip(a_val, a_min, a_max)
    phi_val = phi(x, y, z)
    dphidx = dphi_dx(x, y, z)
    dphidy = dphi_dy(x, y, z)
    dphidz = dphi_dz(x, y, z)
    dadtau_val = dadtau(a_val)
    
    gamma = {}
    # Γ^τ_{μν}
    gamma['000'] = dadtau_val/a_val
    gamma['001'] = dphidx/(1 + 2*phi_val)
    gamma['002'] = dphidy/(1 + 2*phi_val)
    gamma['003'] = dphidz/(1 + 2*phi_val)
    gamma['011'] = (1 - 2*phi_val)*dadtau_val/(a_val*(1 + 2*phi_val))
    gamma['022'] = gamma['011']
    gamma['033'] = gamma['011']
    
    # Γ^x_{μν}
    gamma['100'] = dphidx/(1 - 2*phi_val)
    gamma['101'] = dadtau_val/a_val
    gamma['111'] = -dphidx/(1 - 2*phi_val)
    gamma['112'] = -dphidy/(1 - 2*phi_val)
    gamma['113'] = -dphidz/(1 - 2*phi_val)
    gamma['122'] = dphidx/(1 - 2*phi_val)
    gamma['133'] = dphidx/(1 - 2*phi_val)
    
    # Γ^y_{μν}
    gamma['200'] = dphidy/(1 - 2*phi_val)
    gamma['202'] = dadtau_val/a_val
    gamma['211'] = dphidy/(1 - 2*phi_val)
    gamma['212'] = -dphidx/(1 - 2*phi_val)
    gamma['222'] = -dphidy/(1 - 2*phi_val)
    gamma['223'] = -dphidz/(1 - 2*phi_val)
    gamma['233'] = dphidy/(1 - 2*phi_val)
    
    # Γ^z_{μν}
    gamma['300'] = dphidz/(1 - 2*phi_val)
    gamma['303'] = dadtau_val/a_val
    gamma['311'] = dphidz/(1 - 2*phi_val)
    gamma['313'] = -dphidx/(1 - 2*phi_val)
    gamma['322'] = dphidz/(1 - 2*phi_val)
    gamma['323'] = -dphidy/(1 - 2*phi_val)
    gamma['333'] = -dphidz/(1 - 2*phi_val)
    
    return gamma

# %% [markdown]
# Initial Conditions
# Set initial conditions for backward integration to get positive redshift.

beta = np.pi/4  # Avoids origin
alpha = np.pi/2
tau0 = 0.0
x0 = 300.0
y0 = 10.0
z0 = 10.0
a0 = 1.0

phi0 = phi(x0, y0, z0)
g00_0 = -a0**2*(1.0 + 2.0*phi0)
g11_0 = a0**2*(1.0 - 2.0*phi0)
g22_0 = a0**2*(1.0 - 2.0*phi0)
g33_0 = a0**2*(1.0 - 2.0*phi0)

# For null geodesics: g00 (k^tau)^2 + g11 (k^x)^2 + g22 (k^y)^2 + g33 (k^z)^2 = 0
ktau_0 = -1.0  # Backward integration
norm_factor = np.sqrt(-g00_0 / g11_0)
kx_0 = np.cos(alpha)*np.sin(beta) * norm_factor
ky_0 = np.sin(alpha)*np.sin(beta) * norm_factor
kz_0 = np.cos(beta) * norm_factor

h0 = [tau0, ktau_0, x0, kx_0, y0, ky_0, z0, kz_0, a0]

# %% [markdown]
# Geodesic Equations
# Define geodesic equations with safeguards.

def geodesic(u, h):
    tau, ktau, x, kx, y, ky, z, kz, a_val = h
    a_val = np.clip(a_val, a_min, a_max)
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < r_min:
        return [0.0] * 9  # Halt integration near origin
    
    gamma = calculate_christoffel(tau, x, y, z, a_val, ktau, kx, ky, kz)
    
    dtau_dl = ktau
    dktau_dl = -(gamma['000']*ktau**2 + 2*gamma['001']*ktau*kx + 2*gamma['002']*ktau*ky + 
                 2*gamma['003']*ktau*kz + gamma['011']*kx**2 + gamma['022']*ky**2 + gamma['033']*kz**2)
    
    dx_dl = kx
    dkx_dl = -(gamma['100']*ktau**2 + 2*gamma['101']*ktau*kx + 2*gamma['112']*kx*ky + 
               2*gamma['113']*kx*kz + gamma['111']*kx**2 + gamma['122']*ky**2 + gamma['133']*kz**2)
    
    dy_dl = ky
    dky_dl = -(gamma['200']*ktau**2 + 2*gamma['202']*ktau*ky + 2*gamma['212']*kx*ky + 
               2*gamma['223']*ky*kz + gamma['211']*kx**2 + gamma['222']*ky**2 + gamma['233']*kz**2)
    
    dz_dl = kz
    dkz_dl = -(gamma['300']*ktau**2 + 2*gamma['303']*ktau*kz + 2*gamma['313']*kx*kz + 
               2*gamma['323']*ky*kz + gamma['311']*kx**2 + gamma['322']*ky**2 + gamma['333']*kz**2)
    
    da_dl = dadtau(a_val) * ktau
    
    return [dtau_dl, dktau_dl, dx_dl, dkx_dl, dy_dl, dky_dl, dz_dl, dkz_dl, da_dl]

# %% [markdown]
# Integration Setup
# Use solve_ivp with Radau method, moderate range for general exploration.

l_span = (0, 25)  # Moderate range for exploration, no specific redshift target
l_eval = np.linspace(0, 25, 1000)
sol = solve_ivp(geodesic, l_span, h0, method='Radau', t_eval=l_eval, rtol=1e-8, atol=1e-10)
if not sol.success:
    print(f"Integration failed: {sol.message}")
sol_y = sol.y.T
n = len(sol_y)
l = l_eval[:n]

# %% [markdown]
# ## Compute Metric and Null Condition
# Calculate metric components and null condition.

phi_metric = np.array([phi(sol_y[i, 2], sol_y[i, 4], sol_y[i, 6]) for i in range(n)])
g00 = -sol_y[:, 8]**2 * (1.0 + 2*phi_metric)
g11 = sol_y[:, 8]**2 * (1.0 - 2*phi_metric)
g22 = sol_y[:, 8]**2 * (1.0 - 2*phi_metric)
g33 = sol_y[:, 8]**2 * (1.0 - 2*phi_metric)

null_condition = g00*sol_y[:,1]**2 + g11*sol_y[:,3]**2 + g22*sol_y[:,5]**2 + g33*sol_y[:,7]**2

# %% [markdown]
# Redshift Calculation
# Compute redshift with safeguards.

redshift = np.where(sol_y[:, 8] > a_min, 1.0 / sol_y[:, 8] - 1.0, np.nan)

# %% [markdown]
# Minimum Distance Check
# Calculate minimum distance to origin.

r_values = np.sqrt(sol_y[:, 2]**2 + sol_y[:, 4]**2 + sol_y[:, 6]**2)
min_r = np.min(r_values)

# %% [markdown]
# Visualization
# Plot results with clipping for stability and adjusted ranges for higher redshift.

fig = plt.figure(figsize=(15, 10))

# Null condition check - log scale
ax1 = fig.add_subplot(231)
ax1.plot(l, np.clip(np.abs(null_condition), 0, 1e10), 'b-', linewidth=2)
ax1.set_yscale('log')
ax1.set_xlabel('Affine Parameter λ')
ax1.set_ylabel('|gₘₙkᵐkⁿ|')
ax1.set_title('Null Condition Check (Log Scale)')
ax1.grid(True, which="both", ls="--")
ax1.set_ylim(1e-16, 1e-12)

# Null condition check - linear scale
ax2 = fig.add_subplot(232)
ax2.plot(l, np.clip(null_condition, -1e10, 1e10), 'r-', linewidth=2)
ax2.set_xlabel('Affine Parameter λ')
ax2.set_ylabel('gₘₙkᵐkⁿ')
ax2.set_title('Null Condition (Linear Scale)')
ax2.grid(True, ls="--")
ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax2.set_ylim(-1e-12, 1e-12)

# Scale factor vs redshift
ax3 = fig.add_subplot(233)
valid = ~np.isnan(redshift)
ax3.plot(sol_y[valid, 8], redshift[valid], 'g-', linewidth=2)
ax3.set_xlabel('Scale Factor a')
ax3.set_ylabel('Redshift z')
ax3.set_title('Scale Factor vs Redshift')
ax3.grid(True, ls="--")
ax3.set_xlim(0.5, 1.0)  # Adjusted for potential z ≈ 0.6–1.0
ax3.set_ylim(0.0, 1.0)

# 3D trajectory
ax4 = fig.add_subplot(234, projection='3d')
ax4.plot(sol_y[:,2], sol_y[:,4], sol_y[:,6], linewidth=2)
ax4.scatter([sol_y[0,2]], [sol_y[0,4]], [sol_y[0,6]], color='red', s=50, label='Start')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title('3D Photon Trajectory')
ax4.legend()

# XY plane projection
ax5 = fig.add_subplot(235)
ax5.plot(sol_y[:,2], sol_y[:,4], 'purple', linewidth=2)
ax5.scatter([sol_y[0,2]], [sol_y[0,4]], color='red', s=50, label='Start')
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_title('XY Plane Projection')
ax5.grid(True, ls="--")
ax5.legend()

# XZ plane projection
ax6 = fig.add_subplot(236)
ax6.plot(sol_y[:,2], sol_y[:,6], 'orange', linewidth=2)
ax6.scatter([sol_y[0,2]], [sol_y[0,6]], color='red', s=50, label='Start')
ax6.set_xlabel('X')
ax6.set_ylabel('Z')
ax6.set_title('XZ Plane Projection')
ax6.grid(True, ls="--")
ax6.legend()

plt.tight_layout()
plt.show()

# %% [markdown]
# Diagnostics
# Print diagnostics with nan/inf handling.

print("=== NULL GEODESIC DIAGNOSTICS ===")
print(f"Initial null condition: {null_condition[0]:.2e}")
print(f"Final null condition: {null_condition[-1]:.2e}")
print(f"Average null condition: {np.nanmean(null_condition):.2e}")
print(f"Maximum absolute null condition: {np.nanmax(np.abs(null_condition)):.2e}")
print(f"Standard deviation of null condition: {np.nanstd(null_condition):.2e}")
print(f"Initial redshift: {redshift[0]:.6f}")
print(f"Final redshift: {redshift[-1]:.6f}")
print(f"Scale factor change: {sol_y[0, 8]:.6f} → {sol_y[-1, 8]:.6f}")
print(f"Time coordinate change: {sol_y[0, 0]:.2f} → {sol_y[-1, 0]:.2f}")
print(f"Minimum distance to origin: {min_r:.6f}")
print(f"Integration steps taken: {sol.nfev}")
print(f"Integration status: {sol.message}")
