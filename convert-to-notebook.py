import json

# Create the notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add cells from the LaTeX content
cells = [
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [],
        "source": [
            "#importing_libraries\n",
            "import numpy as np\n",
            "from numpy import gradient\n",
            "import h5py\n",
            "#%matplotlib inline\n",
            "import matplotlib.pyplot as plt\n",
            "from scipy.interpolate import RegularGridInterpolator\n",
            "from scipy.integrate import odeint\n",
            "from mpl_toolkits.mplot3d import Axes3D\n",
            "from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection\n",
            "#christoffel_symbols with pytearcat\n",
            "import pytearcat as pt"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "$$ds^2 = a^2(\\tau)[-(1+2\\Phi)d\\tau^2 + (1-2\\Phi)\\delta_{ij}dx^i dx^j]$$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 2,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Define the coordinates to be used\n",
            "tau,x,y,z = pt.coords('tau,x,y,z')\n",
            "# Define any constant (more than one constant can be defined at the same time)\n",
            "a = pt.fun('a','tau')\n",
            "Phi = pt.fun('Phi','x,y,z')\n",
            "ds2 = 'ds2 = a**2*(-(1 + 2*Phi)*dtau**2 + (1 - 2*Phi)*(dx**2 + dy**2 + dz**2))'\n",
            "g = pt.metric(ds2)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 3,
        "metadata": {},
        "outputs": [],
        "source": [
            "# To calculate the second kind without the first kind:\n",
            "Chr = pt.christoffel(First_kind = False)\n",
            "\n",
            "# To display only a particular combination of indices, e.g., the Second kind:\n",
            "Chr.display(\"^,_,_\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "$$\\Phi(x, y, z) = \\frac{-m}{\\sqrt{x^2 + y^2 + z^2}}$$\n",
            "\n",
            "$$\\frac{\\partial\\Phi(x, y, z)}{\\partial x} = \\frac{mx}{(x^2 + y^2 + z^2)^{3/2}}$$\n",
            "\n",
            "$$\\frac{\\partial\\Phi(x, y, z)}{\\partial y} = \\frac{my}{(x^2 + y^2 + z^2)^{3/2}}$$\n",
            "\n",
            "$$\\frac{\\partial\\Phi(x, y, z)}{\\partial z} = \\frac{mz}{(x^2 + y^2 + z^2)^{3/2}}$$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 4,
        "metadata": {},
        "outputs": [],
        "source": [
            "m = 0.06\n",
            "\n",
            "def phi(x, y, z):\n",
            "    return -1.0*m / (np.sqrt(x**2 +y**2 +z**2))\n",
            "\n",
            "def dphi_dx(x, y, z):\n",
            "    return 1.0*m * x / ((x**2 +y**2 +z**2)**(3/2))\n",
            "\n",
            "def dphi_dy(x, y, z):\n",
            "    return 1.0*m * y / ((x**2 +y**2 +z**2)**(3/2))\n",
            "\n",
            "def dphi_dz(x, y, z):\n",
            "    return 1.0*m * z / ((x**2 +y**2 +z**2)**(3/2))"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 5,
        "metadata": {},
        "outputs": [],
        "source": [
            "#set_initial_conditions\n",
            "\n",
            "beta = np.pi/2\n",
            "alpha = 0.0\n",
            "tau0 = 0.0\n",
            "x0 = 300.0\n",
            "y0 = 10.0\n",
            "z0 = 10.0\n",
            "a0 = 1.0\n",
            "\n",
            "#initial_metric_components\n",
            "\n",
            "g00_0 = -a0**2*(1.0 + 2.0*(phi(x0, y0, z0)))    # g00_0 = -a0**2 *(1 + 2*phi)\n",
            "g11_0 = a0**2*(1.0 - 2.0*(phi(x0, y0, z0)))    # g11_0 = a0**2 *(1 - 2*phi)\n",
            "g22_0 = a0**2*(1.0 - 2.0*(phi(x0, y0, z0)))    # g22_0 = a0**2 *(1 - 2*phi)\n",
            "g33_0 = a0**2*(1.0 - 2.0*(phi(x0, y0, z0)))    # g33_0 = a0**2 *(1 - 2*phi)\n",
            "\n",
            "ktau_0 = 1 / np.sqrt(np.abs(g00_0))    # k_tau_0 = (dtau/dl)_0 = |g00_0|**(-0.5)\n",
            "kx_0 = np.cos(alpha)*np.sin(beta) / np.sqrt(np.abs(g11_0))  # k_x_0 = (dx/dl)_0 = |g11_0|**(-0.5)*cos(alpha)*sin(beta)\n",
            "ky_0 = np.sin(beta)*np.sin(alpha) / np.sqrt(np.abs(g22_0))  # k_y_0 = (dy/dl)_0 = |g22_0|**(-0.5)*sin(alpha)*sin(beta)\n",
            "kz_0 = np.cos(beta) / np.sqrt(np.abs(g33_0))    # k_z_0 = (dz/dl)_0 = |g33_0|**(-0.5)*cos(beta)\n",
            "\n",
            "h0 = [tau0, ktau_0, x0, kx_0, y0, ky_0, z0, kz_0, a0]"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "\\begin{align}\n",
            "  \\Phi_0 &= \\Phi(x_0, y_0, z_0) \\\\\n",
            " (g_{00})_0 &= -a_0^2[1+2\\Phi_0] \\\\\n",
            " (g_{11})_0 &= a_0^2[1-2\\Phi_0] \\\\\n",
            " (g_{22})_0 &= a_0^2[1-2\\Phi_0] \\\\\n",
            " (g_{33})_0 &= a_0^2[1-2\\Phi_0]\n",
            "\\end{align}\n",
            "\n",
            "\\begin{align}\n",
            "  (k^\\tau)_0 & = \\left. \\frac{d\\tau}{d\\lambda} \\right|_0 = \\frac{1}{\\sqrt{|(g_{00})_0|}} = \\frac{1}{a_0\\sqrt{1+2\\Phi_0}} \\\\\n",
            "  (k^x)_0 &= \\left. \\frac{dx}{d\\lambda} \\right|_0 = \\frac{\\cos\\alpha\\sin\\beta}{\\sqrt{|(g_{11})_0|}} = \\frac{\\cos\\alpha\\sin\\beta}{a_0\\sqrt{1-2\\Phi_0}} \\\\\n",
            "  (k^y)_0 &= \\left. \\frac{dy}{d\\lambda} \\right|_0  = \\frac{\\sin\\alpha\\sin\\beta}{\\sqrt{|(g_{22})_0|}} = \\frac{\\sin\\alpha\\sin\\beta}{a_0\\sqrt{1-2\\Phi_0}} \\\\\n",
            "  (k^z)_0 &= \\left. \\frac{dz}{d\\lambda} \\right|_0 = \\frac{\\cos\\beta}{\\sqrt{|(g_{33})_0|}} = \\frac{\\cos\\beta}{a_0\\sqrt{1-2\\Phi_0}}\n",
            "\\end{align}"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 6,
        "metadata": {},
        "outputs": [],
        "source": [
            "#cosmological_parameters\n",
            "\n",
            "wm = 1.0 # 0.31205  -- Omega_m  = matter related\n",
            "wd = 0.0 # 0.68795  -- Omega_Lambda = dark energy related\n",
            "\n",
            "#Light can travel 1 Mpc distance in 3.26*10^6 year. Here, H0 = 67.556 Km s^-1 Mpc^-1 = 0.070894407 /Gyr.\n",
            "\n",
            "H0 = (0.070894407 * 3.26)/(1000 * 0.67556) # unit in h/Mpc"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 7,
        "metadata": {},
        "outputs": [],
        "source": [
            "#define function\n",
            "\n",
            "def geodesic(h, u):\n",
            "    tau, ktau, x, kx, y, ky, z, kz, a = h       # ktau = dtau/dl, kx = dx/dl, ky = dy/dl, kz = dz/dl\n",
            "\n",
            "    #Friedmann_equation(conformal_time_dependent)\n",
            "\n",
            "    dadtau = H0*a**2 *np.sqrt((wm/a**3) + wd)\n",
            "\n",
            "    #christoffel_symbols\n",
            "\n",
            "    gamma_000 = (dadtau/a)\n",
            "    gamma_001 = dphi_dx(x, y, z)/(2*phi(x, y, z)+1)\n",
            "    gamma_002 = dphi_dy(x, y, z)/(2*phi(x, y, z)+1)\n",
            "    gamma_003 = dphi_dz(x, y, z)/(2*phi(x, y, z)+1)\n",
            "    gamma_010 = gamma_001\n",
            "    gamma_011 = -((2*phi(x,y,z)-1)*(dadtau/a))/((2*phi(x,y,z)+1)*a)\n",
            "    gamma_020 = gamma_002\n",
            "    gamma_022 = -((2*phi(x,y,z)-1)*(dadtau/a))/((2*phi(x,y,z)+1)*a)\n",
            "    gamma_030 = gamma_003\n",
            "    gamma_033 = -((2*phi(x,y,z)-1)*(dadtau/a))/((2*phi(x,y,z)+1)*a)\n",
            "\n",
            "    ################################\n",
            "\n",
            "    gamma_100 = dphi_dx(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_101 = (dadtau/a)\n",
            "    gamma_110 = gamma_101\n",
            "    gamma_111 = dphi_dx(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_112 = dphi_dy(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_113 = dphi_dz(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_121 = gamma_112\n",
            "    gamma_122 = -dphi_dx(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_131 = gamma_113\n",
            "    gamma_133 = -dphi_dx(x,y,z)/(2*phi(x,y,z)-1)\n",
            "\n",
            "    ################################\n",
            "\n",
            "    gamma_200 = -dphi_dx(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_202 = (dadtau/a)\n",
            "    gamma_211 = -dphi_dy(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_212 = dphi_dx(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_220 = gamma_202\n",
            "    gamma_221 = gamma_212\n",
            "    gamma_222 = dphi_dy(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_223 = dphi_dz(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_232 = gamma_223\n",
            "    gamma_233 = -dphi_dy(x,y,z)/(2*phi(x,y,z)-1)\n",
            "\n",
            "    ################################\n",
            "\n",
            "    gamma_300 = -dphi_dz(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_303 = (dadtau/a)\n",
            "    gamma_311 = -dphi_dz(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_313 = dphi_dx(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_322 = -dphi_dz(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_323 = dphi_dy(x,y,z)/(2*phi(x,y,z)-1)\n",
            "    gamma_330 = gamma_303\n",
            "    gamma_331 = gamma_313\n",
            "    gamma_332 = gamma_323\n",
            "    gamma_333 = dphi_dz(x,y,z)/(2*phi(x,y,z)-1)\n",
            "\n",
            "    #Solving_geodesic_equations\n",
            "\n",
            "    dhdl = [\n",
            "        ktau,\n",
            "        -1.0*(gamma_000*ktau**2 + 2.0*gamma_001*ktau*kx + 2.0*gamma_002*ktau*ky + 2.0*gamma_003*ktau*kz + gamma_011*kx**2 + gamma_022*ky**2 + gamma_033*kz**2),\n",
            "        kx,\n",
            "        -1.0*(gamma_100*ktau**2 + 2.0*gamma_110*kx*ktau + 2.0*gamma_112*kx*ky + 2.0*gamma_113*kx*kz + gamma_111*kx**2 + gamma_122*ky**2 + gamma_133*kz**2),\n",
            "        ky,\n",
            "        -1.0*(gamma_200*ktau**2 + 2.0*gamma_220*ky*ktau + 2.0*gamma_221*ky*kx + 2.0*gamma_223*ky*kz + gamma_211*kx**2 + gamma_222*ky**2 + gamma_233*kz**2),\n",
            "        kz,\n",
            "        -1.0*(gamma_300*ktau**2 + 2.0*gamma_330*kz*ktau + 2.0*gamma_331*kz*kx + 2.0*gamma_332*kz*ky + gamma_311*kx**2 + gamma_322*ky**2 + gamma_333*kz**2),\n",
            "        ktau*H0*a**2*(np.sqrt((wm/a**3) + wd))\n",
            "    ]\n",
            "    return dhdl"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "$$ \\frac{dh}{d\\lambda} = \\left\\{ \\begin{aligned}\n",
            "  & \\frac{d\\tau}{d\\lambda} = k^\\tau  \\\\\n",
            "  & \\frac{d^2\\tau}{d\\lambda^2} = \\frac{dk^\\tau}{d\\lambda} = -\\left[ \\Gamma^0_{00} (k^\\tau)^2 + \\Gamma^0_{01} k^tk^x + \\Gamma^0_{02} k^tk^y + \\Gamma^0_{03} k^tk^z + \\Gamma^0_{11} (k^x)^2 + \\Gamma^0_{22} (k^y)^2 + \\Gamma^0_{33} (k^z)^2\\right] \\\\\n",
            "  & \\frac{dx}{d\\lambda} = k^x \\\\\n",
            "  & \\frac{d^2x}{d\\lambda^2} = \\frac{dk^x}{d\\lambda} = -\\left[ \\Gamma^1_{00} (k^\\tau)^2 + \\Gamma^1_{10} k^xk^t + \\Gamma^1_{12} k^xk^y + \\Gamma^1_{13} k^xk^z + \\Gamma^1_{11} (k^x)^2 + \\Gamma^1_{22} (k^y)^2 + \\Gamma^1_{33} (k^z)^2\\right] \\\\\n",
            "  & \\frac{dy}{d\\lambda} = k^y \\\\\n",
            "  & \\frac{d^2y}{d\\lambda^2} = \\frac{dk^y}{d\\lambda} = -\\left[ \\Gamma^2_{00} (k^\\tau)^2 + \\Gamma^2_{20} k^yk^t + \\Gamma^2_{21} k^yk^x + \\Gamma^2_{23} k^yk^z + \\Gamma^2_{11} (k^x)^2 + \\Gamma^2_{22} (k^y)^2 + \\Gamma^2_{33} (k^z)^2\\right] \\\\\n",
            "  & \\frac{dz}{d\\lambda} = k^z \\\\\n",
            "  & \\frac{d^2z}{d\\lambda^2} = \\frac{dk^z}{d\\lambda} = -\\left[ \\Gamma^2_{00} (k^\\tau)^2 + \\Gamma^2_{20} k^yk^t + \\Gamma^2_{21} k^yk^x + \\Gamma^2_{23} k^yk^z + \\Gamma^2_{11} (k^x)^2 + \\Gamma^2_{22} (k^y)^2 + \\Gamma^2_{33} (k^z)^2\\right] \\\\\n",
            "  & \\frac{da}{d\\lambda} = k^\\tau H_0 a^2 \\sqrt{\\Omega_m a^{-3}+\\Omega_\\Lambda}\n",
            "\\end{aligned}\n",
            "\\right. $$"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 8,
        "metadata": {},
        "outputs": [],
        "source": [
            "#linear_space_of_affine_parameter (l -> lambda)\n",
            "\n",
            "l = np.linspace(0, -210, 1000)\n",
            "\n",
            "#atol=1e-15"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 9,
        "metadata": {},
        "outputs": [],
        "source": [
            "#calling_integrator\n",
            "\n",
            "sol = odeint(geodesic, h0, l)\n",
            "\n",
            "# sol[:, 0] = tau, sol[:, 1] = k_tau, sol[:, 2] = x, sol[:, 3] = k_x, sol[:, 4] = y, sol[:, 5] = k_y, sol[:, 6] = z, sol[:, 7] = k_z, sol[:, 8] = a"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 10,
        "metadata": {},
        "outputs": [],
        "source": [
            "#Metric_component\n",
            "\n",
            "phi_metric = [phi(sol[i, 2], sol[i, 4], sol[i, 6]) for i in range(0, 1000)]\n",
            "\n",
            "phi_xyz = np.reshape(phi_metric, (1000, 1))\n",
            "\n",
            "g00 = -sol[:,8]**2 *(1.0 + 2*phi_xyz)\n",
            "g11 = sol[:,8]**2 *(1.0 - 2*phi_xyz)\n",
            "g22 = sol[:,8]**2 *(1.0 - 2*phi_xyz)\n",
            "g33 = sol[:,8]**2 *(1.0 - 2*phi_xyz)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 11,
        "metadata": {},
        "outputs": [],
        "source": [
            "#null_vector_condition\n",
            "\n",
            "null_condition = g00*sol[:,1]**2 + g11*sol[:,3]**2 + g22*sol[:,5]**2 + g33*sol[:,7]**2\n",
            "null_condition"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 12,
        "metadata": {},
        "outputs": [],
        "source": [
            "#Energy_of_photon\n",
            "\n",
            "E = sol[:, 1]\n",
            "\n",
            "#Redshift\n",
            "\n",
            "z = (E*sol[:, 8]) - 1"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 13,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig = plt.figure()\n",
            "ax = fig.add_subplot(projection = '3d')\n",
            "ax.plot(sol[:,2], sol[:,4], sol[:,6])\n",
            "ax.set_xlabel('X Label')\n",
            "ax.set_ylabel('Y Label')\n",
            "ax.set_zlabel('Z Label')\n",
            "plt.title('The 3D path of photon (initially projected in X-direction)')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 14,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.plot(l, np.abs(null_condition), ls='None', marker='x', c='r')\n",
            "plt.yscale('log')\n",
            "plt.xlabel('Affine parameter')\n",
            "plt.ylabel('Null vector condition')\n",
            "plt.title('Affine parameter vs Null vector condition (initially projected in X-direction)')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 15,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.plot(sol[:, 8], z, ls='None', marker='x', c='r')\n",
            "plt.xlabel('Scale factor')\n",
            "plt.ylabel('Redshift')\n",
            "plt.title('Scale factor vs Redshift (initially projected in X-direction)')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 16,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.plot(sol[:,2], sol[:,4], 'r')\n",
            "plt.xlabel('X')\n",
            "plt.ylabel('Y')\n",
            "plt.title('The trajectory of photon along XY plane (initially projected in X-direction)')\n",
            "plt.show()"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 17,
        "metadata": {},
        "outputs": [],
        "source": [
            "plt.plot(sol[:,2], sol[:,6], 'r')\n",
            "plt.xlabel('X')\n",
            "plt.ylabel('Z')\n",
            "plt.title('The trajectory of photon along XZ plane (initially projected in X-direction)')\n",
            "plt.show()"
        ]
    }
]

# Add all cells to the notebook
notebook["cells"] = cells

# Save the notebook as a .ipynb file
with open("null-geodesics.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)

print("Jupyter Notebook created successfully: null-geodesics.ipynb")
