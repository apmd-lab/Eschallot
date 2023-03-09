# Eschallot: Comprehensive Package for the Simulation and Optimization of Spherical Particles

This package implements transfer matrix methods for Mie scattering of multi-shell particles **[1,2]**

Optimization can be done by:

**1. Needle optimization** (simultaneously optimize boundary positions, materials, and the number of layers)

**2. Gradient descent** (optimize boundary positions only with fixed materials and the number of layers)

The needle optimization algorithm is a popular method in multilayer film design **[3-5]** and can be conceptualized as **(a)** optimizing the boundary positions by gradient descent (*shape optimization*) and **(b)** optimizing the materials and the number of layers by inserting an infinitesimal needle layer at an optimal location (*topology optimization*) in an alternating manner.

![](flowchart.png)

Refer to this paper for more details.

**References:**

**[1]** A. Moroz, A recursive transfer-matrix solution for a dipole radiating inside and outside a stratified sphere, Ann. Phys. 315, 352-418 (2005).

**[2]** I. Rasskazov, P. Carney, A. Moroz, STRATIFY: a comprehensive and versatile MATLAB code for a multilayered sphere, OSA Contin. 3, 2290 (2020).

**[3]** A. V. Tikhonravov, M. K. Trubetskov, G. W. DeBell, Application of the needle optimization technique to the design of optical coatings, Appl. Opt. 35, 5493-5508 (1996).

**[4]** S. Larouche, L. Martinu, OpenFilters: open-source software for the design, optimization, and synthesis of optical filters, Appl. Opt. 47, C219-C230 (2008).

**[5]** M. Trubetskov, Deep search methods for multilayer coating design, 59, A75-A82 (2020).

# Quickstart

**Compute efficiencies and the phase function:** `simulate_particle.py`

**Run needle optimization:** `run_needle_optimization.py`

- To define a custom cost function, change the variables in 'radius_sweep'

- To define a custom range for the initial particle radius, change the inputs to 'radius_sweep' (under "if __name__ == '__main__'")

**Run gradient descent:** `run_gradient_optimization.py`

# Requirements

- NumPy

- SciPy

- Matplotlib

- Numba
