# Eschallot: Comprehensive Package for the Simulation and Optimization of Spherical Particles

Inspired by the needle optimization algorithm popular in multilayer film design **[1-3]**, we developed a topology optimization algorithm for multi-shell spherical particles. The algorithm can be conceptualized as **(a)** optimizing the layer boundary positions by gradient descent (*shape optimization*) and **(b)** optimizing the materials and the number of layers by inserting an infinitesimal needle layer at an optimal location (*topology nucleation*) in an alternating manner.

The above algorithm is combined with the transfer matrix method for the Mie scattering of multi-shell particles **[4-5]** to enable the efficient optimization of various far-field scattering quantities.

![](flowchart.png)

If you find this code helpful in your research, please consider citing:

(to be added)

**References:**

**[1]** A. V. Tikhonravov, M. K. Trubetskov, G. W. DeBell, Application of the needle optimization technique to the design of optical coatings, Appl. Opt. 35, 5493-5508 (1996).

**[2]** S. Larouche, L. Martinu, OpenFilters: open-source software for the design, optimization, and synthesis of optical filters, Appl. Opt. 47, C219-C230 (2008).

**[3]** M. Trubetskov, Deep search methods for multilayer coating design, 59, A75-A82 (2020).

**[4]** A. Moroz, A recursive transfer-matrix solution for a dipole radiating inside and outside a stratified sphere, Ann. Phys. 315, 352-418 (2005).

**[5]** I. Rasskazov, P. Carney, A. Moroz, STRATIFY: a comprehensive and versatile MATLAB code for a multilayered sphere, OSA Contin. 3, 2290 (2020).

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
