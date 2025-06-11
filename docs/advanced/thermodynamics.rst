===============================
Thermodynamics and Free Energy
===============================

Introduction
============

Thermodynamics provides the fundamental framework for understanding molecular processes and relating simulation results to experimental measurements. Free energy calculations are among the most important applications of molecular dynamics, providing quantitative predictions of binding affinities, solubilities, and reaction equilibria.

**Key Thermodynamic Quantities:**

- **Free Energy (A, G)**: Determines spontaneous processes and equilibrium
- **Enthalpy (H)**: Heat content and bond formation energy
- **Entropy (S)**: Disorder and accessible microstates
- **Chemical Potential (μ)**: Driving force for particle transfer

**Relationship to Simulation:**

.. math::

   G = H - TS = U + PV - TS

where all quantities can be computed from molecular simulations.

Fundamental Thermodynamic Relations
===================================

Maxwell Relations
----------------

**Thermodynamic Potentials:**

Internal Energy: :math:`dU = TdS - PdV + \mu dN`

Helmholtz Free Energy: :math:`dA = -SdT - PdV + \mu dN`

Gibbs Free Energy: :math:`dG = -SdT + VdP + \mu dN`

**Maxwell Relations:**

.. math::

   \left(\frac{\partial T}{\partial V}\right)_S = -\left(\frac{\partial P}{\partial S}\right)_V

.. math:

   \left(\frac{\partial S}{\partial V}\right)_T = \left(\frac{\partial P}{\partial T}\right)_V

These relations connect different thermodynamic derivatives and enable calculation of unmeasurable quantities from accessible ones.

Response Functions
-----------------

**Heat Capacity:**

.. math::

   C_V = \left(\frac{\partial U}{\partial T}\right)_V = \frac{\langle (\Delta E)^2 \rangle}{k_B T^2}

**Isothermal Compressibility:**

.. math:

   \kappa_T = -\frac{1}{V}\left(\frac{\partial V}{\partial P}\right)_T = \frac{\langle (\Delta V)^2 \rangle}{k_B T \langle V \rangle}

**Thermal Expansion Coefficient:**

.. math:

   \alpha = \frac{1}{V}\left(\frac{\partial V}{\partial T}\right)_P = \frac{\langle \Delta V \Delta H \rangle}{k_B T^2 \langle V \rangle}

Free Energy Methods
===================

Free Energy Perturbation (FEP)
------------------------------

**Basic Theory:**

Consider transformation from state A to state B:

.. math:

   \Delta A = A_B - A_A = -k_B T \ln \frac{Q_B}{Q_A}

**FEP Formula:**

.. math:

   \Delta A = -k_B T \ln \langle e^{-\beta \Delta U} \rangle_A

where :math:`\Delta U = U_B - U_A` and the average is over configurations from state A.

**Exponential Averaging Problem:**

FEP requires overlap between distributions. For large perturbations, a few high-energy configurations dominate the average, leading to poor convergence.

**Staging Strategy:**

Break large perturbation into smaller steps:

.. math:

   \Delta A = \sum_{i=0}^{n-1} \Delta A_{i \to i+1}

**Bidirectional FEP:**

Compute both forward and backward perturbations as consistency check:

.. math::

   \Delta A_{A \to B} = -\Delta A_{B \to A}

Thermodynamic Integration (TI)
-----------------------------

**Coupling Parameter Approach:**

Introduce parameter λ to connect states A (λ=0) and B (λ=1):

.. math:

   U(\lambda) = (1-\lambda)U_A + \lambda U_B

**TI Formula:**

.. math:

   \Delta A = \int_0^1 \left\langle \frac{\partial U(\lambda)}{\partial \lambda} \right\rangle_\lambda d\lambda

**Advantages over FEP:**
- Linear averaging instead of exponential
- Better numerical behavior
- Easier error analysis
- More robust convergence

**Implementation Details:**

Choose λ values (typically 11-21 points):

.. code-block:: python

   lambda_values = np.linspace(0, 1, 21)
   dudl_values = []
   
   for lam in lambda_values:
       dudl = simulate_and_calculate_dudl(lam)
       dudl_values.append(dudl)
   
   delta_A = np.trapz(dudl_values, lambda_values)

**Integration Methods:**
- Trapezoidal rule (most common)
- Simpson's rule (higher accuracy)
- Gaussian quadrature (optimal spacing)

Alchemical Transformations
=========================

Particle Insertion/Deletion
---------------------------

**Soft-Core Potentials:**

Standard Lennard-Jones potential becomes singular at r=0. Use soft-core form:

.. math::

   U_{sc}(r,\lambda) = 4\epsilon\lambda^n \left[\frac{1}{(\alpha(1-\lambda)^m + (r/\sigma)^6)^2} - \frac{1}{\alpha(1-\lambda)^m + (r/\sigma)^6}\right]

**Benefits:**
- Prevents numerical instabilities
- Smooth λ-dependence
- Better overlap between λ states

Charge Transformation
--------------------

**Linear Scaling:**

.. math:

   q_i(\lambda) = (1-\lambda)q_i^A + \lambda q_i^B

**Electrostatic Component:**

.. math:

   \frac{\partial U_{elec}}{\partial \lambda} = \sum_i \sum_{j>i} \frac{(q_i^B - q_i^A)(q_j^B - q_j^A)}{4\pi\epsilon_0 r_{ij}}

**Long-range Corrections:**

Electrostatic interactions require careful treatment with Ewald summation:

.. math:

   \frac{\partial U_{Ewald}}{\partial \lambda} = \frac{\partial U_{real}}{\partial \lambda} + \frac{\partial U_{reciprocal}}{\partial \lambda} + \frac{\partial U_{self}}{\partial \lambda}

Binding Free Energy Calculations
================================

Absolute Binding Free Energies
------------------------------

**Thermodynamic Cycle:**

.. math:

   \Delta G_{bind} = \Delta G_{complex} - \Delta G_{ligand} - \Delta G_{protein}

**Challenges:**
- Large configurational changes
- Long correlation times
- Standard state corrections

**Restraint-Based Methods:**

Use restraints to maintain binding pose during alchemical transformation:

.. math::

   \Delta G_{bind} = \Delta G_{decouple} + \Delta G_{restraint} + \Delta G_{standard}

Relative Binding Free Energies
------------------------------

**Double Decoupling:**

Compute difference in binding free energies for two ligands:

.. math:

   \Delta \Delta G = \Delta G_{B,bound} - \Delta G_{A,bound} - (\Delta G_{B,free} - \Delta G_{A,free})

**Advantages:**
- Many systematic errors cancel
- More accurate than absolute calculations
- Widely used in drug design

**Implementation:**

.. code-block:: python

   def relative_binding_free_energy(ligand_A, ligand_B):
       # Transform A to B in complex
       dG_complex = thermodynamic_integration(
           complex_with_A, complex_with_B)
       
       # Transform A to B in solution
       dG_solution = thermodynamic_integration(
           ligand_A_solvated, ligand_B_solvated)
       
       return dG_complex - dG_solution

Solvation Free Energies
=======================

Hydration Free Energy
--------------------

**Definition:**

Free energy to transfer molecule from gas phase to aqueous solution:

.. math:

   \Delta G_{hyd} = G_{solution} - G_{gas}

**Computational Approach:**

1. Turn off all interactions in solution
2. Typically separate van der Waals and electrostatic contributions
3. Use soft-core potentials for numerical stability

**Typical Protocol:**

.. math:

   \Delta G_{hyd} = \Delta G_{elec} + \Delta G_{vdW}

where electrostatics are turned off first, then van der Waals interactions.

Partition Coefficients
----------------------

**Octanol-Water Partition:**

.. math:

   \log P = \frac{\Delta G_{water} - \Delta G_{octanol}}{k_B T \ln 10}

**Applications:**
- Drug ADMET properties
- Environmental fate modeling
- Membrane permeability prediction

Advanced Free Energy Methods
============================

Multistate Reweighting
----------------------

**Bennett Acceptance Ratio (BAR):**

Optimal estimator for free energy difference between two states:

.. math:

   \Delta A = k_B T \ln \frac{Q_1}{Q_0} + k_B T \ln \frac{\langle f(U_1 - U_0 + C) \rangle_0}{\langle f(U_0 - U_1 - C) \rangle_1}

where :math:`f(x) = 1/(1 + e^x)` is the Fermi function.

**Multistate BAR (MBAR):**

Extends BAR to multiple states simultaneously:

.. math:

   \hat{A}_i = -k_B T \ln \sum_{j=1}^K \sum_{n=1}^{N_j} \frac{e^{-\beta U_i(x_{jn})}}{\sum_{k=1}^K N_k e^{\hat{A}_k - \beta U_k(x_{jn})}}

**Benefits:**
- Uses all available simulation data
- Provides optimal free energy estimates
- Built-in error analysis

Replica Exchange Thermodynamic Integration
------------------------------------------

**Concept:**

Combine replica exchange with thermodynamic integration for better sampling:

.. math:

   P_{exchange} = \min\left(1, e^{(\beta_i - \beta_j)[\Delta U_j(\lambda_j) - \Delta U_i(\lambda_i)]}\right)

**Applications:**
- Complex conformational changes
- Systems with kinetic barriers
- Improved convergence for difficult transformations

Non-Equilibrium Methods
======================

Jarzynski Equality
-----------------

**Fast Switching:**

For non-equilibrium processes:

.. math:

   \Delta A = -k_B T \ln \langle e^{-\beta W} \rangle

where W is the work performed on the system.

**Crooks Fluctuation Theorem:**

.. math:

   \frac{P_F(W)}{P_R(-W)} = e^{\beta(W - \Delta A)}

**Practical Implementation:**

.. code-block:: python

   def jarzynski_free_energy(work_values):
       """Calculate free energy using Jarzynski equality"""
       beta_work = work_values / (kB * temperature)
       max_work = np.max(beta_work)
       
       # Use numerical tricks to avoid overflow
       exponentials = np.exp(-(beta_work - max_work))
       average = np.mean(exponentials)
       
       return -kB * temperature * (np.log(average) - max_work)

**Challenges:**
- Requires many realizations for convergence
- Sensitive to rare high-work trajectories
- Bias toward low-work events

Error Analysis and Validation
=============================

Statistical Uncertainty
-----------------------

**Block Averaging:**

Divide simulation into blocks and analyze variance:

.. math:

   \sigma^2_{block} = \frac{1}{N_{blocks}-1} \sum_{i=1}^{N_{blocks}} (A_i - \langle A \rangle)^2

**Bootstrap Resampling:**

Generate synthetic datasets by resampling:

.. code-block:: python

   def bootstrap_free_energy_error(data, n_bootstrap=1000):
       errors = []
       for _ in range(n_bootstrap):
           sample = np.random.choice(data, size=len(data), replace=True)
           dG = calculate_free_energy(sample)
           errors.append(dG)
       return np.std(errors)

**Autocorrelation Analysis:**

Account for temporal correlation in data:

.. math::

   \tau_{corr} = 1 + 2\sum_{t=1}^{\infty} C(t)

where C(t) is the normalized autocorrelation function.

Systematic Errors
----------------

**Finite Size Effects:**
- Periodic boundary conditions
- Electrostatic artifacts
- Surface effects

**Sampling Errors:**
- Inadequate phase space exploration
- Metastable state trapping
- Poor overlap between states

**Force Field Errors:**
- Parameter accuracy
- Missing physics (polarization)
- Transferability limitations

Validation Strategies
--------------------

**Experimental Comparison:**
- Direct measurement when available
- Consistent trends across similar systems
- Physical reasonableness of results

**Internal Consistency:**
- Bidirectional calculations
- Different methods for same quantity
- Thermodynamic cycle closure

**Convergence Testing:**
- Simulation length dependence
- Parameter sensitivity analysis
- Multiple independent runs

Best Practices
==============

Protocol Design
---------------

**Free Energy Perturbation:**
- Use ≤ 2 kᵦT perturbations per step
- Include bidirectional calculations
- Monitor overlap between states

**Thermodynamic Integration:**
- Use 11-21 λ points for smooth transformations
- Concentrate points where ∂U/∂λ changes rapidly
- Validate with different λ schedules

**General Guidelines:**
- Always perform convergence analysis
- Use multiple independent runs
- Document all technical details
- Compare with experimental data when available

Common Applications
==================

Drug Design
-----------

**Lead Optimization:**
- Relative binding free energies
- ADMET property prediction
- Selectivity optimization

**Typical Accuracy:**
- 1-2 kcal/mol for relative binding
- 2-3 kcal/mol for absolute binding
- System-dependent performance

**Industrial Implementation:**
- Free energy perturbation (FEP+)
- Thermodynamic integration
- Automated workflow tools

Environmental Chemistry
-----------------------

**Pollutant Fate:**
- Partition coefficients
- Bioaccumulation factors
- Solubility predictions

**Atmospheric Chemistry:**
- Henry's law constants
- Phase partitioning
- Aerosol interactions

Summary
=======

Thermodynamics and free energy calculations provide quantitative connections between molecular simulations and experimental observables:

1. **Fundamental Framework**: Statistical mechanics connects microscopic and macroscopic properties
2. **Multiple Methods**: Different approaches suited for different problems
3. **Rigorous Error Analysis**: Essential for reliable predictions
4. **Experimental Validation**: Always compare with available data
5. **Continuous Development**: Methods continue to improve in accuracy and efficiency

**Current Best Practices:**
- Thermodynamic integration for most applications
- MBAR for optimal data utilization
- Careful validation and error analysis
- Integration with experimental measurements

Free energy calculations have become increasingly reliable and are now routinely used in drug discovery, materials design, and fundamental research. Understanding the theoretical foundations and practical limitations is essential for successful application of these powerful methods.
