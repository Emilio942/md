================================
Statistical Mechanics Foundation
================================

Introduction
============

Statistical mechanics provides the theoretical framework connecting the microscopic dynamics observed in MD simulations to macroscopic thermodynamic properties. This connection is essential for interpreting simulation results and relating them to experimental measurements.

Classical Statistical Mechanics
===============================

Phase Space and Ensembles
--------------------------

**Phase Space**

A system of N particles is described by a 6N-dimensional phase space, where each particle contributes 3 position and 3 momentum coordinates:

.. math::

   \Gamma = (r_1, r_2, ..., r_N, p_1, p_2, ..., p_N)

**Liouville's Theorem**

The phase space density :math:`\rho(\Gamma, t)` evolves according to Liouville's equation:

.. math::

   \frac{\partial \rho}{\partial t} + \{\rho, H\} = 0

where :math:`\{,\}` is the Poisson bracket and H is the Hamiltonian.

**Conservation of Phase Space Volume**

.. math::

   \frac{d\rho}{dt} = 0

This fundamental result ensures that MD trajectories preserve the phase space density.

Statistical Ensembles
======================

Different experimental conditions correspond to different statistical ensembles, each characterized by conserved quantities.

Microcanonical Ensemble (NVE)
------------------------------

**Conserved Quantities:** Number of particles (N), Volume (V), Energy (E)

**Probability Distribution:**

.. math::

   P(\Gamma) = \frac{1}{\Omega(N,V,E)} \delta(H(\Gamma) - E)

where :math:`\Omega(N,V,E)` is the density of states.

**Entropy:**

.. math::

   S = k_B \ln \Omega(N,V,E)

**Applications:**
- Isolated systems
- Validation of integrators
- Energy conservation checks

Canonical Ensemble (NVT)
-------------------------

**Conserved Quantities:** Number of particles (N), Volume (V), Temperature (T)

**Probability Distribution:**

.. math::

   P(\Gamma) = \frac{1}{Z(N,V,T)} \exp(-\beta H(\Gamma))

where :math:`\beta = 1/(k_B T)` and Z is the partition function.

**Partition Function:**

.. math::

   Z(N,V,T) = \frac{1}{N! h^{3N}} \int \exp(-\beta H(\Gamma)) d\Gamma

**Free Energy:**

.. math::

   F = -k_B T \ln Z

**Implementation in MD:**
- Thermostat maintains constant temperature
- Most common ensemble for biological simulations
- Corresponds to experimental conditions with heat bath

Isothermal-Isobaric Ensemble (NPT)
-----------------------------------

**Conserved Quantities:** Number of particles (N), Pressure (P), Temperature (T)

**Probability Distribution:**

.. math::

   P(\Gamma, V) = \frac{1}{Z(N,P,T)} \exp(-\beta[H(\Gamma) + PV])

**Partition Function:**

.. math::

   Z(N,P,T) = \frac{1}{N! h^{3N}} \int_0^{\infty} \int \exp(-\beta[H(\Gamma) + PV]) d\Gamma dV

**Gibbs Free Energy:**

.. math::

   G = -k_B T \ln Z

**Implementation in MD:**
- Barostat maintains constant pressure
- Most relevant for condensed phase systems
- Standard conditions for most experiments

Grand Canonical Ensemble (μVT)
-------------------------------

**Conserved Quantities:** Chemical potential (μ), Volume (V), Temperature (T)

**Applications:**
- Open systems with particle exchange
- Adsorption studies
- Rarely used in protein simulations

Thermodynamic Relations
=======================

Statistical mechanics provides exact relationships between microscopic quantities and thermodynamic properties.

Average Values
--------------

For any observable A, the ensemble average is:

.. math::

   \langle A \rangle = \frac{\int A(\Gamma) P(\Gamma) d\Gamma}{\int P(\Gamma) d\Gamma}

**Energy:**

.. math::

   \langle E \rangle = -\frac{\partial \ln Z}{\partial \beta}

**Heat Capacity:**

.. math::

   C_V = \frac{\partial \langle E \rangle}{\partial T} = k_B \beta^2 \langle (\Delta E)^2 \rangle

where :math:`\langle (\Delta E)^2 \rangle` is the energy fluctuation.

**Pressure:**

.. math::

   \langle P \rangle = -\frac{\partial F}{\partial V} = \frac{k_B T}{V} + \frac{1}{3V} \langle \sum_i \vec{r_i} \cdot \vec{F_i} \rangle

Fluctuations and Response Functions
===================================

Statistical mechanics relates fluctuations to experimentally measurable response functions.

**General Fluctuation-Dissipation Relation:**

.. math::

   \langle (\Delta A)^2 \rangle = k_B T^2 \frac{\partial \langle A \rangle}{\partial T}

**Specific Examples:**

Energy fluctuations → Heat capacity:

.. math::

   C_V = \frac{\langle (\Delta E)^2 \rangle}{k_B T^2}

Volume fluctuations → Isothermal compressibility:

.. math::

   \kappa_T = \frac{\langle (\Delta V)^2 \rangle}{k_B T \langle V \rangle}

Pressure fluctuations → Bulk modulus:

.. math::

   K = \frac{1}{\kappa_T} = \frac{k_B T \langle V \rangle}{\langle (\Delta V)^2 \rangle}

Time Correlation Functions
==========================

Dynamical properties are characterized by time correlation functions, which connect equilibrium fluctuations to transport properties.

**Autocorrelation Function:**

.. math::

   C_{AA}(t) = \langle A(0) A(t) \rangle

**Cross-correlation Function:**

.. math::

   C_{AB}(t) = \langle A(0) B(t) \rangle

**Properties:**
- :math:`C_{AA}(0) = \langle A^2 \rangle` (maximum value)
- :math:`C_{AA}(\infty) = \langle A \rangle^2` (for equilibrium systems)
- Decay time reflects characteristic relaxation processes

Linear Response Theory
======================

Linear response theory relates equilibrium fluctuations to the system's response to small perturbations.

**General Linear Response:**

.. math::

   \langle B(t) \rangle = \int_0^t \chi_{BA}(t-t') h_A(t') dt'

where :math:`h_A(t')` is a small external field and :math:`\chi_{BA}(t)` is the response function.

**Fluctuation-Dissipation Theorem:**

.. math::

   \chi_{BA}(t) = \frac{1}{k_B T} \frac{d}{dt} C_{BA}(t)

**Transport Coefficients:**

This framework allows calculation of transport properties from equilibrium MD simulations:

Diffusion coefficient:

.. math::

   D = \frac{1}{6} \int_0^{\infty} \langle \vec{v}(0) \cdot \vec{v}(t) \rangle dt

Viscosity:

.. math::

   \eta = \frac{V}{k_B T} \int_0^{\infty} \langle \sigma_{xy}(0) \sigma_{xy}(t) \rangle dt

Free Energy Calculations
=========================

Free energy differences are central to understanding molecular processes but cannot be calculated directly from MD.

**Free Energy Perturbation (FEP):**

.. math::

   \Delta F = F_1 - F_0 = -k_B T \ln \langle \exp(-\beta \Delta U) \rangle_0

where :math:`\Delta U = U_1 - U_0` is the potential energy difference.

**Thermodynamic Integration (TI):**

.. math::

   \Delta F = \int_0^1 \left\langle \frac{\partial U(\lambda)}{\partial \lambda} \right\rangle_\lambda d\lambda

**Umbrella Sampling:**

For processes with high energy barriers, biasing potentials are used:

.. math::

   w_i(\xi) = -k_B T \ln P_i(\xi) + W_i(\xi) + C_i

where :math:`W_i(\xi)` is the biasing potential and the unbiased distribution is recovered by WHAM.

Ergodicity and Sampling
=======================

Proper sampling is crucial for obtaining reliable statistical averages from MD simulations.

**Ergodic Hypothesis:**

.. math::

   \langle A \rangle_{ensemble} = \lim_{T \to \infty} \frac{1}{T} \int_0^T A(t) dt

**Requirements for Ergodicity:**
1. System must access all relevant phase space
2. No broken ergodicity (multiple basins)
3. Simulation time >> correlation times

**Sampling Problems:**
- Energy barriers between conformations
- Slow relaxation processes
- Metastable states
- Rare events

**Enhanced Sampling Methods:**
- Replica exchange MD
- Metadynamics
- Accelerated MD
- Steered MD

Temperature Coupling
====================

Thermostats modify the equations of motion to maintain constant temperature while preserving the canonical distribution.

**Velocity Rescaling (Berendsen):**

.. math::

   v_i^{new} = v_i \sqrt{1 + \frac{\Delta t}{\tau_T} \left(\frac{T_0}{T} - 1\right)}

Pros: Simple, stable
Cons: Does not generate canonical ensemble

**Nosé-Hoover Thermostat:**

Extended Lagrangian with additional degree of freedom:

.. math::

   \dot{v_i} = \frac{F_i}{m_i} - \zeta v_i

.. math::

   \dot{\zeta} = \frac{1}{Q} \left(\sum_i m_i v_i^2 - N_f k_B T \right)

Pros: Generates correct canonical ensemble
Cons: More complex, can show oscillations

**Langevin Dynamics:**

.. math::

   m_i \ddot{r_i} = F_i - \gamma m_i \dot{r_i} + \sqrt{2\gamma m_i k_B T} R_i(t)

where :math:`R_i(t)` is white noise with :math:`\langle R_i(t) R_j(t') \rangle = \delta_{ij} \delta(t-t')`.

Pros: Natural coupling to environment
Cons: Modified dynamics, affects diffusion

Pressure Coupling
=================

Barostats control pressure by allowing volume fluctuations while maintaining the NPT ensemble.

**Berendsen Barostat:**

.. math::

   \frac{dV}{dt} = \frac{V}{\tau_P} \kappa_T (P_0 - P)

Simple but does not generate correct NPT ensemble.

**Parrinello-Rahman Barostat:**

Allows both volume and shape changes:

.. math::

   \ddot{h} = V W^{-1} [P - P_0]

where h is the box matrix and W is the barostat mass.

**Considerations:**
- Coupling time must be much larger than vibrational periods
- Protein simulations often use semi-isotropic coupling
- Membrane simulations require anisotropic pressure coupling

Error Analysis
==============

Statistical errors in MD simulations arise from finite sampling.

**Standard Error:**

For uncorrelated samples:

.. math::

   \sigma_{\langle A \rangle} = \frac{\sigma_A}{\sqrt{N}}

**Correlation Effects:**

For correlated data with correlation time :math:`\tau_c`:

.. math::

   \sigma_{\langle A \rangle} = \frac{\sigma_A}{\sqrt{N_{eff}}} = \frac{\sigma_A}{\sqrt{N/(2\tau_c + 1)}}

**Block Averaging:**

Divide trajectory into blocks and analyze block averages to estimate correlation time and statistical error.

**Bootstrap Methods:**

Resample trajectory frames to estimate confidence intervals for complex observables.

Practical Guidelines
====================

**Simulation Length:**
- Equilibration: 5-10 correlation times
- Production: 50-100 correlation times for good statistics
- Monitor convergence of properties of interest

**System Size:**
- Large enough to avoid finite size effects
- Rule of thumb: protein should not interact with its periodic image
- Minimum 8-10 Å buffer for solvated systems

**Time Step:**
- 1-2 fs for systems with hydrogen atoms
- 2-4 fs with SHAKE/RATTLE constraints
- Monitor energy conservation and temperature

**Temperature and Pressure:**
- Use weak coupling (large τ values) to avoid artifacts
- Berendsen: τ = 0.1-1.0 ps
- Nosé-Hoover: τ = 0.5-2.0 ps

Summary
=======

Statistical mechanics provides the theoretical foundation for:

1. **Ensemble Theory**: Connecting microscopic dynamics to thermodynamic quantities
2. **Fluctuation-Dissipation Relations**: Relating equilibrium fluctuations to response functions
3. **Free Energy Methods**: Calculating thermodynamic driving forces
4. **Error Analysis**: Quantifying statistical uncertainties
5. **Enhanced Sampling**: Overcoming sampling limitations

Understanding these principles is essential for:
- Choosing appropriate simulation conditions
- Interpreting results correctly
- Estimating statistical uncertainties
- Designing enhanced sampling strategies

The next sections will apply these concepts to specific aspects of MD simulation, including force field theory and integration algorithms.
