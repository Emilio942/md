===============================
Sampling Methods API Reference
===============================

.. currentmodule:: proteinMD.sampling

The sampling module provides various enhanced sampling methods for efficient exploration 
of protein conformational space and calculation of thermodynamic properties.

Overview
========

The sampling module includes:

* **Monte Carlo Methods** - Metropolis Monte Carlo and parallel tempering
* **Enhanced Sampling** - Umbrella sampling, metadynamics, and replica exchange
* **Free Energy Methods** - Thermodynamic integration and free energy perturbation
* **Advanced Techniques** - Accelerated MD, steered MD, and adaptive sampling

Quick Start
===========

Basic umbrella sampling simulation::

    from proteinMD.sampling import UmbrellaSampling
    from proteinMD.core import Simulation
    
    # Setup umbrella sampling
    umbrella = UmbrellaSampling(
        reaction_coordinate='distance',
        atoms=[(1, 10), (20, 30)],
        windows=20,
        force_constant=100.0,
        window_range=(2.0, 8.0)
    )
    
    # Run sampling
    simulation = Simulation('protein.pdb')
    umbrella.run(simulation, steps=100000)
    
    # Analyze results
    pmf = umbrella.calculate_pmf()
    umbrella.plot_pmf('pmf.png')

Monte Carlo Methods
===================

MetropolisMC
------------

.. autoclass:: MetropolisMC
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Basic Monte Carlo sampling**

.. code-block:: python

    from proteinMD.sampling import MetropolisMC
    from proteinMD.structure import Protein
    
    # Setup MC simulation
    protein = Protein('protein.pdb')
    mc = MetropolisMC(
        temperature=300.0,
        max_displacement=0.1,
        move_types=['translation', 'rotation', 'torsion']
    )
    
    # Configure move probabilities
    mc.set_move_probabilities({
        'translation': 0.3,
        'rotation': 0.3,
        'torsion': 0.4
    })
    
    # Run MC simulation
    mc.run(protein, steps=50000)
    
    # Analyze acceptance rates
    print(f"Overall acceptance: {mc.acceptance_rate:.2f}")
    print(f"Move statistics: {mc.move_statistics}")

ParallelTempering
-----------------

.. autoclass:: ParallelTempering
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Replica exchange molecular dynamics**

.. code-block:: python

    from proteinMD.sampling import ParallelTempering
    import numpy as np
    
    # Setup temperature ladder
    temperatures = np.linspace(300, 500, 8)
    
    pt = ParallelTempering(
        temperatures=temperatures,
        exchange_frequency=1000,
        replica_count=len(temperatures)
    )
    
    # Setup replicas
    pt.setup_replicas('protein.pdb', forcefield='amber14sb')
    
    # Run parallel tempering
    pt.run(steps=1000000, output_frequency=1000)
    
    # Analyze exchange statistics
    exchange_matrix = pt.get_exchange_matrix()
    pt.plot_temperature_walk('temp_walk.png')

Enhanced Sampling
=================

UmbrellaSampling
----------------

.. autoclass:: UmbrellaSampling
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Free energy along reaction coordinate**

.. code-block:: python

    from proteinMD.sampling import UmbrellaSampling
    from proteinMD.analysis import ReactionCoordinate
    
    # Define reaction coordinate
    rc = ReactionCoordinate.distance(
        atom1=(0, 'CA'),  # Residue 0, CA atom
        atom2=(10, 'CA')  # Residue 10, CA atom
    )
    
    # Setup umbrella sampling
    umbrella = UmbrellaSampling(
        reaction_coordinate=rc,
        windows=25,
        force_constant=500.0,  # kcal/mol/Å²
        window_range=(3.0, 15.0),
        temperature=300.0
    )
    
    # Generate window centers
    umbrella.generate_windows()
    
    # Run all windows
    for i, window in enumerate(umbrella.windows):
        print(f"Running window {i+1}/{len(umbrella.windows)}")
        window.run(steps=500000)
    
    # Calculate PMF using WHAM
    pmf, error = umbrella.calculate_pmf(method='wham')
    
    # Save results
    umbrella.save_pmf('pmf_data.txt')
    umbrella.plot_pmf('pmf.png', show_error=True)

Metadynamics
------------

.. autoclass:: Metadynamics
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Well-tempered metadynamics**

.. code-block:: python

    from proteinMD.sampling import Metadynamics
    from proteinMD.analysis import CollectiveVariable
    
    # Define collective variables
    cv1 = CollectiveVariable.dihedral(
        atoms=[('PHE', 5, 'N'), ('PHE', 5, 'CA'), 
               ('PHE', 5, 'C'), ('GLY', 6, 'N')],
        name='phi_5'
    )
    
    cv2 = CollectiveVariable.dihedral(
        atoms=[('PHE', 5, 'C'), ('GLY', 6, 'N'), 
               ('GLY', 6, 'CA'), ('GLY', 6, 'C')],
        name='psi_5'
    )
    
    # Setup metadynamics
    metad = Metadynamics(
        collective_variables=[cv1, cv2],
        gaussian_height=1.0,  # kcal/mol
        gaussian_width=[0.2, 0.2],  # radians
        deposition_frequency=500,
        well_tempered=True,
        bias_factor=10.0
    )
    
    # Run simulation
    simulation = Simulation('protein.pdb')
    metad.run(simulation, steps=2000000)
    
    # Reconstruct free energy surface
    fes = metad.calculate_fes(temperature=300.0)
    metad.plot_fes('fes_2d.png', contour_levels=20)

ReplicaExchangeMD
------------------

.. autoclass:: ReplicaExchange
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Hamiltonian replica exchange**

.. code-block:: python

    from proteinMD.sampling import ReplicaExchange
    from proteinMD.forcefield import ForceField
    
    # Create force field variants
    ff_base = ForceField('amber14sb')
    
    # Scale non-bonded interactions
    replicas = []
    for i, scale in enumerate([1.0, 0.9, 0.8, 0.7, 0.6]):
        ff = ff_base.copy()
        ff.scale_nonbonded(scale)
        replicas.append(('replica_%d' % i, ff))
    
    # Setup replica exchange
    rex = ReplicaExchange(
        replicas=replicas,
        exchange_frequency=2000,
        temperature=300.0
    )
    
    # Run simulation
    rex.run('protein.pdb', steps=5000000)
    
    # Analyze mixing
    mixing_stats = rex.analyze_mixing()
    rex.plot_replica_traces('replica_traces.png')

Free Energy Methods
===================

ThermodynamicIntegration
-------------------------

.. autoclass:: ThermodynamicIntegration
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Solvation free energy calculation**

.. code-block:: python

    from proteinMD.sampling import ThermodynamicIntegration
    from proteinMD.environment import WaterBox
    
    # Setup solvation TI
    ti = ThermodynamicIntegration(
        lambda_values=np.linspace(0.0, 1.0, 21),
        perturbation_type='solvation',
        soft_core=True
    )
    
    # Define initial and final states
    ti.set_initial_state('ligand_vacuum.pdb')
    ti.set_final_state('ligand_solvated.pdb')
    
    # Add water box for final state
    water_box = WaterBox(
        box_size=(40, 40, 40),
        model='tip3p',
        ion_concentration=0.15
    )
    
    # Run TI windows
    results = ti.run_all_windows(
        steps_per_window=1000000,
        equilibration_steps=100000
    )
    
    # Calculate free energy
    delta_g, error = ti.integrate()
    print(f"Solvation free energy: {delta_g:.2f} ± {error:.2f} kcal/mol")

FreeEnergyPerturbation
----------------------

.. autoclass:: FreeEnergyPerturbation
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Mutation free energy**

.. code-block:: python

    from proteinMD.sampling import FreeEnergyPerturbation
    from proteinMD.structure import Mutation
    
    # Define mutation
    mutation = Mutation(
        residue_id=25,
        from_residue='ALA',
        to_residue='VAL'
    )
    
    # Setup FEP calculation
    fep = FreeEnergyPerturbation(
        mutation=mutation,
        lambda_schedule=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
        soft_core_parameters={'alpha': 0.5, 'sigma': 1.0}
    )
    
    # Run forward and backward transformations
    forward_results = fep.run_forward(steps=2000000)
    backward_results = fep.run_backward(steps=2000000)
    
    # Calculate free energy with BAR
    delta_g_bar = fep.calculate_bar()
    delta_g_ti = fep.calculate_ti()
    
    print(f"ΔG (BAR): {delta_g_bar:.2f} kcal/mol")
    print(f"ΔG (TI):  {delta_g_ti:.2f} kcal/mol")

Advanced Sampling Techniques
============================

AcceleratedMD
-------------

.. autoclass:: AcceleratedMD
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Dual-boost accelerated MD**

.. code-block:: python

    from proteinMD.sampling import AcceleratedMD
    
    # Setup dual-boost aMD
    amd = AcceleratedMD(
        boost_type='dual',
        dihedral_boost={
            'threshold': 'auto',  # Automatically determine
            'alpha': 0.2
        },
        total_boost={
            'threshold': 'auto',
            'alpha': 0.2
        }
    )
    
    # Run accelerated simulation
    simulation = Simulation('protein.pdb')
    amd.setup(simulation)
    
    # Production run
    amd.run(steps=10000000, output_frequency=1000)
    
    # Reweight trajectories
    unbiased_weights = amd.calculate_reweighting()
    
    # Analyze enhanced sampling
    acceleration_factor = amd.get_acceleration_factor()
    print(f"Average acceleration: {acceleration_factor:.1f}x")

SteeredMD
---------

.. autoclass:: SteeredMD
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Protein unfolding simulation**

.. code-block:: python

    from proteinMD.sampling import SteeredMD
    from proteinMD.analysis import ReactionCoordinate
    
    # Define pulling coordinate
    pulling_coord = ReactionCoordinate.distance(
        atom1=(0, 'CA'),    # N-terminus
        atom2=(-1, 'CA')    # C-terminus
    )
    
    # Setup steered MD
    smd = SteeredMD(
        coordinate=pulling_coord,
        force_constant=10.0,  # kcal/mol/Å²
        pulling_rate=0.01,    # Å/ps
        direction='increase'
    )
    
    # Run pulling simulation
    simulation = Simulation('folded_protein.pdb')
    smd.run(simulation, steps=5000000)
    
    # Analyze work and force
    work_profile = smd.get_work_profile()
    force_profile = smd.get_force_profile()
    
    # Plot results
    smd.plot_pulling_profile('pulling_profile.png')
    
    # Estimate unfolding force
    max_force = smd.get_maximum_force()
    print(f"Maximum unfolding force: {max_force:.1f} pN")

AdaptiveSampling
----------------

.. autoclass:: AdaptiveSampling
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Adaptive sampling workflow**

.. code-block:: python

    from proteinMD.sampling import AdaptiveSampling
    from proteinMD.analysis import ConformationalClustering
    
    # Setup adaptive sampling
    adaptive = AdaptiveSampling(
        initial_structure='protein.pdb',
        simulation_length=1000000,  # steps per round
        n_rounds=10,
        selection_method='least_sampled'
    )
    
    # Define conformational space clustering
    clustering = ConformationalClustering(
        method='kmeans',
        n_clusters=50,
        features=['backbone_dihedrals', 'distances']
    )
    
    adaptive.set_clustering(clustering)
    
    # Run adaptive sampling rounds
    for round_num in range(adaptive.n_rounds):
        print(f"Round {round_num + 1}")
        
        # Run simulations from selected conformations
        adaptive.run_round()
        
        # Update clustering and selection
        adaptive.update_sampling_map()
        
        # Select new starting conformations
        selected_frames = adaptive.select_next_round()
        
        print(f"Selected {len(selected_frames)} new starting points")
    
    # Analyze final conformational landscape
    final_clustering = adaptive.get_final_clustering()
    adaptive.plot_sampling_map('sampling_map.png')

Utilities and Analysis
======================

SamplingAnalyzer
----------------

.. autoclass:: SamplingAnalyzer
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Comprehensive sampling analysis**

.. code-block:: python

    from proteinMD.sampling import SamplingAnalyzer
    
    # Load trajectory from sampling simulation
    analyzer = SamplingAnalyzer('trajectory.dcd', 'topology.pdb')
    
    # Analyze convergence
    convergence = analyzer.analyze_convergence(
        properties=['rmsd', 'radius_of_gyration', 'energy'],
        block_size=10000
    )
    
    # Calculate sampling efficiency
    efficiency = analyzer.calculate_efficiency(
        reference_trajectory='reference.dcd'
    )
    
    # Assess conformational coverage
    coverage = analyzer.assess_coverage(
        reference_ensemble='reference_ensemble.pdb'
    )
    
    # Generate comprehensive report
    analyzer.generate_report('sampling_report.html')

Constants and Configuration
===========================

.. autodata:: DEFAULT_MC_PARAMETERS
.. autodata:: ENHANCED_SAMPLING_METHODS
.. autodata:: FREE_ENERGY_METHODS

See Also
========

* :doc:`core` - Core simulation engine
* :doc:`analysis` - Analysis tools for sampling results
* :doc:`../user_guide/tutorials` - Step-by-step sampling tutorials
* :doc:`../advanced/performance` - Performance optimization for sampling

References
==========

1. Frenkel, D. & Smit, B. Understanding Molecular Simulation (Academic Press, 2001)
2. Chipot, C. & Pohorille, A. Free Energy Calculations (Springer, 2007)
3. Laio, A. & Parrinello, M. Escaping free-energy minima. PNAS 99, 12562-12566 (2002)
4. Sugita, Y. & Okamoto, Y. Replica-exchange molecular dynamics method. Chem. Phys. Lett. 314, 141-151 (1999)
