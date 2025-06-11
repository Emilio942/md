===========
Quick Start
===========

Get up and running with ProteinMD in just a few minutes. This guide will walk you through your first molecular dynamics simulation.

Your First Simulation in 5 Minutes
===================================

1. **Install ProteinMD** (if not already done)::

    pip install proteinMD

2. **Download a sample protein**::

    # Download ubiquitin (PDB: 1UBQ) as an example
    wget https://files.rcsb.org/download/1UBQ.pdb

3. **Run a basic simulation**::

    from proteinMD import Simulation
    
    # Create and run simulation
    sim = Simulation.from_pdb('1UBQ.pdb')
    sim.run(steps=10000)  # 20 ps simulation
    
    # Analyze results
    sim.plot_energy('energy.png')
    sim.save_trajectory('trajectory.dcd')

That's it! You've just run your first molecular dynamics simulation.

Basic Example - Step by Step
=============================

Let's break down a complete simulation workflow:

Loading a Protein Structure
---------------------------

.. code-block:: python

    from proteinMD.structure import PDBParser, Protein
    
    # Parse PDB file
    parser = PDBParser()
    protein = parser.parse('protein.pdb')
    
    # Basic structure information
    print(f"Protein has {protein.n_atoms} atoms")
    print(f"Protein has {protein.n_residues} residues")
    print(f"Chains: {protein.chain_ids}")

Setting Up the Force Field
---------------------------

.. code-block:: python

    from proteinMD.forcefield import AmberFF14SB
    
    # Load AMBER ff14SB force field
    forcefield = AmberFF14SB()
    
    # Apply to protein
    forcefield.apply(protein)
    
    # Verify parameters are assigned
    print(f"Force field applied: {forcefield.is_complete}")

Adding Solvent Environment
--------------------------

.. code-block:: python

    from proteinMD.environment import WaterBox, TIP3P
    
    # Create water box
    water_model = TIP3P()
    water_box = WaterBox(
        protein=protein,
        model=water_model,
        padding=1.0,  # 10 Å padding
        ion_concentration=0.15  # 150 mM NaCl
    )
    
    # Solvate the protein
    solvated_system = water_box.solvate()
    print(f"Solvated system has {solvated_system.n_atoms} atoms")

Running the Simulation
----------------------

.. code-block:: python

    from proteinMD.core import Simulation
    
    # Create simulation
    simulation = Simulation(
        system=solvated_system,
        forcefield=forcefield,
        temperature=300.0,  # 300 K
        pressure=1.0,       # 1 atm
        timestep=0.002      # 2 fs
    )
    
    # Energy minimization
    simulation.minimize(steps=1000)
    
    # Equilibration (NVT)
    simulation.equilibrate_nvt(
        steps=50000,    # 100 ps
        temperature=300.0
    )
    
    # Equilibration (NPT)
    simulation.equilibrate_npt(
        steps=50000,    # 100 ps
        temperature=300.0,
        pressure=1.0
    )
    
    # Production run
    simulation.run(
        steps=500000,   # 1 ns
        output_frequency=1000,
        save_trajectory=True
    )

Analyzing Results
-----------------

.. code-block:: python

    from proteinMD.analysis import RMSD, RMSF, SecondaryStructure
    
    # Load trajectory
    trajectory = simulation.get_trajectory()
    
    # Calculate RMSD
    rmsd_analyzer = RMSD(reference_frame=0)
    rmsd_data = rmsd_analyzer.calculate(trajectory)
    
    # Calculate RMSF
    rmsf_analyzer = RMSF()
    rmsf_data = rmsf_analyzer.calculate(trajectory)
    
    # Secondary structure analysis
    ss_analyzer = SecondaryStructure()
    ss_data = ss_analyzer.analyze(trajectory)
    
    # Plot results
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # RMSD plot
    axes[0,0].plot(rmsd_data['time'], rmsd_data['rmsd'])
    axes[0,0].set_xlabel('Time (ps)')
    axes[0,0].set_ylabel('RMSD (Å)')
    axes[0,0].set_title('Backbone RMSD')
    
    # RMSF plot
    axes[0,1].plot(rmsf_data['residue'], rmsf_data['rmsf'])
    axes[0,1].set_xlabel('Residue')
    axes[0,1].set_ylabel('RMSF (Å)')
    axes[0,1].set_title('Residue Flexibility')
    
    # Energy plot
    energy_data = simulation.get_energy_data()
    axes[1,0].plot(energy_data['time'], energy_data['potential'])
    axes[1,0].set_xlabel('Time (ps)')
    axes[1,0].set_ylabel('Energy (kcal/mol)')
    axes[1,0].set_title('Potential Energy')
    
    # Secondary structure plot
    ss_analyzer.plot_timeline(ss_data, ax=axes[1,1])
    axes[1,1].set_title('Secondary Structure')
    
    plt.tight_layout()
    plt.savefig('analysis_summary.png', dpi=300)

Command Line Interface
======================

ProteinMD also provides a powerful command-line interface:

Basic CLI Usage
---------------

Run a simulation from the command line::

    # Basic simulation
    proteinmd simulate protein.pdb --steps 100000 --output results/
    
    # With custom parameters
    proteinmd simulate protein.pdb \
        --forcefield amber14sb \
        --water tip3p \
        --temperature 310 \
        --steps 1000000 \
        --output high_temp_sim/

Analysis from CLI
-----------------

Analyze existing trajectories::

    # Calculate RMSD
    proteinmd analyze rmsd trajectory.dcd topology.pdb
    
    # Multiple analyses
    proteinmd analyze trajectory.dcd topology.pdb \
        --rmsd --rmsf --secondary-structure \
        --output analysis_results/

Batch Processing
----------------

Process multiple structures::

    # Simulate all PDB files in directory
    proteinmd batch simulate input_pdbs/ --output batch_results/
    
    # Custom workflow
    proteinmd workflow protein_folding.yaml --input structures/

Interactive Tutorials
====================

Jupyter Notebook Examples
--------------------------

Start with interactive examples::

    # Download tutorial notebooks
    proteinmd examples --download
    
    # Start Jupyter
    jupyter notebook proteinmd_tutorials/

**Tutorial 1: Basic MD Simulation**
  - Load protein structure
  - Setup force field and solvent
  - Run MD simulation
  - Basic analysis

**Tutorial 2: Enhanced Sampling**
  - Umbrella sampling
  - Replica exchange
  - Free energy calculations

**Tutorial 3: Protein Folding Study**
  - Temperature replica exchange
  - Folding pathway analysis
  - Free energy landscapes

Common Workflows
================

Protein Stability Study
-----------------------

.. code-block:: python

    from proteinMD import *
    
    # Load wild-type and mutant proteins
    wt_protein = PDBParser().parse('wildtype.pdb')
    mut_protein = PDBParser().parse('mutant.pdb')
    
    # Setup simulations
    wt_sim = Simulation.from_structure(wt_protein)
    mut_sim = Simulation.from_structure(mut_protein)
    
    # Run parallel simulations
    wt_sim.run(steps=1000000)  # 2 ns
    mut_sim.run(steps=1000000)
    
    # Compare stability
    wt_rmsd = RMSD().calculate(wt_sim.trajectory)
    mut_rmsd = RMSD().calculate(mut_sim.trajectory)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(wt_rmsd['time'], wt_rmsd['rmsd'], label='Wild-type')
    plt.plot(mut_rmsd['time'], mut_rmsd['rmsd'], label='Mutant')
    plt.xlabel('Time (ps)')
    plt.ylabel('RMSD (Å)')
    plt.legend()
    plt.title('Protein Stability Comparison')
    plt.savefig('stability_comparison.png')

Ligand Binding Study
--------------------

.. code-block:: python

    from proteinMD.structure import Complex
    from proteinMD.analysis import BindingAnalysis
    
    # Load protein-ligand complex
    complex_structure = Complex.from_pdb('protein_ligand.pdb')
    
    # Setup simulation with flexible ligand
    simulation = Simulation(complex_structure)
    simulation.set_flexible_residues(['ligand'])
    
    # Run simulation
    simulation.run(steps=2000000)  # 4 ns
    
    # Analyze binding
    binding_analyzer = BindingAnalysis(
        protein_selection='protein',
        ligand_selection='resname LIG'
    )
    
    binding_data = binding_analyzer.analyze(simulation.trajectory)
    
    # Plot binding pose stability
    binding_analyzer.plot_rmsd('ligand_rmsd.png')
    binding_analyzer.plot_contacts('protein_ligand_contacts.png')

Membrane Protein Simulation
---------------------------

.. code-block:: python

    from proteinMD.environment import LipidBilayer, POPC
    
    # Load membrane protein
    membrane_protein = PDBParser().parse('membrane_protein.pdb')
    
    # Setup lipid bilayer
    lipid_model = POPC()
    bilayer = LipidBilayer(
        protein=membrane_protein,
        lipid_type=lipid_model,
        lipid_ratio={'POPC': 0.7, 'POPE': 0.3},
        water_thickness=2.0  # 20 Å water layer
    )
    
    # Insert protein into membrane
    membrane_system = bilayer.insert_protein()
    
    # Run membrane simulation
    simulation = Simulation(membrane_system)
    simulation.set_membrane_constraints()
    simulation.run(steps=5000000)  # 10 ns

Performance Tips
================

GPU Acceleration
----------------

Enable GPU for faster simulations::

    # Check GPU availability
    from proteinMD.core import check_gpu_support
    print(f"GPU available: {check_gpu_support()}")
    
    # Use GPU platform
    simulation = Simulation(system, platform='CUDA')

Parallel Simulations
--------------------

Run multiple simulations in parallel::

    from proteinMD.utils import ParallelRunner
    
    # Setup multiple simulations
    simulations = [
        Simulation.from_pdb(f'replica_{i}.pdb') 
        for i in range(8)
    ]
    
    # Run in parallel
    runner = ParallelRunner(n_processes=8)
    results = runner.run_simulations(simulations)

Memory Optimization
-------------------

For large systems::

    # Use memory-efficient options
    simulation = Simulation(
        system=large_system,
        trajectory_format='xtc',  # Compressed format
        save_frequency=10000,     # Save less frequently
        precision='single'        # Use single precision
    )

Next Steps
==========

Now that you've run your first simulation, explore more advanced features:

**Learning Path:**

1. **Tutorials** (:doc:`tutorials`) - Detailed step-by-step guides
2. **CLI Reference** (:doc:`cli_reference`) - Master the command-line tools
3. **API Documentation** (:doc:`../api/core`) - Explore all functionality
4. **Examples** (:doc:`examples`) - Real-world simulation examples

**Advanced Topics:**

- :doc:`../advanced/performance` - Optimization for large systems
- :doc:`../advanced/extending` - Add custom force fields
- :doc:`../api/sampling` - Enhanced sampling methods
- :doc:`../api/visualization` - Advanced visualization

**Community:**

- Join the discussion forum for questions
- Follow tutorials on YouTube
- Check out example simulations on GitHub

Common Issues
=============

**Simulation Crashes**
  - Check force field assignment is complete
  - Verify structure has no clashes
  - Use energy minimization before MD

**Poor Performance**
  - Enable GPU acceleration if available
  - Reduce output frequency for large systems
  - Use appropriate precision settings

**Analysis Errors**  
  - Ensure trajectory and topology match
  - Check frame numbers are valid
  - Verify selection strings are correct

Need help? See :doc:`../advanced/troubleshooting` for detailed solutions.
