==================================
Visualization Tools API Reference
==================================

.. currentmodule:: proteinMD.visualization

The visualization module provides comprehensive tools for creating publication-quality 
molecular visualizations, trajectory animations, and data plots for molecular dynamics simulations.

Overview
========

The visualization module includes:

* **3D Molecular Rendering** - PyMOL and VMD integration with custom renderers
* **Trajectory Visualization** - Animation tools and trajectory players
* **Data Plotting** - Scientific plots for analysis results
* **Interactive Visualization** - Web-based and Jupyter notebook integration

Quick Start
===========

Basic protein visualization::

    from proteinMD.visualization import MolecularViewer
    from proteinMD.structure import Protein
    
    # Load protein structure
    protein = Protein('protein.pdb')
    
    # Create viewer
    viewer = MolecularViewer(backend='pymol')
    
    # Visualize structure
    viewer.show_structure(protein, style='cartoon')
    viewer.color_by_secondary_structure()
    viewer.save_image('protein_structure.png', dpi=300)

3D Molecular Rendering
======================

MolecularViewer
---------------

.. autoclass:: MolecularViewer
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Advanced protein visualization**

.. code-block:: python

    from proteinMD.visualization import MolecularViewer
    from proteinMD.structure import Protein
    from proteinMD.analysis import SecondaryStructure
    
    # Load protein and analyze
    protein = Protein('protein.pdb')
    ss_analyzer = SecondaryStructure()
    ss_data = ss_analyzer.analyze(protein)
    
    # Create high-quality viewer
    viewer = MolecularViewer(
        backend='pymol',
        quality='high',
        ray_tracing=True
    )
    
    # Setup visualization
    viewer.load_structure(protein)
    
    # Custom styling
    viewer.set_style('cartoon', selection='protein')
    viewer.set_style('sticks', selection='ligand')
    viewer.set_style('spheres', selection='water and name O')
    
    # Color by secondary structure
    viewer.color_by_secondary_structure(ss_data)
    
    # Add surface representation
    viewer.show_surface(
        transparency=0.5,
        color_by='electrostatic',
        selection='protein'
    )
    
    # Set camera and lighting
    viewer.set_camera_position([30, 45, 0])
    viewer.set_lighting('dramatic')
    
    # Save high-resolution image
    viewer.save_image(
        'protein_detailed.png',
        width=1920, height=1080,
        dpi=300, antialias=True
    )

PyMOLRenderer
-------------

.. autoclass:: PyMOLRenderer
    :members:
    :undoc-members:
    :show-inheritance:

**Example: PyMOL-specific rendering**

.. code-block:: python

    from proteinMD.visualization import PyMOLRenderer
    
    # Create PyMOL renderer
    pymol = PyMOLRenderer(session_name='md_analysis')
    
    # Load structures
    pymol.load_structure('protein.pdb', object_name='protein')
    pymol.load_trajectory('trajectory.dcd', 'protein')
    
    # Apply visual styles
    pymol.cmd.show('cartoon', 'protein')
    pymol.cmd.color('spectrum', 'protein')
    
    # Create custom selection
    pymol.cmd.select('active_site', 'resi 25+30+45')
    pymol.cmd.show('sticks', 'active_site')
    pymol.cmd.color('red', 'active_site')
    
    # Generate movie
    pymol.create_movie(
        frames=range(1, 101),
        output='protein_movie.mp4',
        fps=30
    )
    
    # Ray trace final frame
    pymol.cmd.ray(1920, 1080)
    pymol.cmd.png('final_frame.png')

VMDRenderer
-----------

.. autoclass:: VMDRenderer
    :members:
    :undoc-members:
    :show-inheritance:

**Example: VMD visualization and scripting**

.. code-block:: python

    from proteinMD.visualization import VMDRenderer
    
    # Create VMD renderer
    vmd = VMDRenderer()
    
    # Load data
    mol_id = vmd.load_structure('protein.pdb')
    vmd.load_trajectory('trajectory.dcd', mol_id)
    
    # Create custom representation
    rep_id = vmd.add_representation(
        mol_id=mol_id,
        style='NewCartoon',
        color='ResType',
        selection='protein'
    )
    
    # Add surface representation
    surface_rep = vmd.add_representation(
        mol_id=mol_id,
        style='QuickSurf',
        color='ColorID 1',
        selection='protein',
        material='Transparent'
    )
    
    # Animate trajectory
    vmd.animate_trajectory(
        start_frame=0,
        end_frame=100,
        step=1,
        output='trajectory_movie.mp4'
    )
    
    # Execute custom TCL script
    vmd.execute_script('''
        display projection orthographic
        axes location off
        color Display Background white
        material change ambient Transparent 0.3
    ''')

NGLViewer
---------

.. autoclass:: NGLViewer
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Interactive web-based visualization**

.. code-block:: python

    from proteinMD.visualization import NGLViewer
    import ipywidgets as widgets
    
    # Create NGL viewer for Jupyter
    ngl = NGLViewer(width=800, height=600)
    
    # Load structure
    ngl.add_structure('protein.pdb')
    
    # Add multiple representations
    ngl.add_representation('cartoon', color='residueindex')
    ngl.add_representation('ball+stick', selection='ligand')
    ngl.add_representation('surface', 
                          selection='protein',
                          opacity=0.3,
                          color='hydrophobicity')
    
    # Create interactive controls
    def update_representation(representation_type):
        ngl.clear_representations()
        ngl.add_representation(representation_type)
    
    rep_dropdown = widgets.Dropdown(
        options=['cartoon', 'ball+stick', 'surface', 'ribbon'],
        value='cartoon',
        description='Style:'
    )
    
    widgets.interact(update_representation, 
                    representation_type=rep_dropdown)
    
    # Display viewer
    display(ngl.widget)

Trajectory Visualization
========================

TrajectoryPlayer
----------------

.. autoclass:: TrajectoryPlayer
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Interactive trajectory playback**

.. code-block:: python

    from proteinMD.visualization import TrajectoryPlayer
    from proteinMD.core import Trajectory
    
    # Load trajectory
    trajectory = Trajectory('trajectory.dcd', 'topology.pdb')
    
    # Create player
    player = TrajectoryPlayer(
        trajectory=trajectory,
        backend='ngl',
        controls=True
    )
    
    # Configure playback
    player.set_frame_rate(30)  # fps
    player.set_loop(True)
    player.set_quality('high')
    
    # Add visualization layers
    player.add_layer('protein', style='cartoon')
    player.add_layer('ligand', style='ball+stick')
    player.add_layer('water', style='line', selection='name O')
    
    # Add measurement overlay
    player.add_measurement(
        'distance',
        atoms=[(100, 'CA'), (200, 'CA')],
        label='Helix distance'
    )
    
    # Start interactive session
    player.play()

AnimationCreator
----------------

.. autoclass:: AnimationCreator
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Custom animation creation**

.. code-block:: python

    from proteinMD.visualization import AnimationCreator
    from proteinMD.analysis import RMSD, RadiusOfGyration
    
    # Create animation with analysis overlay
    animator = AnimationCreator(
        output_format='mp4',
        resolution=(1920, 1080),
        fps=30
    )
    
    # Load trajectory
    animator.load_trajectory('trajectory.dcd', 'topology.pdb')
    
    # Setup main visualization
    animator.add_structure_view(
        style='cartoon',
        color_scheme='bfactor',
        position='left'
    )
    
    # Add analysis plots
    rmsd_analyzer = RMSD()
    rmsd_data = rmsd_analyzer.calculate('trajectory.dcd', 'topology.pdb')
    
    animator.add_plot(
        data=rmsd_data,
        plot_type='line',
        position='top_right',
        title='RMSD vs Time',
        xlabel='Time (ps)',
        ylabel='RMSD (Å)'
    )
    
    # Add progress indicator
    animator.add_progress_indicator(
        rmsd_data['time'],
        style='vertical_line'
    )
    
    # Render animation
    animator.render('protein_dynamics.mp4')

Data Plotting
=============

ScientificPlotter
-----------------

.. autoclass:: ScientificPlotter
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Publication-quality plots**

.. code-block:: python

    from proteinMD.visualization import ScientificPlotter
    from proteinMD.analysis import RMSD, RMSF, EnergyAnalyzer
    import numpy as np
    
    # Create plotter with publication style
    plotter = ScientificPlotter(
        style='publication',
        figure_size=(12, 8),
        dpi=300
    )
    
    # Load analysis data
    rmsd_data = RMSD().calculate('trajectory.dcd', 'topology.pdb')
    rmsf_data = RMSF().calculate('trajectory.dcd', 'topology.pdb')
    energy_data = EnergyAnalyzer().load_energy_file('energy.txt')
    
    # Create multi-panel figure
    fig = plotter.create_figure(rows=2, cols=2)
    
    # Panel 1: RMSD time series
    ax1 = plotter.subplot(0, 0)
    plotter.plot_time_series(
        rmsd_data['time'], rmsd_data['rmsd'],
        xlabel='Time (ns)',
        ylabel='RMSD (Å)',
        title='Backbone RMSD',
        color='blue'
    )
    
    # Panel 2: RMSF per residue
    ax2 = plotter.subplot(0, 1)
    plotter.plot_line(
        rmsf_data['residue'], rmsf_data['rmsf'],
        xlabel='Residue Number',
        ylabel='RMSF (Å)',
        title='Residue Flexibility',
        color='red'
    )
    
    # Panel 3: Energy distribution
    ax3 = plotter.subplot(1, 0)
    plotter.plot_histogram(
        energy_data['potential'],
        bins=50,
        xlabel='Potential Energy (kcal/mol)',
        ylabel='Frequency',
        title='Energy Distribution',
        alpha=0.7
    )
    
    # Panel 4: Energy correlation
    ax4 = plotter.subplot(1, 1)
    plotter.plot_scatter(
        energy_data['kinetic'], energy_data['potential'],
        xlabel='Kinetic Energy (kcal/mol)',
        ylabel='Potential Energy (kcal/mol)',
        title='Energy Correlation',
        alpha=0.5
    )
    
    # Save figure
    plotter.save_figure('analysis_summary.png', bbox_inches='tight')

HeatmapGenerator
----------------

.. autoclass:: HeatmapGenerator
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Contact maps and correlation matrices**

.. code-block:: python

    from proteinMD.visualization import HeatmapGenerator
    from proteinMD.analysis import ContactMap, CrossCorrelation
    
    # Create heatmap generator
    heatmap = HeatmapGenerator(
        colormap='RdYlBu_r',
        figure_size=(10, 8)
    )
    
    # Generate contact map
    contact_analyzer = ContactMap(cutoff=8.0)
    contact_matrix = contact_analyzer.calculate('trajectory.dcd', 'topology.pdb')
    
    # Plot contact frequency map
    heatmap.plot_matrix(
        contact_matrix,
        xlabel='Residue i',
        ylabel='Residue j',
        title='Residue Contact Frequency',
        vmin=0, vmax=1,
        cbar_label='Contact Frequency'
    )
    
    # Add secondary structure annotation
    ss_data = SecondaryStructure().analyze('topology.pdb')
    heatmap.add_structure_annotation(ss_data)
    
    # Save with high quality
    heatmap.save('contact_map.png', dpi=300)
    
    # Cross-correlation analysis
    cc_analyzer = CrossCorrelation()
    cc_matrix = cc_analyzer.calculate('trajectory.dcd', 'topology.pdb')
    
    # Plot correlation heatmap
    heatmap.plot_correlation_matrix(
        cc_matrix,
        title='Residue Cross-Correlation',
        cluster=True,
        dendrogram=True
    )
    
    heatmap.save('correlation_matrix.png')

Interactive Visualization
=========================

JupyterVisualizer
-----------------

.. autoclass:: JupyterVisualizer
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Interactive Jupyter notebook widgets**

.. code-block:: python

    from proteinMD.visualization import JupyterVisualizer
    import ipywidgets as widgets
    from IPython.display import display
    
    # Create interactive visualizer
    viz = JupyterVisualizer()
    
    # Load trajectory
    viz.load_trajectory('trajectory.dcd', 'topology.pdb')
    
    # Create frame slider
    frame_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=viz.n_frames-1,
        step=1,
        description='Frame:'
    )
    
    # Create representation selector
    rep_selector = widgets.Dropdown(
        options=['cartoon', 'ball+stick', 'surface'],
        value='cartoon',
        description='Style:'
    )
    
    # Create color scheme selector
    color_selector = widgets.Dropdown(
        options=['residueindex', 'secondary_structure', 'bfactor'],
        value='residueindex',
        description='Color:'
    )
    
    # Interactive update function
    def update_visualization(frame, style, color):
        viz.show_frame(frame)
        viz.set_representation(style)
        viz.set_color_scheme(color)
        return viz.widget
    
    # Create interactive widget
    interactive_plot = widgets.interact(
        update_visualization,
        frame=frame_slider,
        style=rep_selector,
        color=color_selector
    )

WebViewer
---------

.. autoclass:: WebViewer
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Web-based molecular viewer**

.. code-block:: python

    from proteinMD.visualization import WebViewer
    from flask import Flask
    
    # Create web viewer application
    app = Flask(__name__)
    viewer = WebViewer(app=app)
    
    # Configure viewer
    viewer.set_default_style('cartoon')
    viewer.enable_measurements()
    viewer.enable_trajectory_controls()
    
    # Add protein structure
    viewer.add_structure(
        'protein.pdb',
        name='Protein',
        representations=['cartoon', 'surface']
    )
    
    # Add trajectory
    viewer.add_trajectory(
        'trajectory.dcd',
        topology='protein.pdb',
        name='MD Trajectory'
    )
    
    # Setup analysis panels
    viewer.add_analysis_panel('rmsd')
    viewer.add_analysis_panel('secondary_structure')
    viewer.add_analysis_panel('contact_map')
    
    # Custom route for analysis
    @app.route('/api/calculate_rmsd')
    def calculate_rmsd():
        return viewer.calculate_rmsd_json()
    
    # Run web server
    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5000, debug=True)

Utilities and Styling
=====================

ColorSchemes
------------

.. autoclass:: ColorSchemes
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Custom color schemes**

.. code-block:: python

    from proteinMD.visualization import ColorSchemes
    import matplotlib.pyplot as plt
    
    # Create custom color scheme
    colors = ColorSchemes()
    
    # Define protein secondary structure colors
    ss_colors = colors.create_scheme(
        'secondary_structure',
        {
            'helix': '#FF6B6B',     # Red
            'sheet': '#4ECDC4',     # Teal
            'coil': '#45B7D1',      # Blue
            'turn': '#96CEB4'       # Green
        }
    )
    
    # Define residue type colors
    residue_colors = colors.create_scheme(
        'residue_type',
        {
            'hydrophobic': '#8B4513',
            'polar': '#4169E1',
            'charged_positive': '#FF0000',
            'charged_negative': '#0000FF',
            'special': '#800080'
        }
    )
    
    # Apply to visualization
    viewer = MolecularViewer()
    viewer.load_structure('protein.pdb')
    viewer.apply_color_scheme(ss_colors, attribute='ss')

RenderingEngine
---------------

.. autoclass:: RenderingEngine
    :members:
    :undoc-members:
    :show-inheritance:

**Example: Custom rendering pipeline**

.. code-block:: python

    from proteinMD.visualization import RenderingEngine
    
    # Create custom rendering engine
    engine = RenderingEngine(
        backend='custom',
        quality='ultra',
        multisample=8
    )
    
    # Configure lighting
    engine.add_light(
        type='directional',
        position=[1, 1, 1],
        intensity=0.8,
        color='white'
    )
    
    engine.add_light(
        type='ambient',
        intensity=0.3,
        color='#F0F0F0'
    )
    
    # Setup materials
    engine.create_material(
        'protein_cartoon',
        diffuse=0.8,
        specular=0.2,
        shininess=50
    )
    
    # Render scene
    scene = engine.create_scene()
    scene.add_structure('protein.pdb', material='protein_cartoon')
    
    # Set camera and render
    engine.set_camera(
        position=[50, 50, 50],
        target=[0, 0, 0],
        up=[0, 1, 0]
    )
    
    image = engine.render(width=2048, height=2048)
    engine.save_image(image, 'rendered_protein.png')

Constants and Configuration
===========================

.. autodata:: DEFAULT_STYLES
.. autodata:: COLOR_PALETTES
.. autodata:: RENDERING_BACKENDS

See Also
========

* :doc:`analysis` - Analysis tools that generate visualization data
* :doc:`core` - Core trajectory and structure classes
* :doc:`../user_guide/tutorials` - Visualization tutorials
* :doc:`../advanced/performance` - Performance tips for large visualizations

References
==========

1. Humphrey, W., Dalke, A. & Schulten, K. VMD: Visual molecular dynamics. J. Mol. Graph. 14, 33-38 (1996)
2. Schrödinger, LLC. The PyMOL Molecular Graphics System, Version 2.0
3. Rose, A.S. & Hildebrand, P.W. NGL viewer: web-based molecular graphics. Nucleic Acids Res. 43, W576-W579 (2015)
