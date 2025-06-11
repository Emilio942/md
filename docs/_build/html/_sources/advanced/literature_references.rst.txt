Literature References for Molecular Dynamics
==========================================

This section provides a comprehensive collection of literature references organized by topic to support further study in molecular dynamics simulation theory and practice.

.. contents:: Table of Contents
   :local:
   :depth: 3

Foundational Texts and Reviews
-------------------------------

Comprehensive Textbooks
~~~~~~~~~~~~~~~~~~~~~~~~

**Classical Molecular Dynamics**

1. **Allen, M. P., & Tildesley, D. J.** (2017). *Computer Simulation of Liquids* (2nd ed.). Oxford University Press.
   
   - The definitive textbook for molecular simulation methods
   - Comprehensive coverage of algorithms and theory
   - Essential reading for understanding MD fundamentals

2. **Frenkel, D., & Smit, B.** (2001). *Understanding Molecular Simulation: From Algorithms to Applications* (2nd ed.). Academic Press.
   
   - Excellent balance of theory and practical applications
   - Strong coverage of Monte Carlo and molecular dynamics
   - Good introduction to enhanced sampling methods

3. **Rapaport, D. C.** (2004). *The Art of Molecular Dynamics Simulation* (2nd ed.). Cambridge University Press.
   
   - Practical focus with extensive code examples
   - Good for implementation details
   - Covers both methodology and analysis techniques

**Statistical Mechanics Foundations**

4. **McQuarrie, D. A.** (2000). *Statistical Mechanics*. University Science Books.
   
   - Rigorous treatment of statistical mechanical principles
   - Essential for understanding ensemble theory
   - Strong mathematical foundations

5. **Chandler, D.** (1987). *Introduction to Modern Statistical Mechanics*. Oxford University Press.
   
   - Clear exposition of statistical mechanical concepts
   - Good coverage of fluctuation-dissipation theory
   - Emphasis on understanding physical principles

Major Review Articles
~~~~~~~~~~~~~~~~~~~~~

**Methodological Reviews**

6. **Karplus, M., & McCammon, J. A.** (2002). Molecular dynamics simulations of biomolecules. *Nature Structural Biology*, 9(9), 646-652.

   - Historical perspective on biomolecular MD
   - Overview of major achievements and challenges
   - Influential review by pioneers in the field

7. **Dror, R. O., Dirks, R. M., Grossman, J. P., Xu, H., & Shaw, D. E.** (2012). Biomolecular simulation: a computational microscope for molecular biology. *Annual Review of Biophysics*, 41, 429-452.

   - Modern perspective on long-timescale simulations
   - Discussion of specialized hardware (Anton)
   - Focus on biological applications

8. **Hollingsworth, S. A., & Dror, R. O.** (2018). Molecular dynamics simulation for all. *Neuron*, 99(6), 1129-1143.

   - Accessible introduction to MD for biologists
   - Practical guidance for experimental researchers
   - Overview of current capabilities and limitations

Force Fields and Parameterization
----------------------------------

Force Field Development
~~~~~~~~~~~~~~~~~~~~~~~

**General Force Field Theory**

9. **Jorgensen, W. L., Maxwell, D. S., & Tirado-Rives, J.** (1996). Development and testing of the OPLS all-atom force field on conformational energetics and properties of organic liquids. *Journal of the American Chemical Society*, 118(45), 11225-11236.

   - Seminal paper on OPLS force field development
   - Systematic approach to parameter optimization
   - Validation against experimental data

10. **Cornell, W. D., Cieplak, P., Bayly, C. I., Gould, I. R., Merz, K. M., Ferguson, D. M., ... & Kollman, P. A.** (1995). A second generation force field for the simulation of proteins, nucleic acids, and organic molecules. *Journal of the American Chemical Society*, 117(19), 5179-5197.

    - Development of AMBER ff94 force field
    - Influential approach to biomolecular parameterization
    - Widely cited and used methodology

**Protein Force Fields**

11. **Maier, J. A., Martinez, C., Kasavajhala, K., Wickstrom, L., Hauser, K. E., & Simmerling, C.** (2015). ff14SB: improving the accuracy of protein side chain and backbone parameters from ff99SB. *Journal of Chemical Theory and Computation*, 11(8), 3696-3713.

    - Modern AMBER protein force field
    - Systematic improvement of backbone parameters
    - Extensive validation studies

12. **Best, R. B., Zhu, X., Shim, J., Lopes, P. E., Mittal, J., Feig, M., & MacKerell Jr, A. D.** (2012). Optimization of the additive CHARMM all-atom protein force field targeting improved sampling of the backbone φ, ψ and side-chain χ1 and χ2 dihedral angles. *Journal of Chemical Theory and Computation*, 8(9), 3257-3273.

    - CHARMM22* force field development
    - Focus on conformational sampling improvements
    - Validation against NMR data

**Water Models**

13. **Jorgensen, W. L., Chandrasekhar, J., Madura, J. D., Impey, R. W., & Klein, M. L.** (1983). Comparison of simple potential functions for simulating liquid water. *Journal of Chemical Physics*, 79(2), 926-935.

    - Classic paper introducing TIP3P and TIP4P models
    - Systematic comparison of water models
    - Foundation for modern water model development

14. **Abascal, J. L., & Vega, C.** (2005). A general purpose model for the condensed phases of water: TIP4P/2005. *Journal of Chemical Physics*, 123(23), 234505.

    - Improved TIP4P water model
    - Better reproduction of experimental properties
    - Widely used in modern simulations

Integration Algorithms
----------------------

Fundamental Algorithms
~~~~~~~~~~~~~~~~~~~~~~

**Verlet Integration Family**

15. **Verlet, L.** (1967). Computer "experiments" on classical fluids. I. Thermodynamical properties of Lennard-Jones molecules. *Physical Review*, 159(1), 98-103.

    - Original Verlet algorithm paper
    - Foundation of modern MD integration
    - Historic significance in computational physics

16. **Swope, W. C., Andersen, H. C., Berens, P. H., & Wilson, K. R.** (1982). A computer simulation method for the calculation of equilibrium constants for the formation of physical clusters of molecules: Application to small water clusters. *Journal of Chemical Physics*, 76(1), 637-649.

    - Introduction of velocity Verlet algorithm
    - Improved stability and energy conservation
    - Widely adopted in MD software

**Constraint Algorithms**

17. **Ryckaert, J. P., Ciccotti, G., & Berendsen, H. J.** (1977). Numerical integration of the cartesian equations of motion of a system with constraints: molecular dynamics of n-alkanes. *Journal of Computational Physics*, 23(3), 327-341.

    - SHAKE algorithm development
    - Essential for constraining bond lengths
    - Allows larger timesteps in MD simulations

18. **Andersen, H. C.** (1983). Rattle: A "velocity" version of the shake algorithm for molecular dynamics calculations. *Journal of Computational Physics*, 52(1), 24-34.

    - RATTLE algorithm for velocity constraints
    - Consistent with velocity Verlet integration
    - Improved accuracy for constrained dynamics

Enhanced Sampling Methods
-------------------------

Replica Exchange Methods
~~~~~~~~~~~~~~~~~~~~~~~~

**Temperature Replica Exchange**

19. **Sugita, Y., & Okamoto, Y.** (1999). Replica-exchange molecular dynamics method for protein folding. *Chemical Physics Letters*, 314(1-2), 141-151.

    - Introduction of replica exchange MD
    - Revolutionary method for enhanced sampling
    - Widely applied to protein folding studies

20. **Earl, D. J., & Deem, M. W.** (2005). Parallel tempering: theory, applications, and new perspectives. *Physical Chemistry Chemical Physics*, 7(23), 3910-3916.

    - Comprehensive review of parallel tempering
    - Theoretical foundations and practical applications
    - Guidelines for optimal implementation

**Hamiltonian Replica Exchange**

21. **Fukunishi, H., Watanabe, O., & Takada, S.** (2002). On the Hamiltonian replica exchange method for efficient sampling of biomolecular systems: application to protein structure prediction. *Journal of Chemical Physics*, 116(20), 9058-9067.

    - Hamiltonian replica exchange development
    - Application to protein structure prediction
    - Alternative to temperature-based methods

Metadynamics and Free Energy Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Metadynamics**

22. **Laio, A., & Parrinello, M.** (2002). Escaping free-energy minima. *Proceedings of the National Academy of Sciences*, 99(20), 12562-12566.

    - Original metadynamics paper
    - Breakthrough in free energy landscape exploration
    - Foundation for modern enhanced sampling

23. **Barducci, A., Bussi, G., & Parrinello, M.** (2008). Well-tempered metadynamics: a smoothly converging and tunable free-energy method. *Physical Review Letters*, 100(2), 020603.

    - Well-tempered metadynamics development
    - Improved convergence properties
    - Widely adopted enhancement

**Umbrella Sampling**

24. **Torrie, G. M., & Valleau, J. P.** (1977). Nonphysical sampling distributions in Monte Carlo free-energy estimation: umbrella sampling. *Journal of Computational Physics*, 23(2), 187-199.

    - Original umbrella sampling method
    - Fundamental technique for free energy calculation
    - Basis for many modern methods

25. **Kumar, S., Rosenberg, J. M., Bouzida, D., Swendsen, R. H., & Kollman, P. A.** (1992). The weighted histogram analysis method for free-energy calculations on biomolecules. I. The method. *Journal of Computational Chemistry*, 13(8), 1011-1021.

    - WHAM method development
    - Optimal combination of umbrella sampling data
    - Standard analysis technique

Statistical Mechanics and Ensemble Theory
------------------------------------------

Ensemble Methods
~~~~~~~~~~~~~~~~

**Thermostats and Barostats**

26. **Nosé, S.** (1984). A molecular dynamics method for simulations in the canonical ensemble. *Molecular Physics*, 52(2), 255-268.

    - Nosé thermostat development
    - Rigorous canonical ensemble sampling
    - Theoretical foundation for temperature control

27. **Hoover, W. G.** (1985). Canonical dynamics: equilibrium phase-space distributions. *Physical Review A*, 31(3), 1695-1697.

    - Nosé-Hoover thermostat formulation
    - Practical implementation of canonical sampling
    - Widely used in MD simulations

28. **Parrinello, M., & Rahman, A.** (1981). Polymorphic transitions in single crystals: A new molecular dynamics method. *Journal of Applied Physics*, 52(12), 7182-7190.

    - Parrinello-Rahman barostat development
    - Constant pressure MD simulation
    - Flexible unit cell dynamics

**Langevin Dynamics**

29. **Pastor, R. W., Brooks, B. R., & Szabo, A.** (1988). An analysis of the accuracy of Langevin and molecular dynamics algorithms. *Molecular Physics*, 65(6), 1409-1419.

    - Analysis of Langevin dynamics accuracy
    - Comparison with standard MD
    - Important for understanding stochastic dynamics

Biomolecular Applications
-------------------------

Protein Dynamics and Folding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Pioneering Studies**

30. **McCammon, J. A., Gelin, B. R., & Karplus, M.** (1977). Dynamics of folded proteins. *Nature*, 267(5612), 585-590.

    - First MD simulation of a protein
    - Historic milestone in computational biology
    - Demonstrated feasibility of biomolecular MD

31. **Levitt, M., & Warshel, A.** (1975). Computer simulation of protein folding. *Nature*, 253(5494), 694-698.

    - Early protein folding simulation
    - Pioneering work in computational structural biology
    - Foundation for modern folding studies

**Modern Long-Timescale Studies**

32. **Shaw, D. E., Maragakis, P., Lindorff-Larsen, K., Piana, S., Dror, R. O., Eastwood, M. P., ... & Wriggers, W.** (2010). Atomic-level characterization of the structural dynamics of proteins. *Science*, 330(6002), 341-346.

    - Millisecond-timescale protein simulations
    - Specialized hardware (Anton) applications
    - Major advance in accessible timescales

Membrane Simulations
~~~~~~~~~~~~~~~~~~~~

**Lipid Bilayer Studies**

33. **Marrink, S. J., Risselada, H. J., Yefimov, S., Tieleman, D. P., & De Vries, A. H.** (2007). The MARTINI force field: coarse grained model for biomolecular simulations. *Journal of Physical Chemistry B*, 111(27), 7812-7824.

    - MARTINI coarse-grained force field
    - Enables large-scale membrane simulations
    - Widely used for lipid systems

34. **Tieleman, D. P., & Berendsen, H. J.** (1996). Molecular dynamics simulations of a fully hydrated dipalmitoylphosphatidylcholine bilayer with different macroscopic boundary conditions and parameters. *Journal of Chemical Physics*, 105(11), 4871-4880.

    - Comprehensive lipid bilayer MD study
    - Validation of simulation parameters
    - Reference for membrane simulations

DNA and RNA Simulations
~~~~~~~~~~~~~~~~~~~~~~~

**Nucleic Acid Force Fields**

35. **Cheatham III, T. E., Cieplak, P., & Kollman, P. A.** (1999). A modified version of the Cornell et al. force field with improved sugar pucker phases and helical repeat. *Journal of Biomolecular Structure and Dynamics*, 16(4), 845-862.

    - Improved nucleic acid parameters
    - Better representation of sugar conformations
    - Foundation for modern RNA/DNA simulations

36. **Pérez, A., Marchán, I., Svozil, D., Sponer, J., Cheatham III, T. E., Laughton, C. A., & Orozco, M.** (2007). Refinement of the AMBER force field for nucleic acids: improving the description of α/γ conformers. *Biophysical Journal*, 92(11), 3817-3829.

    - Parmbsc0 force field development
    - Improved backbone conformations
    - Better agreement with experimental data

Free Energy Calculations
-------------------------

Alchemical Methods
~~~~~~~~~~~~~~~~~~

**Free Energy Perturbation**

37. **Zwanzig, R. W.** (1954). High‐temperature equation of state by a perturbation method. I. Nonpolar gases. *Journal of Chemical Physics*, 22(8), 1420-1426.

    - Zwanzig equation derivation
    - Theoretical foundation for FEP
    - Classic statistical mechanics result

38. **Jorgensen, W. L., & Ravimohan, C.** (1985). Monte Carlo simulation of differences in free energies of hydration. *Journal of Chemical Physics*, 83(6), 3050-3054.

    - Early application of FEP to solvation
    - Demonstration of method viability
    - Important validation study

**Thermodynamic Integration**

39. **Kirkwood, J. G.** (1935). Statistical mechanics of fluid mixtures. *Journal of Chemical Physics*, 3(5), 300-313.

    - Kirkwood coupling parameter method
    - Theoretical basis for thermodynamic integration
    - Fundamental statistical mechanics

**Bennett Acceptance Ratio**

40. **Bennett, C. H.** (1976). Efficient estimation of free energy differences from Monte Carlo data. *Journal of Computational Physics*, 22(2), 245-268.

    - Bennett acceptance ratio method
    - Optimal free energy estimation
    - Widely used in modern calculations

Software and Implementation
---------------------------

MD Software Packages
~~~~~~~~~~~~~~~~~~~~

**GROMACS**

41. **Abraham, M. J., Murtola, T., Schulz, R., Páll, S., Smith, J. C., Hess, B., & Lindahl, E.** (2015). GROMACS: High performance molecular simulations through multi-level parallelism from laptops to supercomputers. *SoftwareX*, 1, 19-25.

    - Modern GROMACS overview
    - High-performance implementation details
    - Widely used open-source package

**AMBER**

42. **Case, D. A., Cheatham III, T. E., Darden, T., Gohlke, H., Luo, R., Merz Jr, K. M., ... & Woods, R. J.** (2005). The Amber biomolecular simulation programs. *Journal of Computational Chemistry*, 26(16), 1668-1688.

    - AMBER software suite overview
    - Comprehensive biomolecular simulation tools
    - Widely used in biological applications

**OpenMM**

43. **Eastman, P., Swails, J., Chodera, J. D., McGibbon, R. T., Zhao, Y., Beauchamp, K. A., ... & Pande, V. S.** (2017). OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. *PLoS Computational Biology*, 13(7), e1005659.

    - Modern OpenMM framework
    - GPU-accelerated simulations
    - Flexible and extensible platform

Performance and Hardware
~~~~~~~~~~~~~~~~~~~~~~~~

**GPU Computing**

44. **Stone, J. E., Phillips, J. C., Freddolino, P. L., Hardy, D. J., Trabuco, L. G., & Schulten, K.** (2007). Accelerating molecular modeling applications with the CUDA programming model. *Journal of Computational Chemistry*, 28(16), 2618-2640.

    - Early GPU acceleration work
    - Demonstration of GPU potential for MD
    - Foundation for modern GPU computing

**Specialized Hardware**

45. **Shaw, D. E., Deneroff, M. M., Dror, R. O., Kuskin, J. S., Larson, R. H., Salmon, J. K., ... & Bank, J. A.** (2008). Anton, a special-purpose machine for molecular dynamics simulation. *Communications of the ACM*, 51(7), 91-97.

    - Anton supercomputer development
    - Specialized MD hardware design
    - Breakthrough in simulation timescales

Analysis Methods
----------------

Trajectory Analysis
~~~~~~~~~~~~~~~~~~~

**Principal Component Analysis**

46. **Amadei, A., Linssen, A. B., & Berendsen, H. J.** (1993). Essential dynamics of proteins. *Proteins: Structure, Function, and Bioinformatics*, 17(4), 412-425.

    - Essential dynamics/PCA for proteins
    - Dimensionality reduction in MD analysis
    - Widely used analysis technique

**Clustering Methods**

47. **Daura, X., Gademann, K., Jaun, B., Seebach, D., Van Gunsteren, W. F., & Mark, A. E.** (1999). Peptide folding: when simulation meets experiment. *Angewandte Chemie International Edition*, 38(1‐2), 236-240.

    - Clustering analysis for MD trajectories
    - Conformational state identification
    - Important analysis methodology

Network Analysis
~~~~~~~~~~~~~~~~

**Dynamical Networks**

48. **Sethi, A., Eargle, J., Black, A. A., & Luthey-Schulten, Z.** (2009). Dynamical networks in tRNA: protein complexes. *Proceedings of the National Academy of Sciences*, 106(16), 6620-6625.

    - Network analysis of protein dynamics
    - Allosteric communication pathways
    - Modern analysis approach

Machine Learning Applications
-----------------------------

Enhanced Sampling with ML
~~~~~~~~~~~~~~~~~~~~~~~~~

**Collective Variable Discovery**

49. **Sultan, M. M., & Pande, V. S.** (2018). Automated design of collective variables using supervised machine learning. *Journal of Chemical Physics*, 149(9), 094106.

    - ML-guided collective variable selection
    - Automated enhanced sampling
    - Modern approach to rare event sampling

**Neural Network Potentials**

50. **Behler, J., & Parrinello, M.** (2007). Generalized neural-network representation of high-dimensional potential-energy surfaces. *Physical Review Letters*, 98(14), 146401.

    - Neural network potential development
    - ML-based force field construction
    - Foundation for modern ML potentials

Markov State Models
~~~~~~~~~~~~~~~~~~~

51. **Pande, V. S., Beauchamp, K., & Bowman, G. R.** (2010). Everything you wanted to know about Markov State Models but were afraid to ask. *Methods*, 52(1), 99-105.

    - Comprehensive MSM overview
    - Kinetic modeling of MD data
    - Important analysis framework

Quantum Mechanics/Molecular Mechanics
--------------------------------------

QM/MM Theory and Implementation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

52. **Warshel, A., & Levitt, M.** (1976). Theoretical studies of enzymic reactions: dielectric, electrostatic and steric stabilization of the carbonium ion in the reaction of lysozyme. *Journal of Molecular Biology*, 103(2), 227-249.

    - Original QM/MM method
    - Nobel Prize-winning work
    - Foundation of hybrid methods

53. **Field, M. J., Bash, P. A., & Karplus, M.** (1990). A combined quantum mechanical and molecular mechanical potential for molecular dynamics simulations. *Journal of Computational Chemistry*, 11(6), 700-733.

    - Practical QM/MM implementation
    - MD with QM/MM potentials
    - Important methodological development

Specialized Topics
------------------

Coarse-Grained Modeling
~~~~~~~~~~~~~~~~~~~~~~~

54. **Voth, G. A.** (Ed.). (2008). *Coarse-graining of condensed phase and biomolecular systems*. CRC press.

    - Comprehensive coarse-graining methods
    - Theory and applications
    - Essential reference for CG modeling

Constant pH Simulations
~~~~~~~~~~~~~~~~~~~~~~~

55. **Mongan, J., Case, D. A., & McCammon, J. A.** (2004). Constant pH molecular dynamics in generalized Born implicit solvent. *Journal of Computational Chemistry*, 25(16), 2038-2048.

    - Constant pH MD implementation
    - Important for biological systems
    - Advanced simulation technique

Polarizable Force Fields
~~~~~~~~~~~~~~~~~~~~~~~~

56. **Lopes, P. E., Huang, J., Shim, J., Luo, Y., Li, H., Roux, B., & MacKerell Jr, A. D.** (2013). Polarizable force field for peptides and proteins based on the classical Drude oscillator. *Journal of Chemical Theory and Computation*, 9(12), 5430-5449.

    - Drude polarizable force field
    - Advanced electrostatic treatment
    - Next-generation force field development

Historical and Foundational Papers
-----------------------------------

Early Computational Physics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

57. **Alder, B. J., & Wainwright, T. E.** (1957). Phase transition for a hard sphere system. *Journal of Chemical Physics*, 27(5), 1208-1209.

    - First molecular dynamics simulation
    - Historic breakthrough in computational physics
    - Foundation of the entire field

58. **Rahman, A.** (1964). Correlations in the motion of atoms in liquid argon. *Physical Review*, 136(2A), A405-A411.

    - First simulation of a realistic system
    - Liquid argon with Lennard-Jones potential
    - Demonstrated viability of MD for liquids

Statistical Mechanics Foundations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

59. **Boltzmann, L.** (1872). Weitere studien über das wärmegleichgewicht unter gasmolekülen. *Wiener Berichte*, 66, 275-370.

    - Boltzmann equation and H-theorem
    - Fundamental statistical mechanics
    - Theoretical foundation for all MD

60. **Gibbs, J. W.** (1902). *Elementary principles in statistical mechanics*. Yale University Press.

    - Gibbs ensemble theory
    - Foundation of statistical mechanics
    - Essential theoretical background

Modern Developments and Future Directions
------------------------------------------

Recent Advances
~~~~~~~~~~~~~~~

61. **Noé, F., Olsson, S., Köhler, J., & Wu, H.** (2019). Boltzmann generators: Sampling equilibrium states of many-body systems with deep learning. *Science*, 365(6457), eaaw1147.

    - Deep learning for equilibrium sampling
    - Modern ML applications to MD
    - Future direction for the field

62. **Wang, J., Olsson, S., Wehmeyer, C., Pérez, A., Charron, N. E., De Fabritiis, G., ... & Clementi, C.** (2019). Machine learning of coarse-grained molecular dynamics force fields. *ACS Central Science*, 5(5), 755-767.

    - ML-based coarse-grained force fields
    - Systematic multiscale modeling
    - Modern computational approaches

Methodological Reviews
~~~~~~~~~~~~~~~~~~~~~~

63. **Shirts, M. R., & Chodera, J. D.** (2008). Statistically optimal analysis of samples from multiple equilibrium states. *Journal of Chemical Physics*, 129(12), 124105.

    - MBAR method development
    - Optimal statistical analysis
    - Important for free energy calculations

64. **Henin, J., Fiorin, G., Chipot, C., & Klein, M. L.** (2010). Exploring multidimensional free energy landscapes using time-dependent biases on collective variables. *Journal of Chemical Theory and Computation*, 6(1), 35-47.

    - Adaptive biasing force method
    - Enhanced sampling technique
    - Modern free energy calculations

Specialized Journals and Resources
----------------------------------

Key Journals
~~~~~~~~~~~~

**Primary Research Journals**

- *Journal of Chemical Physics* - Premier venue for MD methodology
- *Journal of Chemical Theory and Computation* - Focus on computational methods
- *Journal of Computational Chemistry* - Software and applications
- *Biophysical Journal* - Biological applications
- *Physical Review Letters* - Breakthrough results
- *Nature*, *Science* - High-impact applications

**Specialized Publications**

- *Computer Physics Communications* - Software and code sharing
- *Journal of Molecular Modeling* - Applied computational chemistry
- *Proteins: Structure, Function, and Bioinformatics* - Protein simulations
- *Journal of Biomolecular Structure and Dynamics* - Biomolecular focus

Online Resources
~~~~~~~~~~~~~~~~

**Educational Materials**

- *Molecular Dynamics Tutorials* (various universities)
- *GROMACS Documentation and Tutorials*
- *AMBER Manual and Tutorials*
- *OpenMM Documentation*

**Databases and Repositories**

- *Protein Data Bank (PDB)* - Structural data
- *CHARMM-GUI* - System setup tools
- *Martini Portal* - Coarse-grained models
- *Force Field Repository* - Parameter databases

Research Groups and Centers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Leading Research Groups**

- *Shaw Research* - Long-timescale simulations
- *D.E. Shaw Research* - Specialized hardware
- *Various university groups* - Methodology development

**International Collaborations**

- *CECAM* - European Center for Atomic and Molecular Calculations
- *MolSSI* - Molecular Sciences Software Institute
- *BioExcel* - European Centre of Excellence for Computational Biomolecular Research

This literature collection provides a comprehensive foundation for understanding the theory, methods, and applications of molecular dynamics simulations. The references span from foundational work to cutting-edge developments, supporting both learning and research in computational molecular science.
