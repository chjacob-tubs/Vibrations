************************
Localized Harmonic Modes
************************

Here we see the script for localized harmonic modes:

.. code-block:: console

   /vibrations/examples/
            ├── 2.Localized_modes_harmonic/ 
                     ├── coord
                     ├── Example2_Harmonic_L-VCI-S.py
                     ├── restart
                     └── snf.out

Here we see the script for the first part of the example *Example2_Harmonic_L-VCI-S.py*:

.. literalinclude:: ../examples/2.Localized_modes_harmonic/Example2_Harmonic_L-VCI-S.py
    :language: python

When we run the script we get:

.. code-block:: console

   WARNING: Fortran routines not used for integrals,     this might be very slow.
            see src/Vibrations/README_f2py 
   
    ***************************************************************************
    *
    *  Vibrations v0.96
    *
    *  Vibrations - a Python Code for Anharmonic Theoretical Vibrational Spectroscopy
    *  Copyright (C) 2014-2023 by Pawel T. Panek, Adrian A. Hoeske, Julia Brüggemann,
    *  Michael Welzel, and Christoph R. Jacob.
                            .
                            .
                            .
    *  In scientific publications using Vibrations please cite:
    *    P. T. Panek, Ch. R. Jacob, ChemPhysChem 15 (2014) 3365.
    *    P. T. Panek, Ch. R. Jacob, J. Chem. Phys. 144 (2016) 164111.
    *
    *  The most recent version of Vibrations is available at
    *    http://www.christophjacob.eu/software
    *
    ***************************************************************************
    *
    *
   +------------------+
   |--- Example 2: ---|
   +------------------+

   Localization of modes in subsets, harmonic L-VCI-S
   provides initial normal modes' frequencies and intensities
   
    See Section VI A of  J. Phys. Chem. 144 (2016) 164111
    for a description of a similar example.
   
   
   Modes localized: 12, modes in total: 12
   Normal mode localization: Cycle   1    p:    1.423   change:  0.5793322     1.71249
   Normal mode localization: Cycle   2    p:    1.430   change:  0.0067229     0.19422
   Normal mode localization: Cycle   3    p:    1.438   change:  0.0084692     0.19886
   Normal mode localization: Cycle   4    p:    1.442   change:  0.0033325     0.12330
   Normal mode localization: Cycle   5    p:    1.443   change:  0.0012660     0.06618
   Normal mode localization: Cycle   6    p:    1.444   change:  0.0005905     0.04309
   Normal mode localization: Cycle   7    p:    1.444   change:  0.0003128     0.03171
   Normal mode localization: Cycle   8    p:    1.444   change:  0.0001826     0.02443
                            .
                            .
                            .
   Normal mode localization: Cycle  15    p:    1.632   change:  0.0000000     0.00024
   Normal mode localization: Cycle  16    p:    1.632   change:  0.0000000     0.00013
   Normal mode localization: Cycle  17    p:    1.632   change:  0.0000000     0.00007
   Obtaining coupling matrix in [a.u.]
   Obtaining coupling matrix in [a.u.]
   Obtaining coupling matrix in [a.u.]
   Obtaining coupling matrix in [a.u.]
   Normal mode localization: Cycle   1    p:    2.955   change:  2.0932835     3.25317
   Normal mode localization: Cycle   2    p:    3.410   change:  0.4544177     0.66040
   Normal mode localization: Cycle   3    p:    3.411   change:  0.0013718     0.02516
   Normal mode localization: Cycle   4    p:    3.411   change:  0.0000000     0.00000
   Obtaining coupling matrix in [a.u.]
   Obtaining coupling matrix in [a.u.]
   Obtaining coupling matrix in [a.u.]
   Obtaining coupling matrix in [a.u.]
   
   +--------------------+
   |--- Solving VSCF ---|
   +--------------------+

   +-----------------------------------------------------------+
   |--- Solving State: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ---|
   +-----------------------------------------------------------+
   
   Iteration: 1
   Mode State   Eigv
      1     0    496.4
      2     0    482.8
      3     0    482.8
      4     0    408.3
      5     0    796.2
      6     0    710.2
      7     0    660.0
      8     0    656.6
      9     0   1553.1
     10     0   1553.1
     11     0   1552.7
     12     0   1552.7
   Sum of eigenvalues 10904.8, SCF correction -0.0, total energy 10904.8 / cm^-1
   
   Iteration: 2
   Mode State   Eigv
      1     0    496.4
      2     0    482.8
      3     0    482.8
      4     0    408.3
      5     0    796.2
      6     0    710.2
      7     0    660.0
      8     0    656.6
      9     0   1553.1
     10     0   1553.1
     11     0   1552.7
     12     0   1552.7
   Sum of eigenvalues 10904.8, SCF correction -0.0, total energy 10904.8 / cm^-1
   
   +-----------------+
   |--- VSCF Done ---|
   +-----------------+
   
   
   
   +--------------------+
   |--- VSCF Results ---|
   +--------------------+
   
   VSCF states energies in cm^-1
   ---------------------------
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 10904.8
   
   Initial state:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   Transition energies in cm^-1
   WARNING: Fortran routines not used for integrals,             this might be very slow.
            see src/Vibrations/README_f2py
   +---------------------------+
   |--- There are 13 states ---|
   +---------------------------+
   
   combgenerator 0 of 13
   +---------------------------------------------------------+
   |--- Hamiltonian matrix constructed.Diagonalization... ---|
   +---------------------------------------------------------+
   
   +--------------------------+
   |--- Results of the VCI ---|
   +--------------------------+

   State        Contrib   E /cm^-1  DE /cm^-1
   [0 0 0 0 0 0 0 0 0 0 0 0]     1.0000 10904.3846     0.0000
   [0 0 0 1 0 0 0 0 0 0 0 0]     1.0000 11720.9193   816.5347
   [0 0 1 0 0 0 0 0 0 0 0 0]     0.5001 11845.6242   941.2397
   [1 0 0 0 0 0 0 0 0 0 0 0]     0.4851 11849.7270   945.3424
   [1 0 0 0 0 0 0 0 0 0 0 0]     0.5149 11941.5801  1037.1956
   [0 0 0 0 0 0 0 1 0 0 0 0]     0.5144 12115.5214  1211.1368
   [0 0 0 0 0 1 0 0 0 0 0 0]     0.4020 12255.0293  1350.6447
   [0 0 0 0 0 1 0 0 0 0 0 0]     0.5035 12337.5679  1433.1833
   [0 0 0 0 1 0 0 0 0 0 0 0]     0.8057 12554.9042  1650.5196
   [0 0 0 0 0 0 0 0 0 0 1 0]     0.2524 13964.4593  3060.0747
   [0 0 0 0 0 0 0 0 0 0 0 1]     0.2526 13977.1676  3072.7830
   [0 0 0 0 0 0 0 0 0 1 0 0]     0.2526 14034.7354  3130.3509
   [0 0 0 0 0 0 0 0 1 0 0 0]     0.2524 14063.0973  3158.7127
   
   +--------------------------+
   |--- VCI IR Intensities ---|
   +--------------------------+
   
   Only one set of dipole moments given.
   
     Freq.    Int.
   [cm^-1] [km*mol^-1]
     816.5     1.550986
     941.2     0.000001
     945.3    97.028918
    1037.2     0.006884
    1211.1     0.000000
    1350.6     0.001040
    1433.2    16.559495
    1650.5     0.008138
    3060.1    18.564699
    3072.8     0.000000
    3130.4     0.000000
    3158.7    22.176752

    +---------------+
    |--- Results ---|
    +---------------+
    
              Normal           Localized          L-VCI-S
    No      Freq.    Int      Freq.    Int      Freq.    Int
    --------------------------------------------------------
     1      816.6    1.6      991.7   48.6      816.5    1.6
     2      941.0    0.0      964.8   25.9      941.2    0.0
     3      944.7  100.4      964.8   25.9      945.3   97.0
     4     1035.5    0.0      816.6    1.6     1037.2    0.0
     5     1206.6    0.0     1588.0    0.0     1211.1    0.0
     6     1345.9    0.0     1417.9    8.0     1350.6    0.0
     7     1430.8   15.8     1314.4    4.2     1433.2   16.6
     8     1644.6    0.0     1307.7    3.6     1650.5    0.0
     9     3059.9   18.8     3105.8   10.1     3060.1   18.6
    10     3072.8    0.0     3105.8   10.1     3072.8    0.0
    11     3130.4    0.0     3105.0   10.3     3130.4    0.0
    12     3158.4   21.8     3105.0   10.3     3158.7   22.2
    
    
    +------------------------------------+
    |--- http://www.christophjacob.eu ---|
    +------------------------------------+
