*******************
Existing Potentials
*******************

Here we see the script for the first example with existing potentials:

.. code-block:: console

   /vibrations/examples/
            ├── 1.Existing_potentials/ 
                     ├── 1D.npy 
                     ├── 2D.npy
                     ├── Example_VCI.py
                     └── grids.py

Here we see the script for the first part of the example *Example_VCI.py*:

.. literalinclude:: ../examples/1.Existing_potentials/Example1_VCI.py
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
    *
   .
   .
   .
    *
   +------------------+
   |--- Example 1: ---|
   +------------------+
   
   Use of existing grids and potentials.
   
     Data taken from: http://pes-database.theochem.uni-stuttgart.de/surfaces/index.php
     By Guntram Rauhut and co-workers
   
   +--------------------+
   |--- Solving VSCF ---|
   +--------------------+
   
   +-----------------------------------------------------------+
   |--- Solving State: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] ---|
   +-----------------------------------------------------------+
   
   Iteration: 1
   Mode State   Eigv
      1     0    408.2
      2     0    464.6
      3     0    470.0
      4     0    512.1
      5     0    614.7
      6     0    671.9
      7     0    722.9
      8     0    820.5
      9     0   1542.1
     10     0   1534.4
     11     0   1578.1
     12     0   1591.4
   Sum of eigenvalues 10930.9, SCF correction -169.4, total energy 11100.3 / cm^-1
   .
   .
   .
   Iteration: 7
   Mode State   Eigv
      1     0    403.8
      2     0    457.5
      3     0    463.8
      4     0    504.8
      5     0    606.8
      6     0    658.4
      7     0    711.9
      8     0    812.4
      9     0   1519.1
     10     0   1443.0
     11     0   1552.4
     12     1   4591.1
   Sum of eigenvalues 13725.0, SCF correction -446.7, total energy 14171.7 / cm^-1
   
   
   +-----------------+
   |--- VSCF Done ---|
   +-----------------+
   
   
   
   +--------------------+
   
   +--------------------+
   |--- VSCF Results ---|
   +--------------------+
   
   VSCF states energies in cm^-1
   ---------------------------
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 11060.2
   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 11897.9
   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 12005.2
   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 12018.6
   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] 12092.8
   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] 12293.0
   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 12408.7
   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] 12509.1
   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] 12698.3
   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] 14079.4
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] 14117.0
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] 14144.7
   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 14171.7
   
   Initial state:  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
   Transition energies in cm^-1
   ->  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 837.7
   ->  [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] 944.9
   ->  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] 958.3
   ->  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0] 1032.6
   ->  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0] 1232.8
   ->  [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0] 1348.5
   ->  [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0] 1448.9
   ->  [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] 1638.1
   ->  [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0] 3019.2
   ->  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0] 3056.7
   ->  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0] 3084.4
   ->  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] 3111.4
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
   [0 0 0 0 0 0 0 0 0 0 0 0]     1.0000 11060.1362     0.0000
   [1 0 0 0 0 0 0 0 0 0 0 0]     1.0000 11906.7657   846.6294
   [0 1 0 0 0 0 0 0 0 0 0 0]     1.0000 12010.9518   950.8155
   [0 0 1 0 0 0 0 0 0 0 0 0]     1.0000 12025.8865   965.7502
   [0 0 0 1 0 0 0 0 0 0 0 0]     1.0000 12096.9668  1036.8306
   [0 0 0 0 1 0 0 0 0 0 0 0]     1.0000 12294.6914  1234.5552
   [0 0 0 0 0 1 0 0 0 0 0 0]     0.9978 12409.8796  1349.7433
   [0 0 0 0 0 0 1 0 0 0 0 0]     1.0000 12510.6109  1450.4747
   [0 0 0 0 0 0 0 1 0 0 0 0]     0.9978 12702.3966  1642.2604
   [0 0 0 0 0 0 0 0 0 1 0 0]     1.0000 14117.9949  3057.8587
   [0 0 0 0 0 0 0 0 1 0 0 0]     1.0000 14119.3745  3059.2383
   [0 0 0 0 0 0 0 0 0 0 1 0]     1.0000 14190.6644  3130.5281
   [0 0 0 0 0 0 0 0 0 0 0 1]     1.0000 14216.6420  3156.5057
   
   
   
   +------------------------------------+
   |--- http://www.christophjacob.eu ---|
   +------------------------------------+
