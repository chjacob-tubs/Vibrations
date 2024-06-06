*****************************
Step by Step Normal Modes H2O
*****************************

Here we see the script for localized harmonic modes:

.. code-block:: console

   /vibrations/examples/
            ├── 4.StepByStep_H2O_NM/ 
                     ├── A_h2o_make_normalmode_grid.py
                     ├── B_calculate_singlepoints.README
                     ├── C_make_potentials.py
                     ├── D_vscf_vci.py
                     ├── energies/
                     ├── grid/
                     ├── potentials/
                     ├── README
                     └── snf_h2o/


Instructions
============

.. include:: ../examples/4.StepByStep_H2O_NM/README
   :literal:


Step A - Generate Grid
=======================

.. literalinclude:: ../examples/4.StepByStep_H2O_NM/A_h2o_make_normalmode_grid.py

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
    *     Vibrations is free software: you can redistribute it and/or modify
    *     it under the terms of the GNU General Public License as published by
    *     the Free Software Foundation, either version 3 of the License, or
    *     (at your option) any later version.
    *
    *     Vibrations is distributed in the hope that it will be useful,
    *     but WITHOUT ANY WARRANTY;      without even the implied warranty of
    *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    *     GNU General Public License for more details.
    *
    *     You should have received a copy of the GNU General Public License
    *     along with Vibrations.  If not, see <http://www.gnu.org/licenses/>.
    *
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
    *** Writing  Mode:  0 , Point:  0
    *** Writing  Mode:  0 , Point:  1
    *** Writing  Mode:  0 , Point:  2
    *** Writing  Mode:  0 , Point:  3
    *** Writing  Mode:  0 , Point:  4
    *** Writing  Mode:  0 , Point:  5
    *** Writing  Mode:  0 , Point:  6
    *** Writing  Mode:  0 , Point:  7
    *** Writing  Mode:  0 , Point:  8
    *** Writing  Mode:  0 , Point:  9
    *** Writing  Mode:  0 , Point:  10
    *** Writing  Mode:  0 , Point:  11
    *** Writing  Mode:  0 , Point:  12
    *** Writing  Mode:  0 , Point:  13
    *** Writing  Mode:  0 , Point:  14
    *** Writing  Mode:  0 , Point:  15
   
    *** Writing  Mode:  1 , Point:  0
    *** Writing  Mode:  1 , Point:  1
    *** Writing  Mode:  1 , Point:  2
    *** Writing  Mode:  1 , Point:  3
    *** Writing  Mode:  1 , Point:  4
    *** Writing  Mode:  1 , Point:  5
    *** Writing  Mode:  1 , Point:  6
    *** Writing  Mode:  1 , Point:  7
    *** Writing  Mode:  1 , Point:  8
    *** Writing  Mode:  1 , Point:  9
    *** Writing  Mode:  1 , Point:  10
    *** Writing  Mode:  1 , Point:  11
    *** Writing  Mode:  1 , Point:  12
    *** Writing  Mode:  1 , Point:  13
    *** Writing  Mode:  1 , Point:  14
    *** Writing  Mode:  1 , Point:  15
   
    *** Writing  Mode:  2 , Point:  0
    *** Writing  Mode:  2 , Point:  1
    *** Writing  Mode:  2 , Point:  2
    *** Writing  Mode:  2 , Point:  3
    *** Writing  Mode:  2 , Point:  4
    *** Writing  Mode:  2 , Point:  5
    *** Writing  Mode:  2 , Point:  6
    *** Writing  Mode:  2 , Point:  7
    *** Writing  Mode:  2 , Point:  8
    *** Writing  Mode:  2 , Point:  9
    *** Writing  Mode:  2 , Point:  10
    *** Writing  Mode:  2 , Point:  11
    *** Writing  Mode:  2 , Point:  12
    *** Writing  Mode:  2 , Point:  13
    *** Writing  Mode:  2 , Point:  14
    *** Writing  Mode:  2 , Point:  15
   
    *** Writing V2 - Mode: 0 1 , Point:  0 0
    *** Writing V2 - Mode: 0 1 , Point:  0 1
    ** Writing V2 - Mode: 0 2 , Point:  1 6
    *** Writing V2 - Mode: 0 2 , Point:  1 7
    *** Writing V2 - Mode: 0 2 , Point:  1 8
    *** Writing V2 - Mode: 0 2 , Point:  1 9
    *** Writing V2 - Mode: 0 2 , Point:  1 10
    *** Writing V2 - Mode: 0 2 , Point:  1 11
    *** Writing V2 - Mode: 0 2 , Point:  1 12
    *** Writing V2 - Mode: 0 2 , Point:  1 13
    *** Writing V2 - Mode: 0 2 , Point:  1 14
                       .
                       .
                       .
    *** Writing V2 - Mode: 0 2 , Point:  1 15
    *** Writing V2 - Mode: 0 2 , Point:  2 0
    *** Writing V2 - Mode: 0 2 , Point:  2 1
    *** Writing V2 - Mode: 0 2 , Point:  2 2
    *** Writing V2 - Mode: 0 2 , Point:  2 3
    *** Writing V2 - Mode: 0 2 , Point:  2 4
    *** Writing V2 - Mode: 0 2 , Point:  2 5
    *** Writing V2 - Mode: 0 2 , Point:  2 6
    *** Writing V2 - Mode: 0 2 , Point:  2 7
    *** Writing V2 - Mode: 0 2 , Point:  2 8
    *** Writing V2 - Mode: 0 2 , Point:  2 9
    *** Writing V2 - Mode: 0 2 , Point:  2 10
    *** Writing V2 - Mode: 0 2 , Point:  2 11
    *** Writing V2 - Mode: 0 2 , Point:  2 12
    *** Writing V2 - Mode: 0 2 , Point:  2 13
    *** Writing V2 - Mode: 0 2 , Point:  2 14
    *** Writing V2 - Mode: 0 2 , Point:  2 15
    *** Writing V2 - Mode: 0 2 , Point:  3 0
    *** Writing V2 - Mode: 0 2 , Point:  3 1
    *** Writing V2 - Mode: 0 2 , Point:  3 2
    *** Writing V2 - Mode: 0 2 , Point:  3 3
    *** Writing V2 - Mode: 0 2 , Point:  3 4
    *** Writing V2 - Mode: 0 2 , Point:  3 5
    *** Writing V2 - Mode: 0 2 , Point:  3 6
    *** Writing V2 - Mode: 0 2 , Point:  3 7
    *** Writing V2 - Mode: 0 2 , Point:  3 8
                       .
                       .
                       .
    etc. etc. etc.

Step B - Single Points Calculations
===================================

.. include:: ../examples/4.StepByStep_H2O_NM/B_calculate_singlepoints.README
   :literal:


Step C - Generate 1-Mode and 2-Mode Potentials
==============================================

.. literalinclude:: ../examples/4.StepByStep_H2O_NM/C_make_potentials.py

When we run the script we get:

.. code-block:: console

   ['-76.3451979888', '0.0', '0.0', '-2.14554110001']
   Energy and dipole moment -76.3451979888 0.0 0.0 -2.14554110001
   Mode : 2 14
   ['-76.0770008458', '0.32375287021', '0.0', '-2.15976484493']
   Energy and dipole moment -76.0770008458 0.32375287021 0.0 -2.15976484493
   Mode : 1 13
   ['-76.079552841', '0.0', '0.0', '-1.99131666349']
   Energy and dipole moment -76.079552841 0.0 0.0 -1.99131666349
   Mode : 1 3
   ['-76.2936267945', '0.0', '0.0', '-2.07462571701']
   Energy and dipole moment -76.2936267945 0.0 0.0 -2.07462571701
   Mode : 1 0
   ['-76.2370807561', '0.0', '0.0', '-1.97259906921']
   Energy and dipole moment -76.2370807561 0.0 0.0 -1.97259906921
   Mode : 1 6
   ['-76.3371724032', '0.0', '0.0', '-2.13779124328']
   Energy and dipole moment -76.3371724032 0.0 0.0 -2.13779124328
   Mode : 1 10
   ['-76.3095201341', '0.0', '0.0', '-2.1062885459']
   Energy and dipole moment -76.3095201341 0.0 0.0 -2.1062885459
   Mode : 1 9
   ['-76.3338392543', '0.0', '0.0', '-2.1283765272']
   Energy and dipole moment -76.3338392543 0.0 0.0 -2.1283765272
                           .
                           .
                           .
   Modes : 0 (14) 2 (11)
   ['-76.2544740734', '0.56099151493', '0.0', '-0.91306732825']
   Energy and dipole moment -76.2544740734 0.56099151493 0.0 -0.91306732825
   Modes : 0 (4) 2 (2)
   ['-76.2187440176', '-0.03564324071', '0.0', '-2.46366903321']
   Energy and dipole moment -76.2187440176 -0.03564324071 0.0 -2.46366903321
   Modes : 0 (13) 2 (13)
   ['-76.1993760924', '0.78791819876', '0.0', '-1.18678037247']
   Energy and dipole moment -76.1993760924 0.78791819876 0.0 -1.18678037247
   Modes : 0 (3) 1 (10)
   ['-76.2793569488', '0.0', '0.0', '-2.5462689329']
   Energy and dipole moment -76.2793569488 0.0 0.0 -2.5462689329
   Modes : 0 (15) 1 (4)
   ['-76.2110712557', '0.0', '0.0', '-0.91571331082']
   Energy and dipole moment -76.2110712557 0.0 0.0 -0.91571331082
   768 files found, should be 768
   Saving potentials to file


Step D - VSCF/VCI Calculation
=============================

.. literalinclude:: ../examples/4.StepByStep_H2O_NM/D_vscf_vci.py

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
    *     Vibrations is free software: you can redistribute it and/or modify
    *     it under the terms of the GNU General Public License as published by
    *     the Free Software Foundation, either version 3 of the License, or
    *     (at your option) any later version.
    *
    *     Vibrations is distributed in the hope that it will be useful,
    *     but WITHOUT ANY WARRANTY;      without even the implied warranty of
    *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    *     GNU General Public License for more details.
    *
    *     You should have received a copy of the GNU General Public License
    *     along with Vibrations.  If not, see <http://www.gnu.org/licenses/>.
    *
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
   
   +--------------------+
   |--- Solving VSCF ---|
   +--------------------+
   
   +--------------------------------+
   |--- Solving State: [0, 0, 0] ---|
   +--------------------------------+
   
   Iteration: 1 
   Mode State   Eigv
      1     0    733.0
      2     0   1687.4
      3     0   1786.0
   Sum of eigenvalues 4206.3, SCF correction -186.8, total energy 4393.0 / cm^-1
   
   Iteration: 2 
   Mode State   Eigv
      1     0    739.9
      2     0   1685.2
      3     0   1735.0
   Sum of eigenvalues 4160.0, SCF correction -189.7, total energy 4349.7 / cm^-1
   
   Iteration: 3 
   Mode State   Eigv
      1     0    739.3
      2     0   1683.7
      3     0   1733.3
   Sum of eigenvalues 4156.3, SCF correction -191.2, total energy 4347.5 / cm^-1
   
   Iteration: 4 
   Mode State   Eigv
      1     0    739.5
      2     0   1683.6
      3     0   1733.3
   Sum of eigenvalues 4156.4, SCF correction -191.3, total energy 4347.7 / cm^-1
   
   Iteration: 5 
   Mode State   Eigv
      1     0    739.5
      2     0   1683.6
      3     0   1733.3
   Sum of eigenvalues 4156.4, SCF correction -191.3, total energy 4347.7 / cm^-1
   
   Iteration: 6 
   Mode State   Eigv
      1     0    739.5
      2     0   1683.6
      3     0   1733.3
   Sum of eigenvalues 4156.4, SCF correction -191.3, total energy 4347.7 / cm^-1
   
   
   +-----------------+
   |--- VSCF Done ---|
   +-----------------+
   
   
   
   +--------------------+
   |--- VSCF Results ---|
   +--------------------+
   
   VSCF states energies in cm^-1
   ---------------------------
   [0, 0, 0] 4347.7
   
   Initial state:  [0, 0, 0]
   Transition energies in cm^-1
   WARNING: Fortran routines not used for integrals,             this might be very slow. 
            see src/Vibrations/README_f2py 
   +---------------------------+
   |--- There are 35 states ---|
   +---------------------------+
   
   combgenerator 0 of 35
   +---------------------------------------------------------+
   |--- Hamiltonian matrix constructed.Diagonalization... ---|
   +---------------------------------------------------------+
   
   +--------------------------+
   |--- Results of the VCI ---|
   +--------------------------+
   
   State        Contrib   E /cm^-1  DE /cm^-1
   [0 0 0]     0.9980  4327.3062     0.0000
   [1 0 0]     0.9959  5872.8935  1545.5873
   [0 1 0]     0.9437  7688.8941  3361.5880
   [0 0 1]     0.9563  7762.5207  3435.2145
   
   +--------------------------+
   |--- VCI IR Intensities ---|
   +--------------------------+
   
   Two sets of dipole moments given.
   
     Freq.    Int.
   [cm^-1] [km*mol^-1]
    1545.6    60.812675
    3083.1     0.610284
    3361.6     0.086193
    3435.2    13.304366
    4586.5     0.016510
    4925.9     0.016681
    4991.4     0.073617
    6097.6     0.000449
    6442.6     0.023380
    6491.2     0.000559
    6677.7     0.625113
    6765.5     0.379332
    6937.8     0.047756
    7954.1     0.000070
    7992.9     0.000037
    8219.6     0.000731
    8293.0     0.001931
    8430.7     0.003204
    9734.6     0.003547
    9924.3     0.056185
    9926.1     0.003594
   10076.6     0.010393
   10166.2     0.000604
   10298.0     0.001100
   10400.8     0.013794
   11476.8     0.000964
   11743.5     0.000001
   12537.5     0.001622
   12705.4     0.000006
   13135.1     0.001801
   13627.5     0.000022
   14303.7     0.002598
   15698.6     0.015197
   15731.8     0.004243
   +--------------------------+
   |--- Results of the VCI ---|
   +--------------------------+
   
   State        Contrib   E /cm^-1  DE /cm^-1
   [0 0 0]     0.9980  4327.3062     0.0000
   [1 0 0]     0.9959  5872.8935  1545.5873
   [2 0 0]     0.9560  7410.4067  3083.1005
   [0 1 0]     0.9437  7688.8941  3361.5880
   [0 0 1]     0.9563  7762.5207  3435.2145
   [3 0 0]     0.9126  8913.8077  4586.5015
   [1 1 0]     0.9027  9253.1573  4925.8511
   [1 0 1]     0.9590  9318.6762  4991.3700
   [2 1 0]     0.7829 10769.8879  6442.5817
   [2 0 1]     0.9561 10818.5204  6491.2142
   [0 2 0]     0.8429 11004.9624  6677.6562
   [0 1 1]     0.8595 11092.7654  6765.4592
   [0 0 2]     0.8246 11265.1435  6937.8373
   [1 2 0]     0.7285 12546.8970  8219.5908
   [1 1 1]     0.8304 12620.3500  8293.0438
   [1 0 2]     0.8261 12757.9592  8430.6530
   [0 3 0]     0.7827 14251.5580  9924.2518
   
   +--------------------------+
   |--- Results of the VCI ---|
   +--------------------------+
   
   State         Contrib        E /cm^-1       DE /cm^-1
   State   0      energy =  4327.3062,  excitation energy =     0.0000
      -0.9990   [0 0 0]  0: 
                  GS:   1.00, Fundamentals:   0.00   2:   0.00  3:   0.00  4:   0.00 
   
   State   1      energy =  5872.8935,  excitation energy =  1545.5873
      -0.9979   [1 0 0]  1: 0(1) 
                  GS:   0.00, Fundamentals:   1.00   2:   0.00  3:   0.00  4:   0.00 
   
   State   2      energy =  7410.4067,  excitation energy =  3083.1005
      -0.9778   [2 0 0]  1: 0(2) 
                  GS:   0.00, Fundamentals:   0.04   2:   0.96  3:   0.00  4:   0.00 
   
   State   3      energy =  7688.8941,  excitation energy =  3361.5880
      -0.9714   [0 1 0]  1: 1(1) 
                  GS:   0.00, Fundamentals:   0.94   2:   0.05  3:   0.00  4:   0.00 
   
   State   4      energy =  7762.5207,  excitation energy =  3435.2145
       0.9779   [0 0 1]  1: 2(1) 
                  GS:   0.00, Fundamentals:   0.96   2:   0.04  3:   0.00  4:   0.01 
   
   State   6      energy =  9253.1573,  excitation energy =  4925.8511
      -0.9501   [1 1 0]  2: 0(1) 0(1) 
                  GS:   0.00, Fundamentals:   0.00   2:   0.91  3:   0.09  4:   0.00 
   
   State   7      energy =  9318.6762,  excitation energy =  4991.3700
       0.9793   [1 0 1]  2: 0(1) 0(1) 
                  GS:   0.00, Fundamentals:   0.01   2:   0.96  3:   0.03  4:   0.00 
   
   State  11      energy = 11004.9624,  excitation energy =  6677.6562
      -0.9181   [0 2 0]  1: 1(2) 
                  GS:   0.00, Fundamentals:   0.00   2:   0.88  3:   0.11  4:   0.01 
   
   State  12      energy = 11092.7654,  excitation energy =  6765.4592
       0.9271   [0 1 1]  2: 1(1) 1(1) 
                  GS:   0.00, Fundamentals:   0.03   2:   0.86  3:   0.10  4:   0.01 
   
   State  13      energy = 11265.1435,  excitation energy =  6937.8373
       0.9081   [0 0 2]  1: 2(2) 
                  GS:   0.00, Fundamentals:   0.01   2:   0.89  3:   0.09  4:   0.00 
