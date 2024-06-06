****************************
Step by Step Local Modes H2O
****************************

Here we see the script for localized harmonic modes:

.. code-block:: console

   /vibrations/examples/
            ├── 5.StepByStep_H2O_NM_LM/ 
                     ├── A_h2o_make_localmode_grid.py
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

.. include:: ../examples/5.StepByStep_H2O_LM/README
   :literal:


Step A - Generate Grid
=======================

.. literalinclude:: ../examples/5.StepByStep_H2O_LM/A_h2o_make_localmode_grid.py

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
   
   
   
   *** Localization: 
   Modes localized: 3, modes in total: 3
   Normal mode localization: Cycle   1    p:    1.938   change:  0.6041009     1.95017 
   Normal mode localization: Cycle   2    p:    2.221   change:  0.2827215     0.96345 
   Normal mode localization: Cycle   3    p:    2.221   change:  0.0003344     0.02133 
   Normal mode localization: Cycle   4    p:    2.221   change:  0.0000000     0.00002 
   Obtaining coupling matrix in [a.u.]
   Obtaining coupling matrix in [a.u.]
   Obtaining coupling matrix in [a.u.]
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
    *** Writing V2 - Mode: 0 1 , Point:  0 2
    *** Writing V2 - Mode: 0 1 , Point:  0 3
    *** Writing V2 - Mode: 0 1 , Point:  0 4
    *** Writing V2 - Mode: 0 1 , Point:  0 5
    *** Writing V2 - Mode: 0 1 , Point:  0 6
    *** Writing V2 - Mode: 0 1 , Point:  0 7
    *** Writing V2 - Mode: 0 1 , Point:  0 8
    *** Writing V2 - Mode: 0 1 , Point:  0 9
    *** Writing V2 - Mode: 0 1 , Point:  0 10
    *** Writing V2 - Mode: 0 1 , Point:  0 11
    *** Writing V2 - Mode: 0 1 , Point:  0 12
    *** Writing V2 - Mode: 0 1 , Point:  0 13
    *** Writing V2 - Mode: 0 1 , Point:  0 14
    *** Writing V2 - Mode: 0 1 , Point:  0 15
    *** Writing V2 - Mode: 0 1 , Point:  1 0
    *** Writing V2 - Mode: 0 1 , Point:  1 1
    *** Writing V2 - Mode: 0 1 , Point:  1 2
    *** Writing V2 - Mode: 0 1 , Point:  1 3
    *** Writing V2 - Mode: 0 1 , Point:  1 4
    *** Writing V2 - Mode: 0 1 , Point:  1 5
    *** Writing V2 - Mode: 0 1 , Point:  1 6
    *** Writing V2 - Mode: 0 1 , Point:  1 7
    *** Writing V2 - Mode: 0 1 , Point:  1 8
    *** Writing V2 - Mode: 0 1 , Point:  1 9
    *** Writing V2 - Mode: 0 1 , Point:  1 10
    *** Writing V2 - Mode: 0 1 , Point:  1 11
    *** Writing V2 - Mode: 0 1 , Point:  1 12
    *** Writing V2 - Mode: 0 1 , Point:  1 13
    *** Writing V2 - Mode: 0 1 , Point:  1 14
    *** Writing V2 - Mode: 0 1 , Point:  1 15
    *** Writing V2 - Mode: 0 1 , Point:  2 0
    *** Writing V2 - Mode: 0 1 , Point:  2 1
    *** Writing V2 - Mode: 0 1 , Point:  2 2
    *** Writing V2 - Mode: 0 1 , Point:  2 3
    *** Writing V2 - Mode: 0 1 , Point:  2 4
    *** Writing V2 - Mode: 0 1 , Point:  2 5
                        .
                        .
                        .
    *** Writing V2 - Mode: 1 2 , Point:  15 4
    *** Writing V2 - Mode: 1 2 , Point:  15 5
    *** Writing V2 - Mode: 1 2 , Point:  15 6
    *** Writing V2 - Mode: 1 2 , Point:  15 7
    *** Writing V2 - Mode: 1 2 , Point:  15 8
    *** Writing V2 - Mode: 1 2 , Point:  15 9
    *** Writing V2 - Mode: 1 2 , Point:  15 10
    *** Writing V2 - Mode: 1 2 , Point:  15 11
    *** Writing V2 - Mode: 1 2 , Point:  15 12
    *** Writing V2 - Mode: 1 2 , Point:  15 13
    *** Writing V2 - Mode: 1 2 , Point:  15 14
    *** Writing V2 - Mode: 1 2 , Point:  15 15

Step B - Single Points Calculations
===================================

.. include:: ../examples/5.StepByStep_H2O_LM/B_calculate_singlepoints.README
   :literal:


Step C - Generate 1-Mode and 2-Mode Potentials
==============================================

.. literalinclude:: ../examples/5.StepByStep_H2O_LM/C_make_potentials.py

When we run the script we get:

.. code-block:: console
   
                        .
                        .
                        .
   Modes : 0 (12) 2 (10) 
   ['-76.1551654084', '-0.38930257674', '0.0', '-1.61703594745']
   Energy and dipole moment -76.1551654084 -0.38930257674 0.0 -1.61703594745
   Modes : 0 (0) 2 (1) 
   ['-76.2122043407', '-0.13580931287', '-0.0', '-2.42959914813']
   Energy and dipole moment -76.2122043407 -0.13580931287 -0.0 -2.42959914813
   Modes : 0 (9) 1 (2) 
   ['-75.9482244506', '0.24743114242', '-0.0', '-2.01797220548']
   Energy and dipole moment -75.9482244506 0.24743114242 -0.0 -2.01797220548
   Modes : 0 (12) 1 (0) 
   ['-74.9488365237', '0.24748960313', '0.0', '-1.80162691016']
   Energy and dipole moment -74.9488365237 0.24748960313 0.0 -1.80162691016
   Modes : 0 (5) 1 (9) 
   ['-76.3204704637', '0.009008032879999999', '-0.0', '-2.12205006167']
   Energy and dipole moment -76.3204704637 0.009008032879999999 -0.0 -2.12205006167
   Modes : 1 (14) 2 (2) 
   ['-76.2359800926', '0.12288949596000001', '0.0', '-2.4364034664200003']
   Energy and dipole moment -76.2359800926 0.12288949596000001 0.0 -2.4364034664200003
   Modes : 0 (14) 2 (11) 
   ['-75.8348219861', '-0.70835825599', '-0.0', '-1.2541703204799999']
   Energy and dipole moment -75.8348219861 -0.70835825599 -0.0 -1.2541703204799999
   Modes : 0 (4) 2 (2) 
   ['-76.2639373507', '-0.05608415505000001', '0.0', '-2.50413909515']
   Energy and dipole moment -76.2639373507 -0.05608415505000001 0.0 -2.50413909515
   Modes : 0 (13) 2 (13) 
   ['-76.1588679178', '-0.71813136164', '-0.0', '-0.72513139622']
   Energy and dipole moment -76.1588679178 -0.71813136164 -0.0 -0.72513139622
   Modes : 0 (3) 1 (10) 
   ['-76.285003255', '-0.01007049274', '0.0', '-2.06752146986']
   Energy and dipole moment -76.285003255 -0.01007049274 0.0 -2.06752146986
   Modes : 0 (15) 1 (4) 
   ['-75.0436718024', '-0.31572850232', '0.0', '-1.8516997791600003']
   Energy and dipole moment -75.0436718024 -0.31572850232 0.0 -1.8516997791600003
   768 files found, should be 768
   Saving potentials to file

Step D - VSCF/VCI Calculation
=============================

.. literalinclude:: ../examples/5.StepByStep_H2O_LM/D_vscf_vci.py

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
   
   
   
   *** Localization: 
   Modes localized: 3, modes in total: 3
   Normal mode localization: Cycle   1    p:    1.938   change:  0.6041009     1.95017 
   Normal mode localization: Cycle   2    p:    2.221   change:  0.2827215     0.96345 
   Normal mode localization: Cycle   3    p:    2.221   change:  0.0003344     0.02133 
   Normal mode localization: Cycle   4    p:    2.221   change:  0.0000000     0.00002 
   Obtaining coupling matrix in [a.u.]
   Obtaining coupling matrix in [a.u.]
   Obtaining coupling matrix in [a.u.]
   
   +--------------------+
   |--- Solving VSCF ---|
   +--------------------+
   
   +--------------------------------+
   |--- Solving State: [0, 0, 0] ---|
   +--------------------------------+
   
   Iteration: 1 
   Mode State   Eigv
      1     0   1728.5
      2     0   1728.5
      3     0    742.2
   Sum of eigenvalues 4199.2, SCF correction -138.5, total energy 4337.7 / cm^-1
   
   Iteration: 2 
   Mode State   Eigv
      1     0   1726.1
      2     0   1726.1
      3     0    740.8
   Sum of eigenvalues 4193.1, SCF correction -138.6, total energy 4331.6 / cm^-1
   
   Iteration: 3 
   Mode State   Eigv
      1     0   1726.1
      2     0   1726.1
      3     0    740.8
   Sum of eigenvalues 4193.0, SCF correction -138.6, total energy 4331.6 / cm^-1
   
   Iteration: 4 
   Mode State   Eigv
      1     0   1726.1
      2     0   1726.1
      3     0    740.8
   Sum of eigenvalues 4193.0, SCF correction -138.6, total energy 4331.6 / cm^-1
   
   
   +-----------------+
   |--- VSCF Done ---|
   +-----------------+
   
   
   
   +--------------------+
   |--- VSCF Results ---|
   +--------------------+
   
   VSCF states energies in cm^-1
   ---------------------------
   [0, 0, 0] 4331.6
   
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
   [0 0 0]     0.9997  4329.3051     0.0000
   [0 0 1]     0.9972  5877.3636  1548.0585
   [1 0 0]     0.4754  7687.1313  3357.8261
   [0 1 0]     0.4991  7786.4538  3457.1486
   
   +--------------------------+
   |--- VCI IR Intensities ---|
   +--------------------------+
   
   Two sets of dipole moments given.
   
     Freq.    Int.
   [cm^-1] [km*mol^-1]
    1548.1    59.908986
    3063.0     0.696494
    3357.8     0.077599
    3457.1    12.128407
    4568.0     0.023175
    4897.7     0.035736
    4974.6     0.030999
    6081.5     0.000052
    6411.5     0.009873
    6467.9     0.248613
    6612.3     0.283374
    6664.0     3.260552
    6851.9     0.069789
    7871.0     0.000041
    7955.2     0.001247
    8141.4     0.000116
    8180.3     0.023148
    8349.4     0.000003
    9590.8     0.001602
    9631.3     0.072258
    9738.6     0.035824
    9760.2     0.267920
    9829.5     0.000032
    9986.4     0.005925
   10164.3     0.010803
   11221.8     0.003886
   11237.3     0.006008
   11472.2     0.000378
   11656.6     0.000129
   12699.0     0.004948
   12703.4     0.031941
   13052.2     0.000402
   13186.3     0.001157
   13429.2     0.000012
   +--------------------------+
   |--- Results of the VCI ---|
   +--------------------------+
   
   State        Contrib   E /cm^-1  DE /cm^-1
   [0 0 0]     0.9997  4329.3051     0.0000
   [0 0 1]     0.9972  5877.3636  1548.0585
   [0 0 2]     0.9445  7392.3247  3063.0196
   [1 0 0]     0.4754  7687.1313  3357.8261
   [0 1 0]     0.4991  7786.4538  3457.1486
   [0 0 3]     0.8791  8897.3370  4568.0319
   [1 0 1]     0.4411  9226.9796  4897.6744
   [0 1 1]     0.4970  9303.9395  4974.6343
   [1 0 2]     0.3524 10740.8069  6411.5018
   [0 1 2]     0.4571 10797.2076  6467.9024
   [2 0 0]     0.3495 10941.5645  6612.2594
   [0 2 0]     0.4572 10993.3253  6664.0202
   [1 1 0]     0.7742 11181.2357  6851.9306
   [2 0 1]     0.3364 12470.6903  8141.3851
   [0 2 1]     0.4134 12509.5762  8180.2711
   [1 1 1]     0.7843 12678.6769  8349.3718
   [3 0 0]     0.3497 14067.9448  9738.6397
   [0 3 0]     0.3601 14089.5022  9760.1971
   [2 1 0]     0.3658 14315.7300  9986.4249
   
   +--------------------------+
   |--- Results of the VCI ---|
   +--------------------------+
   
   State         Contrib        E /cm^-1       DE /cm^-1
   State   0      energy =  4329.3051,  excitation energy =     0.0000
       0.9998   [0 0 0]  0: 
                  GS:   1.00, Fundamentals:   0.00   2:   0.00  3:   0.00  4:   0.00 
   
   State   1      energy =  5877.3636,  excitation energy =  1548.0585
       0.9986   [0 0 1]  1: 2(1) 
                  GS:   0.00, Fundamentals:   1.00   2:   0.00  3:   0.00  4:   0.00 
   
   State   2      energy =  7392.3247,  excitation energy =  3063.0196
      -0.9719   [0 0 2]  1: 2(2) 
                  GS:   0.00, Fundamentals:   0.05   2:   0.95  3:   0.00  4:   0.00 
   
   State   3      energy =  7687.1313,  excitation energy =  3357.8261
       0.6895   [0 1 0]  1: 1(1) 
       0.6895   [1 0 0]  1: 0(1) 
                  GS:   0.00, Fundamentals:   0.95   2:   0.05  3:   0.00  4:   0.00 
   
   State   4      energy =  7786.4538,  excitation energy =  3457.1486
       0.7064   [0 1 0]  1: 1(1) 
      -0.7064   [1 0 0]  1: 0(1) 
                  GS:   0.00, Fundamentals:   1.00   2:   0.00  3:   0.00  4:   0.00 
   
   State   6      energy =  9226.9796,  excitation energy =  4897.6744
      -0.6641   [0 1 1]  2: 1(1) 1(1) 
      -0.6642   [1 0 1]  2: 0(1) 0(1) 
       0.3204   [0 0 3]  1: 2(3) 
                  GS:   0.00, Fundamentals:   0.00   2:   0.89  3:   0.11  4:   0.00 
   
   State   7      energy =  9303.9395,  excitation energy =  4974.6343
      -0.7050   [0 1 1]  2: 1(1) 1(1) 
       0.7049   [1 0 1]  2: 0(1) 0(1) 
                  GS:   0.00, Fundamentals:   0.00   2:   0.99  3:   0.00  4:   0.00 
   
   State  11      energy = 10941.5645,  excitation energy =  6612.2594
      -0.5910   [0 2 0]  1: 1(2) 
      -0.4124   [1 1 0]  2: 0(1) 0(1) 
      -0.5912   [2 0 0]  1: 0(2) 
                  GS:   0.00, Fundamentals:   0.00   2:   0.87  3:   0.12  4:   0.01 
   
   State  12      energy = 10993.3253,  excitation energy =  6664.0202
       0.6761   [0 2 0]  1: 1(2) 
      -0.6760   [2 0 0]  1: 0(2) 
                  GS:   0.00, Fundamentals:   0.00   2:   0.92  3:   0.08  4:   0.00 
   
   State  13      energy = 11181.2357,  excitation energy =  6851.9306
      -0.3270   [0 2 0]  1: 1(2) 
       0.8799   [1 1 0]  2: 0(1) 0(1) 
      -0.3270   [2 0 0]  1: 0(2) 
                  GS:   0.00, Fundamentals:   0.00   2:   0.99  3:   0.01  4:   0.00 
