**************************
Localized Anharmonic Modes
**************************

.. code-block:: console

   /vibrations/examples/
            ├── 3.Localized_modes_anharmonic/ 
                     ├── coord
                     ├── Dm1_g16.npy
                     ├── Dm2_g16.npy
                     ├── Example3_1_Harmonic_L-VCI-S.py
                     ├── Example3_2_Anharmonic_L-VCI-S.py
                     ├── Potentials/
                     ├── README
                     ├── restart
                     ├── snf.out
                     ├── V1_g16.npy
                     └── V2_g16.npy

Instructions with PyADF
=======================

.. seealso::
   https://github.com/chjacob-tubs/pyadf-releases

.. include:: ../examples/3.Localized_modes_anharmonic/README
   :literal:

Python Scripts: Harmonic Calculations
=====================================

Here we see the script for the first part of the example *Example3_1_Harmonic_L-VCI-S.py*:


.. literalinclude:: ../examples/3.Localized_modes_anharmonic/Example3_1_Harmonic_L-VCI-S.py
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
   +------------------+
   |--- Example 3: ---|
   +------------------+
   
   Localization of modes of water, harmonic L-VCI-S
   provides initial normal modes' frequencies and intensities
   
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
      1     0   1793.6
      2     0   1793.6
      3     0    802.0
   Sum of eigenvalues 4389.3, SCF correction 0.0, total energy 4389.3 / cm^-1
   
   Iteration: 2 
   Mode State   Eigv
      1     0   1793.6
      2     0   1793.6
      3     0    802.0
   Sum of eigenvalues 4389.3, SCF correction 0.0, total energy 4389.3 / cm^-1
   
   
   +-----------------+
   |--- VSCF Done ---|
   +-----------------+
   
   
   
   +--------------------+
   |--- VSCF Results ---|
   +--------------------+
   
   VSCF states energies in cm^-1
   ---------------------------
   [0, 0, 0] 4389.3
   
   Initial state:  [0, 0, 0]
   Transition energies in cm^-1
   WARNING: Fortran routines not used for integrals,             this might be very slow. 
            see src/Vibrations/README_f2py 
   +---------------------------+
   |--- There are 4 states ---|
   +---------------------------+
   
   combgenerator 0 of 4
   +---------------------------------------------------------+
   |--- Hamiltonian matrix constructed.Diagonalization... ---|
   +---------------------------------------------------------+
   
   +--------------------------+
   |--- Results of the VCI ---|
   +--------------------------+
   
   State        Contrib   E /cm^-1  DE /cm^-1
   [0 0 0]     1.0000  4389.1861     0.0000
   [0 0 1]     0.9999  5992.8745  1603.6884
   [0 1 0]     0.4999  7918.9581  3529.7719
   [1 0 0]     0.5000  8033.6774  3644.4913
   
   +--------------------------+
   |--- VCI IR Intensities ---|
   +--------------------------+
   
   Only one set of dipole moments given.
   
     Freq.    Int.
   [cm^-1] [km*mol^-1]
    1603.7    62.142816
    3529.8     0.092152
    3644.5    16.961158
   
   
   +---------------+
   |--- Results ---|
   +---------------+
   
             Normal           Localized          L-VCI-S
   No      Freq.    Int      Freq.    Int      Freq.    Int
   --------------------------------------------------------
    1     1603.7   62.2     3586.7    8.4     1603.7   62.1
    2     3529.4    0.1     3586.7    8.4     3529.8    0.1
    3     3644.2   16.7     1603.9   62.1     3644.5   17.0
   
   
   +------------------------------------+
   |--- http://www.christophjacob.eu ---|
   +------------------------------------+

Python Scripts: Anharmonic Calculations
=======================================

Here we see the script for the first part of the example *Example3_2_Anharmonic_L-VCI-S.py*:


.. literalinclude:: ../examples/3.Localized_modes_anharmonic/Example3_2_Anharmonic_L-VCI-S.py
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
   +------------------+
   |--- Example 3: ---|
   +------------------+
   
   Localization of modes of water, harmonic L-VCI-S
   provides initial normal modes' frequencies and intensities
   
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
   Sum of eigenvalues 4199.2, SCF correction -138.6, total energy 4337.8 / cm^-1
   
   Iteration: 2 
   Mode State   Eigv
      1     0   1726.1
      2     0   1726.1
      3     0    740.8
   Sum of eigenvalues 4193.0, SCF correction -138.6, total energy 4331.6 / cm^-1
   
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
   |--- There are 4 states ---|
   +---------------------------+
   
   combgenerator 0 of 4
   +---------------------------------------------------------+
   |--- Hamiltonian matrix constructed.Diagonalization... ---|
   +---------------------------------------------------------+
   
   +--------------------------+
   |--- Results of the VCI ---|
   +--------------------------+
   
   State        Contrib   E /cm^-1  DE /cm^-1
   [0 0 0]     1.0000  4331.4452     0.0000
   [0 0 1]     0.9992  5886.2979  1554.8528
   [0 1 0]     0.4997  7679.7971  3348.3520
   [1 0 0]     0.5001  7791.2274  3459.7822
   
   +--------------------------+
   |--- VCI IR Intensities ---|
   +--------------------------+
   
   Two sets of dipole moments given.
   
     Freq.    Int.
   [cm^-1] [km*mol^-1]
    1554.9    59.363190
    3348.4     0.037805
    3459.8    12.075082
   
   
   +---------------+
   |--- Results ---|
   +---------------+
   
             Normal           Localized          L-VCI-S
   No      Freq.    Int      Freq.    Int      Freq.    Int
   --------------------------------------------------------
    1     1603.7   62.2     3586.7    8.4     1554.9   59.4
    2     3529.4    0.1     3586.7    8.4     3348.4    0.0
    3     3644.2   16.7     1603.9   62.1     3459.8   12.1
   
   
   +------------------------------------+
   |--- http://www.christophjacob.eu ---|
   +------------------------------------+
