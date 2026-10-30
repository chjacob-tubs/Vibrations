**************************
Localized Anharmonic Modes
**************************

This example demonstrates the use of localized modes in an **L-VCI-S** (localized-mode 
vibrational configuration interaction with single excitations) calculation for water.
The :ref:`first script <Harmonic Calculation>` uses harmonic potentials. The :ref:`second script <Anharmonic Calculation>` uses anharmonic potentials.

.. code-block:: console

   Vibrations/examples/
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

.. _Harmonic Calculation:

Python Scripts: Harmonic Calculations
=====================================

The script ``Example3_1_Harmonic_L-VCI-S.py`` performs an L-VCI-S calculation using harmonic
potentials:


.. literalinclude:: ../examples/3.Localized_modes_anharmonic/Example3_1_Harmonic_L-VCI-S.py
    :language: python

Running the script, generates the following output:

.. code-block:: console

   
    ***************************************************************************
    *
    * Current Vibrations Header and Copyright *
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

.. _Anharmonic Calculation:   

Python Scripts: Anharmonic Calculations
=======================================

The script ``Example3_2_Anharmonic_L-VCI-S.py`` performs an L-VCI-S calculation using 
anharmonic potentials:


.. literalinclude:: ../examples/3.Localized_modes_anharmonic/Example3_2_Anharmonic_L-VCI-S.py
    :language: python

Running the script, generates the following output:

.. code-block:: console

   
    ***************************************************************************
    *
    * Current Vibrations Header and Copyright *
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
