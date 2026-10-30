.. _Overview:

########
Overview
########

Vibrations is a Python code for vibrational calculations with
user-defined vibrational modes. For an overview of the capabilities
of the code, see the following articles

`ChemPhysChem 15 (2014) 3365 <http://dx.doi.org/10.1002/cphc.201402251>`__,

`J. Chem. Phys. 144 (2016) 164111 <http://dx.doi.org/10.1063/1.4947213>`__,

`J. Phys. Chem. Lett. 7 (2016) 3084 <http://dx.doi.org/10.1021/acs.jpclett.6b01451>`__.


See the ``examples/`` directory for some examples of typical runs.


.. code-block:: console

   Vibrations/examples/
            ├── 1.Existing_potentials/ 
            ├── 2.Localized_modes_harmonic/
            ├── 3.Localized_modes_anharmonic/
            ├── 4.StepByStep_H2O_NM/
            └── 5.StepByStep_H2O_LM/

To demonstrate the workflow for calculating vibrational spectra using localized modes,
the results of **SNF** (https://reiher.ethz.ch/software/snf.html) calculations 
for ethylene and water (harmonic approximation) are used. 


**Application Examples**

#. :doc:`Existing Potentials <example1>`
#. :doc:`Localized Harmonic Modes <example2>`
#. :doc:`Localized Anharmonic Modes <example3>`
#. :doc:`Step by Step Normal Modes <example4>`
#. :doc:`Step by Step Localized Modes <example5>`
