PySNF
=====

This module enables a comfortable connection between SNF and LocVib/src/VibTools.

SNF 4.0.0, Manual (2007):

https://ethz.ch/content/dam/ethz/special-interest/chab/physical-chemistry/reiher-dam/documents/Software/SNF_4.0.0_manual.pdf

SNF-Files: Output, Restart, Control
-----------------------------------

SNFRestartFile
^^^^^^^^^^^^^^

 .. _SNFRestartFile:

.. autoclass:: VibTools.SNFRestartFile
   :members:

Example
"""""""

LocVib(VibTools) should already be installed as a loadable Python module.

Import package:

   >>> import VibTools

Read the SNF restart file from the test folder in LocVib:

   >>> filename = 'LocVib/src/VibTools/tests/test_data/H2O/restart'
   >>> SNFrest = VibTools.SNFRestartFile()
   >>> SNFrest.read(filename)

Show restart file:

   >>> vi LocVib/src/VibTools/tests/test_data/H2O/restart
   /scratch/pawel/20130506-130557                                
   (4D20.10)    
   420    1    2    0    9    2    0
   0.1000000000D-01
   /scratch/pawel/20130506-130557
   /scratch/pawel/20130506-130557
     1  1  1  1  1  1  1
   cfn218c1cfn218c1cfn218c1cfn218c1cfn218c1cfn218c1
     2  1  1  1  1  1  1
   cfn218c1cfn218c1cfn218c1cfn218c1cfn218c1cfn218c1
     3  1  1  1  1  1  1
   cfn218c1cfn218c1cfn218c1cfn218c1cfn218c1cfn218c1
    dipoles atom            1
       0.2268563907D-02   -0.6000000000D-13   -0.8362887813D+00   -0.2300478509D-02
      -0.2112800000D-08   -0.8377209034D+00   -0.4937777690D-05    0.3705567367D-02
      -0.8370071800D+00   -0.4937777740D-05   -0.3705567367D-02   -0.8370071800D+00
       0.1087256160D-02   -0.2112810000D-08   -0.8354534977D+00   -0.1124379904D-02
      -0.4000000000D-13   -0.8385388605D+00
   etc. ...   

Print out some data (see the documentation above for the meaning of the variables):

   >>> print("SNFrest.nvar =", SNFrest.nvar)
   SNFrest.nvar = 9

   >>> print("SNFrest.natoms =", SNFrest.natoms)
   SNFrest.natoms = 3

   >>> print("SNFrest.iaus =", SNFrest.iaus)
   SNFrest.iaus = 2

   >>> print("SNFrest.nmodes =", SNFrest.nmodes)
   SNFrest.nmodes = 0

   >>> print("SNFrest.displ =", SNFrest.displ)
   SNFrest.displ = 0.01

   >>> print("SNFrest.intonly =", SNFrest.intonly)
   SNFrest.intonly = False

   >>> print("SNFrest.ncalcsets =", SNFrest.ncalcsets)
   SNFrest.ncalcsets = 3

Print dipole moments with the array size and shape:
   
   >>> import numpy as np
   >>> dipole = SNFrest.dipole 
   >>> print("size(SNFrest.dipole) =", np.size(dipole))
   size(SNFrest.dipole) = 54
   >>> print("shape(SNFrest.dipole) =", np.shape(dipole))
   shape(SNFrest.dipole) = (3, 6, 3)
   >>> print("SNFrest.dipole =", dipole)
   SNFrest.dipole = [[[ 2.26856391e-03 -6.00000000e-14 -8.36288781e-01]
     [-2.30047851e-03 -2.11280000e-09 -8.37720903e-01]
     [-4.93777769e-06  3.70556737e-03 -8.37007180e-01]
     [-4.93777774e-06 -3.70556737e-03 -8.37007180e-01]
     [ 1.08725616e-03 -2.11281000e-09 -8.35453498e-01]
     [-1.12437990e-03 -4.00000000e-14 -8.38538861e-01]]
    [[-4.54573646e-03 -2.10000000e-13 -8.37013826e-01]
     [ 4.54573641e-03 -1.50000000e-13 -8.37013826e-01]
     [-2.28900000e-11 -7.41105902e-03 -8.36994205e-01]
     [-2.28200000e-11  7.41105902e-03 -8.36994205e-01]
     [-2.26200000e-11 -1.20000000e-13 -8.40054999e-01]
     [-2.28700000e-11 -2.00000000e-14 -8.33917908e-01]]
    [[ 2.30046648e-03 -2.11276000e-09 -8.37720903e-01]
     [-2.26856395e-03 -5.00000000e-14 -8.36288781e-01]
     [ 4.93773211e-06  3.70556737e-03 -8.37007180e-01]
     [ 4.93773206e-06 -3.70556737e-03 -8.37007180e-01]
     [-1.08726819e-03 -2.11277000e-09 -8.35453498e-01]
     [ 1.12437986e-03 -8.00000000e-14 -8.38538861e-01]]]

This is analogous for the other variables/attributes.

Delete the data entries of atom 0 and 2:

    >>> atoms = [0,2]
    >>> SNFrest.del_int_for_atoms(atoms)

Show dipole moments with deleted data entries (example - dipoles):

    >>> print("SNFrest.dipole = ", SNFrest.dipole)
    SNFrest.dipole ) = [[[ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
      [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
      [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
      [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
      [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
      [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]]
     [[-4.54573646e-03 -2.10000000e-13 -8.37013826e-01]
      [ 4.54573641e-03 -1.50000000e-13 -8.37013826e-01]
      [-2.28900000e-11 -7.41105902e-03 -8.36994205e-01]
      [-2.28200000e-11  7.41105902e-03 -8.36994205e-01]
      [-2.26200000e-11 -1.20000000e-13 -8.40054999e-01]
      [-2.28700000e-11 -2.00000000e-14 -8.33917908e-01]]
     [[ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
      [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
      [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
      [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
      [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]
      [ 0.00000000e+00  0.00000000e+00  0.00000000e+00]]]


SNFOutputFile
^^^^^^^^^^^^^

 .. _SNFOutputFile:


.. autoclass:: VibTools.SNFOutputFile
   :members:

Example
"""""""

LocVib(VibTools) should already be installed as a loadable Python module.

Import package:

   >>> import VibTools

First, we need the ``VibToolsMolecule`` class and a water molecule from the test data folder:

   >>> path = 'LocVib/src/VibTools/tests/test_data/H2O/'

   >>> molefilename = path+'coord'
   >>> vtmole = VibTools.VibToolsMolecule()
   >>> vtmole.read_from_coord(molefilename)

Create the ``SNFOutputFile`` object:

   >>> outfilename = path+'snf.out'
   >>> SNFoutput = VibTools.SNFOutputFile(outfilename)

Read the ``snf.out`` file:

   >>> SNFoutput.read(vtmole)

Print some data:
 
Print modes class:
   
   >>> print("modes =", SNFoutput.modes)
   modes = <VibTools.Modes.VibModes object at 0x1545d4222b50>

Print lwl:

   >>> print("lwl =", SNFoutput.lwl)
   None

Print filename:

   >>> print("filename =", SNFoutput.filename)
   filename = test_data/H2O/snf.out

Print intonly:

   >>> print("intonly =", SNFoutput.intonly)
   intonly = False

Print mass-weighted normal modes:

   >>> print("modes_mw =", SNFoutput.modes.modes_mw)
   modes_mw = [[ 0.41227257  0.          0.54138338 -0.         -0.         -0.2717917
     -0.41227257 -0.          0.54138338]
    [-0.57448154  0.          0.38852104 -0.         -0.         -0.19505052
      0.57448154 -0.          0.38852104]
    [-0.536967   -0.          0.41872766  0.26956849  0.         -0.
     -0.536967    0.         -0.41872766]]

Print frequencies corresponding to the normal modes above:

   >>> print("freqs =", SNFoutput.modes.freqs)
   freqs = [1575.16798 3685.32274 3796.19483]

SNFControlFile
^^^^^^^^^^^^^^

 .. _SNFControlFile:

This part of the documentation is still in progress.

.. autoclass:: VibTools.SNFControlFile

Example
"""""""

This part of the documentation is still in progress.

SNFResults
----------

``SNFResults`` is based on the classes `SNFRestartFile`_, `SNFOutputFile`_, `SNFControlFile`_ and
LocVib's main modules: 

.. toctree::
   :maxdepth: 1

   Molecule
   Modes

.. autoclass:: VibTools.SNFResults
   :members:

Example
-------

LocVib(VibTools) should already be installed as a loadable Python module.

Import package:

   >>> import VibTools

First, we need molecule data (water) from the test data folder:

   >>> path = 'LocVib/src/VibTools/tests/test_data/H2O/'

The corresponding files (``snf.out``, ``restart``, ``coord``) are located in this folder.
Initialize ``SNFResults`` and read the data:

   >>> res = VibTools.SNFResults(
   ...     outname=path+'snf.out',
   ...     restartname=path+'restart',
   ...     coordfile=path+'coord')
   >>> res.read()

What data are in the molecular module? 

.. toctree::
   :maxdepth: 1

   Molecule

Here, we see some general molecular informations such as the number of atoms (``natoms``), 
atomic masses (``atmasses``), atomic numbers (``atnums``) and coordinates (``coordinates``).

   >>> mol = res.mol
   >>> print('natoms =', mol.natoms)
   >>> print('atmasses =', mol.atmasses )
   >>> print('atnums =', mol.atnums)
   >>> print('coordinates =', mol.coordinates)
   natoms = 3
   atmasses = [ 1.00782503 15.99491462  1.00782503]
   atnums = [1. 8. 1.]
   coordinates = [[-2.40315717e+00 -1.11797830e+00 -2.19608983e-03]
    [-3.16918773e+00 -1.11797830e+00  5.95151641e-01]
    [-3.93521804e+00 -1.11797830e+00 -2.19608983e-03]]

What data are in the `SNFOutputFile`_ class?

.. toctree::
   :maxdepth: 1

   Modes

Here, we get some technical informations, the mass-weighted normal modes (``modes_mw``) 
and the corresponding frequencies (``freqs``)::

   >>> output = res.snfoutput
   >>> print("lwl =", output.lwl)
   >>> print("filename =", output.filename)
   >>> print("intonly =", output.intonly)
   >>> print("\nmodes_mw =\n", output.modes.modes_mw)
   >>> print("\nfreqs =\n", output.modes.freqs)
   lwl = None
   filename = test_data/H2O/snf.out
   intonly = False   

   modes_mw =
    [[ 0.41227257  0.          0.54138338 -0.         -0.         -0.2717917
     -0.41227257 -0.          0.54138338]
    [-0.57448154  0.          0.38852104 -0.         -0.         -0.19505052
      0.57448154 -0.          0.38852104]
    [-0.536967   -0.          0.41872766  0.26956849  0.         -0.
     -0.536967    0.         -0.41872766]]

   freqs =
    [1575.16798 3685.32274 3796.19483]
