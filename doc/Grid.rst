.. _Grid:

Grid
----

Module containing the Grid-related class.

Grid
^^^^

.. autoclass:: Vibrations.Grid
   :members:
   :show-inheritance:

Example
"""""""

Import the Vibrations package:

.. code-block:: python

    import Vibrations as vib

When importing the package, the following information is displayed:

.. code-block:: console
     
     ***************************************************************************
     *
    * Current Vibrations Header and Copyright *
     *
     ***************************************************************************
     *
     *



A ``Grid`` object can be created. Its attributes provide information about the vibrational modes,
the molecule and the generated grid:

.. code-block:: python

    # Create an empty grid
    grid = vib.Grid()
    
    modes  = grid.modes
    mol    = grid.mol
    ngrid  = grid.ngrid
    amp    = grid.amp
    grids  = grid.grids
    mol    = grid.mol
    natoms = grid.natoms
    nmodes = grid.nmodes
    
    print("modes  =", modes )
    print("mol    =", mol   )
    print("ngrid  =", ngrid )
    print("amp    =", amp   )
    print("grids  =", grids )
    print("mol    =", mol   )
    print("natoms =", natoms)
    print("nmodes =", nmodes)

Terminal output:

.. code-block:: console

    modes  = None
    mol    = None
    ngrid  = 0
    amp    = 0
    grids  = []
    mol    = None
    natoms = 0
    nmodes = 0

As seen, the attributes are initialized with default values: 
empty lists for arrays, `None` for objects and `0` for integers.

The ``src/Vibrations/unittests/test_data`` directory provides data from an 
SNF calculation (SNF: https://reiher.ethz.ch/software/snf.html). To prepare the data, 
the LocVib package is required.

.. note::
   The ``SNFResults`` class from VibTools is used to read the SNF output and extract molecular structure, normal modes and frequencies.

.. code-block:: python

   # Read in normal modes from SNF results
   # using VibTools (LocVib package)
   import VibTools
   path = 'src/Vibrations/unittests/test_data/'
   res = VibTools.SNFResults(outname = path+'snf.out',
                             restartname=path+'restart',
                             coordfile=path+'coord')
   res.read()

   # Now localize modes in separate subsets
   subsets = [list(range(0,4)), list(range(4,8)), list(range(8,12))]

The local modes are prepared using VibTools:

.. code-block:: python

    localmodes,cmat = VibTools.LocVib.localize_subsets(subsets,res.modes)
    print('localmodes =',localmodes)
    print('cmat =',cmat)

.. code-block:: console

   Modes localized: 12, modes in total: 12
   Normal mode localization: Cycle   1    p:    1.423   change:  0.5793322     1.71249 
   Normal mode localization: Cycle   2    p:    1.430   change:  0.0067229     0.19422 
   Normal mode localization: Cycle   3    p:    1.438   change:  0.0084692     0.19886 
   Normal mode localization: Cycle   4    p:    1.442   change:  0.0033325     0.12330
   Normal mode localization: Cycle   5    p:    1.443   change:  0.0012660     0.06618 
                            .
                            .
                            . 
   localmodes = <VibTools.Modes.VibModes object at 0x7f01c74358d0>
   cmat = [[ 2.04579166e-05  1.31985469e-06  1.31897777e-06  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
    [ 1.31985469e-06  1.93579889e-05  9.75374116e-07  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]
                            .
                            .
                            .   
    [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      5.03480168e-06  1.32378250e-06  2.00183060e-04  4.92930927e-07]
    [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
      1.32378249e-06  5.03480167e-06  4.92930927e-07  2.00183060e-04]]

With the pre-generated data, a ``Grid`` object can be initialized and 
its attributes printed:

.. code-block:: python

   grid = vib.Grid(res.mol,localmodes)
    
   print("modes  =", grid.modes )
   print("mol    =", grid.mol   )
   print("ngrid  =", grid.ngrid )
   print("amp    =", grid.amp   )
   print("grids  =", grid.grids )
   print("natoms =", grid.natoms)
   print("nmodes =", grid.nmodes)

.. code-block:: console

   modes  = <VibTools.Modes.VibModes object at 0x7f01c57aeb30>
   mol    = <VibTools.Molecule.VibToolsMolecule object at 0x7f01c57ae9e0>
   ngrid  = 0
   amp    = 0
   grids  = []
   natoms = 6
   nmodes = 12

The grid is initialized with molecular and mode data, but no grid points have been generated yet. 
The ``ngrid``, ``amp``, and ``grids`` attributes remain at their default values.

Grid points are generated using ``generate_grids()``:

.. code-block:: python
    
   ngrid = 16
   amp = 14
   grid.generate_grids(ngrid,amp)

   print("modes  =", grid.modes )
   print("mol    =", grid.mol   )
   print("ngrid  =", grid.ngrid )
   print("amp    =", grid.amp   )
   print("grids  =", grid.grids )
   print("natoms =", grid.natoms)
   print("nmodes =", grid.nmodes)        

.. code-block:: console
   
   modes  = <VibTools.Modes.VibModes object at 0x7f01c57aeb30>
   mol    = <VibTools.Molecule.VibToolsMolecule object at 0x7f01c57ae9e0>
   ngrid  = 16
   amp    = 14
   grids  = [[-80.11445092 -69.43252413 -58.75059734 -48.06867055 -37.38674376
     -26.70481697 -16.02289018  -5.34096339   5.34096339  16.02289018
      26.70481697  37.38674376  48.06867055  58.75059734  69.43252413
      80.11445092]
                                 .
                                 .
                                 .
    [-45.27522914 -39.23853192 -33.2018347  -27.16513748 -21.12844026
     -15.09174305  -9.05504583  -3.01834861   3.01834861   9.05504583
      15.09174305  21.12844026  27.16513748  33.2018347   39.23853192
      45.27522914]]
   natoms = 6
   nmodes = 12

Atomic numbers and Cartesian coordinates can be extracted for specific modes and grid points:

.. code-block:: python

   list_modes = [0,1] # list of modes of a given order
   points = [10,11] # list of points for given modes
   getgrid_structure = grid.get_grid_structure(list_modes,points)
   getgrid_atnums = getgrid_structure[0]
   getgrid_newcoords = getgrid_structure[1]
   print('\ngetgrid_structure =\n',getgrid_structure)
   print('\ngetgrid_atnums =\n',getgrid_atnums)
   print('\ngetgrid_newcoords =\n',getgrid_newcoords)


.. code-block:: console
   
   getgrid_structure =
    (array([1., 6., 6., 1., 1., 1.]), array([[-3.22513599e-02, -3.68791873e-01, -1.52748220e-03],
          [ 9.60981489e-04,  4.17493043e-02,  1.08952719e+00],
          [ 1.15518305e+00, -4.04915939e-02,  1.75597278e+00],
          [-9.60564918e-01,  9.83333440e-02,  1.60634722e+00],
          [ 1.18839541e+00, -1.05796358e-01,  2.84702745e+00],
          [ 2.11670884e+00,  3.61302396e-01,  1.23915287e+00]]))

   getgrid_atnums =
    [1. 6. 6. 1. 1. 1.]

   getgrid_newcoords =
    [[-3.22513599e-02 -3.68791873e-01 -1.52748220e-03]
    [ 9.60981489e-04  4.17493043e-02  1.08952719e+00]
    [ 1.15518305e+00 -4.04915939e-02  1.75597278e+00]
    [-9.60564918e-01  9.83333440e-02  1.60634722e+00]
    [ 1.18839541e+00 -1.05796358e-01  2.84702745e+00]
    [ 2.11670884e+00  3.61302396e-01  1.23915287e+00]]

.. note::

   >>> molecule = grid.get_molecule(list_modes,points)

   Same as `grid.get_grid_structure` but returns VibTools molecule object.

.. note::

   >>> PyadfMol = grid.get_pyadf_molecule(list_modes, points)

   Same as `grid.get_grid_structure` but returns PyADF molecule object.


The number of grid points and vibrational modes can be obtained using the following methods:


.. code-block:: python
   
   print('grid.get_number_of_grid_points() =', grid.get_number_of_grid_points())
   print('grid.get_number_of_modes() =', grid.get_number_of_modes())

.. code-block:: console
   
   grid.get_number_of_grid_points() = 16
   grid.get_number_of_modes() = 12

The grid can be also be saved in NumPy format (`*.npy`) and loaded:

.. code-block:: python
   
   grid.save_grids() 

.. code-block:: python
   
   grid.read_np('grids_12_16.npy')
