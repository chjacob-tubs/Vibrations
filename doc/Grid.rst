Grid
----

Module containing the grid-related class.

.. autoclass:: Vibrations.Grid
   :members:
   :show-inheritance:

Example
^^^^^^^

Import the Vibrations package:

.. code-block:: python

    import Vibrations as vib

The import from the package produces an output:

.. code-block:: console
     
        WARNING: Fortran routines not used for integrals, this might be very slow. 
             see src/Vibrations/README_f2py 
     
     ***************************************************************************
     *
     *  Vibrations v0.9
     *
     *  Vibrations - a Python Code for Anharmonic Theoretical Vibrational Spectroscopy
     *  Copyright (C) 2014-2018 by Pawel T. Panek, and Christoph R. Jacob.
     *
     *     Vibrations is free software: you can redistribute it and/or modify
     *     it under the terms of the GNU General Public License as published by
     *     the Free Software Foundation, either version 3 of the License, or
     *     (at your option) any later version.
     *
     *     Vibrations is distributed in the hope that it will be useful,
     *     but WITHOUT ANY WARRANTY; without even the implied warranty of
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



We create the Grid class and display the attributes:

.. code-block:: python

    # Create an empty grid
    grid = vib.Grid()
    
    modes  =grid.modes
    mol    =grid.mol
    ngrid  =grid.ngrid
    amp    =grid.amp
    grids  =grid.grids
    mol    =grid.mol
    natoms =grid.natoms
    nmodes =grid.nmodes
    
    
    print("modes =",modes  )
    print("mol   =",mol    )
    print("ngrid =",ngrid  )
    print("amp   =",amp    )
    print("grids =",grids  )
    print("mol   =",mol    )
    print("natoms=",natoms )
    print("nmodes=",nmodes )

Terminal output:

.. code-block:: console

    modes = None
    mol   = None
    ngrid = 0
    amp   = 0
    grids = []
    mol   = None
    natoms= 0
    nmodes= 0

We see that the attributes are filled with empty lists, None and zeros.

Now we take data (SNF-Calculations) from the folder (`vibrations/unittests/test_data`) and need the LocVib package to prepare data:

.. code-block:: python

   # Read in normal modes from SNF results
   # using VibTools (LocVib package)
   import VibTools
   path = '../'
   res = VibTools.SNFResults(outname = path+'snf.out',
                             restartname=path+'restart',
                             coordfile=path+'coord')
   res.read()

   # Now localize modes in separate subsets
   
   subsets = [[0,4], [4,8], [8,12]]

Now we still need to prepare the local modes via VibTools:


.. code-block:: python

   def localize_subsets(modes,subsets):
       # method that takes normal modes and list of lists (beginin and end)
       # of subsets and make one set of modes localized in subsets
   
       # first get number of modes in total
       total = 0
       modes_mw = np.zeros((0, 3*modes.natoms))
       freqs = np.zeros((0,))
   
       for subset in subsets:
           n = subset[1] - subset[0]
           total += n
   
   
       print('Modes localized: %i, modes in total: %i' %(total, modes.nmodes))
   
       if total > modes.nmodes:
           raise Exception('Number of modes in the subsets is larger than the total number of modes')
       else:
           cmat = np.zeros((total, total))
           actpos = 0 #actual position in the cmat matrix
           for subset in subsets:
               tmp = localize_subset(modes, range(subset[0], subset[1]))
               modes_mw = np.concatenate((modes_mw, tmp[0]), axis = 0)
               freqs = np.concatenate((freqs, tmp[1]), axis = 0)
               cmat[actpos:actpos + tmp[2].shape[0],actpos:actpos + tmp[2].shape[0]] = tmp[2]
               actpos = actpos + tmp[2].shape[0]
           localmodes = VibTools.VibModes(total, modes.mol)
           localmodes.set_modes_mw(modes_mw)
           localmodes.set_freqs(freqs)
   
           return localmodes, cmat

    localmodes,cmat = datprod.localize_subsets(res.modes,subsets)
    print('localmodes=',localmodes)
    print('cmat=',cmat)

.. code-block:: console

   localmodes= <VibTools.Modes.VibModes object at 0x7f01c74358d0>
   cmat= [[ 2.04579166e-05  1.31985469e-06  1.31897777e-06  0.00000000e+00
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

With the pre-produced data we can create a (defined) grid class and print it out:

.. code-block:: python

   grid = vib.Grid(res.mol,localmodes)
    
   print("modes =", grid.modes )
   print("mol   =", grid.mol   )
   print("ngrid =", grid.ngrid )
   print("amp   =", grid.amp   )
   print("grids =", grid.grids )
   print("natoms=", grid.natom )
   print("nmodes=", grid.nmodes)

.. code-block:: console

   modes = <VibTools.Modes.VibModes object at 0x7f01c57aeb30>
   mol   = <VibTools.Molecule.VibToolsMolecule object at 0x7f01c57ae9e0>
   ngrid = 0
   amp   = 0
   grids = []
   natoms= 6
   nmodes= 12

Now we generate a grid and print the grid class attributes:

.. code-block:: python
    
   ngrid = 16
   amp = 14
   grid.generate_grids(ngrid,amp)

   print("modes =", grid.modes )
   print("mol   =", grid.mol   )
   print("ngrid =", grid.ngrid )
   print("amp   =", grid.amp   )
   print("grids =", grid.grids )
   print("natoms=", grid.natom )
   print("nmodes=", grid.nmodes)        

.. code-block:: console
   
   modes = <VibTools.Modes.VibModes object at 0x7f01c57aeb30>
   mol   = <VibTools.Molecule.VibToolsMolecule object at 0x7f01c57ae9e0>
   ngrid = 16
   amp   = 14
   grids = [[-80.11445092 -69.43252413 -58.75059734 -48.06867055 -37.38674376
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
   natoms= 6
   nmodes= 12

With the next function we get the atomic numbers (atnums) and the corresponding new coodinates:

.. code-block:: python

   list_modes = [0,1] # list of modes of a given order
   points = [10,11] # list of points for given modes
   getgrid_structure = grid.get_grid_structure(list_modes,points)
   getgrid_atnums = getgrid_structure[0]
   getgrid_newcoords = getgrid_structure[1]
   print('\n\ngetgrid_atnums=\n',getgrid_atnums)
   print('\n\ngetgrid_newcoords=\n',getgrid_newcoords)


.. code-block:: console
   
   getgrid_structure=
    (array([1., 6., 6., 1., 1., 1.]), array([[-3.22513599e-02, -3.68791873e-01, -1.52748220e-03],
          [ 9.60981489e-04,  4.17493043e-02,  1.08952719e+00],
          [ 1.15518305e+00, -4.04915939e-02,  1.75597278e+00],
          [-9.60564918e-01,  9.83333440e-02,  1.60634722e+00],
          [ 1.18839541e+00, -1.05796358e-01,  2.84702745e+00],
          [ 2.11670884e+00,  3.61302396e-01,  1.23915287e+00]]))

.. note::

   >>> molecule = grid.get_molecule(list_modes,points)

   Same as `grid.get_grid_structure` but returns VibTools molecule object:

.. note::

   >>> PyadfMol = grid.get_pyadf_molecule(list_modes, points)

   Same as `grid.get_grid_structure` but returns PyADF molecule object:


Getting the number of grid_points and number of modes:


.. code-block:: python
   
   print('grid.get_number_of_grid_points()= ' , grid.get_number_of_grid_points() )
   print('grid.get_number_of_modes()= ' , grid.get_number_of_modes() )

.. code-block:: console
   
   grid.get_number_of_grid_points()= 16
   grid.get_number_of_modes()= 12

We can conveniently save and load the grid with in the numpy format (`*.npy`):

.. code-block:: python
   
   gird.save_grids() 

.. code-block:: python
   
   gird.read_np('grids_12_16.npy')


