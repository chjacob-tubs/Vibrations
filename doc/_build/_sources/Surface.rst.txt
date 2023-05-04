Surface
-------

Module containing classes related to property hypersurfaces: 
potential energy surface, dipole moment surface, and other
properties surfaces.


Surface
^^^^^^^

.. autoclass:: Vibrations.Surface
   :members:
   :show-inheritance:


Example
"""""""

Import the Vibrations and VibTools packages:

.. code-block:: python

    import VibTools
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



We create the (empty) Surface class and display the attributes:

.. code-block:: python
   
   surf = vib.Surface()
   print('grids  = ', surf.grids  )
   print('empty  = ', surf.empty  )
   print('ngrid  = ', surf.ngrid  )
   print('order  = ', surf.order  )
   print('prop   = ', surf.prop   )
   print('indices= ', surf.indices)
   print('data   = ', surf.data   )

.. code-block:: console
   
   grids  =  None
   empty  =  True
   ngrid  =  0
   order  =  1
   prop   =  (1,)
   indices=  []
   data   =  []

Now we create the Surface (Vibrations) object with the Grid (Vibrations depend on VibTools) object.
Therefore we need some SNF-Results (readable with VibTools):

.. code-block:: python
  
   # Load SNF-Results via VT 
   res = VibTools.SNFResults(outname = path+'snf.out',
                          restartname=path+'restart',
                          coordfile=path+'coord')
   res.read()

Now we still need to prepare the local modes via VibTools.
For more details look at the documentation of the Grid module of Vibrations.


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

   # Calculate Localmodes via VT - Data-Prod-function:
   subsets = [[0,4], [4,8], [8,12]]
   localmodes,cmat = datprod.localize_subsets(res.modes,subsets)
   # Create Grid Class
   mol = res.mol
   ngrid = 16
   amp = 14
   grid = vib.Grid(mol,localmodes)
   grid.generate_grids(ngrid,amp)
   
   surf = vib.Surface(grids=grid)
   print('grids  = ', surf.grids  )
   print('empty  = ', surf.empty  )
   print('ngrid  = ', surf.ngrid  )
   print('order  = ', surf.order  )
   print('prop   = ', surf.prop   )
   print('indices= ', surf.indices)
   print('data   = ', surf.data   )

.. code-block:: console
   
   grids  =  Grids:
   Number of modes:       12
   Number of grid points: 16
   
    Mode 0
   
    Normal coordinates: 
    -80.11  -69.43  -58.75  -48.07  -37.39  -26.70  -16.02   -5.34    5.34   16.02   26.70   37.39   48.07   58.75   69.43   80.11  Mode 1
   
    Normal coordinates: 
    -81.22  -70.39  -59.56  -48.73  -37.90  -27.07  -16.24   -5.41    5.41   16.24   27.07   37.90   48.73   59.56   70.39   81.22  Mode 2
    .
    .   
    .  
    Normal coordinates: 
    -45.27  -39.23  -33.20  -27.16  -21.13  -15.09   -9.05   -3.02    3.02    9.05   15.09   21.13   27.16   33.20   39.23   45.27  Mode 10
   
    Normal coordinates: 
    -45.28  -39.24  -33.20  -27.17  -21.13  -15.09   -9.06   -3.02    3.02    9.06   15.09   21.13   27.17   33.20   39.24   45.28  Mode 11
   
    Normal coordinates: 
    -45.28  -39.24  -33.20  -27.17  -21.13  -15.09   -9.06   -3.02    3.02    9.06   15.09   21.13   27.17   33.20   39.24   45.28 
   empty  =  False
   ngrid  =  16
   order  =  1
   prop   =  (1,)
   indices=  []
   data   =  []



  

Potential
^^^^^^^^^

.. autoclass:: Vibrations.Potential
   :members:
   :show-inheritance:


Example
"""""""

blablalbla



Polarizability
^^^^^^^^^^^^^^

blablabla

Example
"""""""

blablalbla




Atensor
^^^^^^^

blablabla

Example
"""""""

blablalbla





Dipole
^^^^^^

blablabalbla


Example
"""""""

blablalbla





