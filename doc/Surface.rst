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

When importing the packages, the following information is displayed:

.. code-block:: console

     
     ***************************************************************************
     *
     * Current Vibrations Header and Copyright *
     *
     ***************************************************************************
     *
     *



An empty ``Surface`` object can be created. Its attributes are printed below:

.. code-block:: python
   
   surf = vib.Surface()
   print('grids   =', surf.grids  )
   print('empty   =', surf.empty  )
   print('ngrid   =', surf.ngrid  )
   print('order   =', surf.order  )
   print('prop    =', surf.prop   )
   print('indices =', surf.indices)
   print('data    =', surf.data   )

.. code-block:: console
   
   grids   =  None
   empty   =  True
   ngrid   =  0
   order   =  1
   prop    =  (1,)
   indices =  []
   data    =  []

To construct a functional ``Surface`` object, a grid with computed vibrational points is required.
The SNF calculation results can be loaded using the ``SNFResults`` class from VibTools 
(SNF: https://reiher.ethz.ch/software/snf.html).

.. code-block:: python
  
   # Load SNF results using VibTools 
   path = 'src/Vibrations/unittests/test_data/'
   res = VibTools.SNFResults(outname = path+'snf.out',
                          restartname=path+'restart',
                          coordfile=path+'coord')
   res.read()

Next, the local modes are prepared using VibTools.
For more details, see the documentation of the :ref:`Grid module <Grid>` in the Vibrations package.

.. code-block:: python

   # Localize modes in separate subsets using VibTools:
   subsets = [[0,1,2,3],[4,5,6,7],[8,9,10,11]]
   localmodes,cmat = VibTools.LocVib.localize_subsets(subsets,res.modes)
   # Create Grid class
   mol = res.mol
   ngrid = 16
   amp = 14
   grid = vib.Grid(mol,localmodes)
   grid.generate_grids(ngrid,amp)
   
   surf = vib.Surface(grids=grid)
   print('grids   =', surf.grids  )
   print('empty   =', surf.empty  )
   print('ngrid   =', surf.ngrid  )
   print('order   =', surf.order  )
   print('prop    =', surf.prop   )
   print('indices =', surf.indices)
   print('data    =', surf.data   )

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

...


Polarizability
^^^^^^^^^^^^^^

.. autoclass:: Vibrations.Polarizability
   :members:
   :show-inheritance:

Example
"""""""

...


Gtensor
^^^^^^^

.. autoclass:: Vibrations.Gtensor
   :members:
   :show-inheritance:


Atensor
^^^^^^^

.. autoclass:: Vibrations.Atensor
   :members:
   :show-inheritance:

Example
"""""""

...


Dipole
^^^^^^

.. autoclass:: Vibrations.Dipole
   :members:
   :show-inheritance:

Example
"""""""

...
