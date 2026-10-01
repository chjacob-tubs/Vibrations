LocVib
======

Localizing normal modes.

LocVib
------

.. autoclass:: VibTools.LocVib
   :members:


Preparation and Initiate VibTools.Modes
"""""""""""""""""""""""""""""""""""""""

 .. _Preparation:

LocVib(VibTools) should already be installed as a loadable Python module.

Import package:

   >>> import VibTools

To create the ``LocVib`` object, we need the normal mode data provided by 
the ``Modes`` class (``Modes`` based on the ``Molecule`` class):

.. toctree::
   :maxdepth: 1

   Molecule
   Modes

For this, we create a dataset using the ``PySNF`` interface class

.. toctree::
   :maxdepth: 1

   PySNF

The following example uses the H2O test data included in the repository.
Make sure to use the correct path to the test data:

   >>> path = '/LocVib/src/VibTools/tests/test_data/H2O/'
   >>> res = VibTools.SNFResults(
   ...     outname=path+'snf.out',
   ...     restartname=path+'restart',
   ...     coordfile=path+'coord')
   >>> res.read()
   >>> vibmodes = res.modes

Show data:

   >>> print('natoms =', vibmodes.natoms)
   >>> print('nmodes =', vibmodes.nmodes)
   >>> print('modes_mw =\n', vibmodes.modes_mw)
   >>> print('modes_c =\n', vibmodes.modes_c)
   >>> print('freqs =\n', vibmodes.freqs)
   natoms = 3
   nmodes = 3
   modes_mw =
    [[ 0.41227257  0.          0.54138338 -0.         -0.         -0.2717917
     -0.41227257 -0.          0.54138338]
    [-0.57448154  0.          0.38852104 -0.         -0.         -0.19505052
      0.57448154 -0.          0.38852104]
    [-0.536967   -0.          0.41872766  0.26956849  0.         -0.
     -0.536967    0.         -0.41872766]]
   modes_c =
    [[ 0.41066895  0.          0.53927756 -0.         -0.         -0.06795872
     -0.41066895 -0.          0.53927756]
    [-0.57224698  0.          0.38700981 -0.         -0.         -0.04877038
      0.57224698 -0.          0.38700981]
    [-0.53487836 -0.          0.41709893  0.06740284  0.         -0.
     -0.53487836  0.         -0.41709893]]
   freqs =
    [1575.16798 3685.32274 3796.19483]

Create the ``LocVib`` object using the normal modes:
 
   >>> lv = VibTools.LocVib(vibmodes)

Show data:

   >>> print('natoms =', lv.natoms)
   >>> print('nmodes =', lv.nmodes)
   >>> print('loctype =', lv.loctype)
   >>> print('transmat =/n', lv.transmat)
   >>> print('subsets =', lv.subsets)
   natoms = 3
   nmodes = 3
   loctype = PM
   transmat =
    [[1. 0. 0.]
    [0. 1. 0.]
    [0. 0. 1.]]
   subsets = None

Maximize Localization
"""""""""""""""""""""

After the normal modes have been prepared as described in the previous `Preparation`_ section,
the normal modes are ready to be localized.

We define the following:

   >>> modes_mw = vibmodes.modes_mw

to calculate the localization measure `ξ(Q)`: 

   >>> p = lv.calc_p(modes_mw)
   >>> print('calc_p= ',p)
   calc_p= 1.333694596761428                         

Next, we want to maximize localization by solving the optimization problem `\max(ξ(Q))`:

   >>> transmat,del_p = lv.try_localize()

With the given optimization parameters, 
the following transformation matrix and the final distance between the last computed 
localization measures (iteration cycles) are obtained:

   >>> print('transmat =\n', transmat)
   transmat =
    [[ 5.78585531e-03  7.07083109e-01 -7.07106781e-01]
    [-9.99966523e-01  8.18243563e-03  5.76447963e-10]
    [ 5.78585613e-03  7.07083110e-01  7.07106781e-01]]
   >>> print('del_p =', del_p)
   del_p = 0.8871338845549153

The next function performs the maximization (above), displays the progress of each cycle, 
and updates the transformation matrix of the ``LocVib`` object:

   >>> lv.localize()
   Normal mode localization: Cycle   1    p:    1.936   change:  0.6020189     1.95004 
   Normal mode localization: Cycle   2    p:    2.220   change:  0.2847694     0.96979 
   Normal mode localization: Cycle   3    p:    2.221   change:  0.0003456     0.02169 
   Normal mode localization: Cycle   4    p:    2.221   change:  0.0000000     0.00002 


Localize Subsets of Normal Modes
""""""""""""""""""""""""""""""""

Normal modes can be put into subsets and localized separately. In this example, 
all three modes are localized in a single subset:

   >>> modelist = [[0,1,2]]
   
   >>> lv.localize_subsets(modelist)
    Normal mode localization: Cycle   1    p:    0.893   change:  0.0000000     0.00000 
    Normal mode localization: Cycle   1    p:    1.328   change:  0.0000000     0.00000 
    Normal mode localization: Cycle   1    p:    2.221   change: -0.0000000     0.00000 

   >>> lv.localize_automatic_subsets(maxerr = 1)
   >>> dif = lv.startmodes.modes_mw - lv.locmodes.modes_mw
   >>> print(dif)
   [[ 8.29232022e-01  0.00000000e+00  1.07956960e+00 -3.87814646e-09
     -0.00000000e+00 -5.41978318e-01 -8.29232006e-01 -0.00000000e+00
      1.07956962e+00]
    [ 2.09032293e-01  0.00000000e+00 -1.85413149e-01 -1.90613706e-01
     -0.00000000e+00 -5.55610471e-02  5.50353706e-01 -0.00000000e+00
      4.06757176e-01]
    [-5.12839155e-01 -0.00000000e+00  4.36963804e-01  4.60182201e-01
      0.00000000e+00  1.39489472e-01 -1.32048085e+00  0.00000000e+00
     -9.92661844e-01]]


AutomaticAssignment
-------------------

.. autoclass:: VibTools.AutomaticAssignment
   :members:

in progress.


