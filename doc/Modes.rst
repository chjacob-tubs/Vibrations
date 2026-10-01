Modes
-----

This module allows normal modes to be edited (getting/setting) within VibTools.


VibModes
^^^^^^^^

.. autoclass:: VibTools.VibModes
   :members:

Examples
^^^^^^^^

Preparation and Initiate VibTools.Modes
"""""""""""""""""""""""""""""""""""""""

 .. _Preparation:

LocVib(VibTools) should already be installed as a loadable Python module.

Import package:

   >>> import VibTools

To create the ``VibModes`` object, we need data from the ``Molecule`` class:

.. toctree::
   :maxdepth: 1

   Molecule

Let's use a water molecule as an example.

We can extract a coordinate file from the test folder ``/LocVib/src/Vibtools/tests/test_data/H2O``.

   >>> mol = VibTools.VibToolsMolecule()
   >>> mol.read_from_coord('test_data/H2O/coord')

The number of modes is given by ``nmodes = 3 * natoms - 6``, where *natoms* is the total number of atoms.

We can now create a ``VibModes`` object using the number of modes (``nmodes``) 
and the molecular structure (``mol``)::

   >>> nmodes = 3*mol.natoms-6
   >>> vimo = VibTools.VibModes(nmodes,mol)

We can display some relevant data (``freqs``: frequencies, ``modes_mw``: mass-weighted normal modes, 
``modes_c``: cartesian normal modes)::

   >>> print("\nnmodes =", vimo.nmodes)
   >>> print("\nnatoms =", vimo.natoms)
   >>> print("\nfreqs =", vimo.freqs)
   >>> print("\nmodes_mw =\n", vimo.modes_mw)
   >>> print("\nmodes_c =\n", vimo.modes_c)

   nmodes = 3 

   natoms = 3

   freqs = [1. 2. 3.]

   modes_mw =
    [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0.]]

   modes_c =
    [[0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0.]
    [0. 0. 0. 0. 0. 0. 0. 0. 0.]]


Set VibTools.Modes with Values from an SNF Calculation
""""""""""""""""""""""""""""""""""""""""""""""""""""""

 .. _SNFResults calculation:

First, execute the steps from the `Preparation`_ section.

We now obtain the normal-mode data for a water molecule from a PySNF calculation::

   >>> path = 'LocVib/src/VibTools/tests/test_data/H2O/'
   >>> res = VibTools.SNFResults(
   ...     outname=path+'snf.out',
   ...     restartname=path+'restart',
   ...     coordfile=path+'coord')
   >>> res.read()
   >>> output = res.snfoutput
   >>> SNF_modes_mw = output.modes.modes_mw
   >>> SNF_modes_c = output.modes.modes_c
   >>> SNF_freqs = output.modes.freqs

Assign the data from the PySNF calculation to the ``vimo`` object
and display the values::

  >>> modes_mw = vimo.set_modes_mw(SNF_modes_mw)
  >>> modes_c = vimo.set_modes_c(SNF_modes_c)
  >>> freqs = vimo.set_freqs(SNF_freqs)

  >>> print("\nfreqs =\n", vimo.freqs)
  >>> print("\nmodes_mw =\n", vimo.modes_mw)
  >>> print("\nmodes_c =\n", vimo.modes_c)

  freqs =
   [1575.16798 3685.32274 3796.19483]
   
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

.. note::

   - ``modes_mw`` is normalized in mass-weighted terms.
   - ``modes_c`` is normalized in Cartesian terms.

To obtain the normalized Cartesian modes, use::

   >>> modes_c_norm = vimo.get_modes_c_norm()
   >>> print("\nmodes_c_norm =\n", modes_c_norm)

   modes_c_norm =
    [[ 0.42732671  0.          0.56115199 -0.         -0.         -0.0707153
     -0.42732671 -0.          0.56115199]
    [-0.58500312  0.          0.39563677 -0.         -0.         -0.04985754
      0.58500312 -0.          0.39563677]
    [-0.55623749 -0.          0.43375482  0.07009441  0.         -0.
     -0.55623749  0.         -0.43375482]]


Editing Single Modes
""""""""""""""""""""

First, execute the steps from the `Preparation`_ section.

Individual rows of the normal-mode arrays can be edited. 
To do this, we select the *i*-th line and set the appropriate values::

   >>> i = 1
   >>> imode = np.array([1., 1., 1., 2., 2., 2., 3., 3., 3.])
   >>> ifreq = 1

Insert *imode* and *ifreq* in the modes-setters (mass-weighted)::

   >>> vimo.set_mode_mw(i, imode, freq=ifreq)
   >>> print("\nmodes_mw =\n", vimo.modes_mw)

   modes_mw =
    [[0.         0.         0.         0.         0.         0.
     0.         0.         0.        ]
    [0.15430335 0.15430335 0.15430335 0.3086067  0.3086067  0.3086067
     0.46291005 0.46291005 0.46291005]
    [0.         0.         0.         0.         0.         0.
     0.         0.         0.        ]]

The values at line *i=1* differ because the setter-function executes 
the update and normalization functions beforehand. 
This is analogous for the cartesian modes::

   >>> vimo.set_mode_c (i, imode, freq=ifreq)
   >>> print("\nmodes_c =\n", vimo.modes_c)

   modes_c =
    [[0.         0.         0.         0.         0.         0.
     0.         0.         0.        ]
    [0.06708936 0.06708936 0.06708936 0.13417872 0.13417872 0.13417872
     0.20126808 0.20126808 0.20126808]
    [0.         0.         0.         0.         0.         0.
     0.         0.         0.        ]]

Finally, we have the adjusted frequencies displayed::

   >>> print("freqs =", vimo.freqs)
   freqs = [1. 1. 3.]

Get a Smaller Subset of the Vibrational Spectrum
""""""""""""""""""""""""""""""""""""""""""""""""

This example shows how to extract a smaller sample of the spectrum. 
It's more useful for larger molecules, but for clarity, let's stick with water.

First, execute the steps from the `Preparation`_ section.
Then, execute the steps from the `SNFResults calculation`_ section::

   >>> path = 'test_data/H2O/'
   >>> res = VibTools.SNFResults(
   ...     outname=path+'snf.out',
   ...     restartname=path+'restart',
   ...     coordfile=path+'coord')
   >>> res.read()
   >>> SNFmodes_mw = output.modes.modes_mw
   >>> SNFmodes_c = output.modes.modes_c
   >>> SNFfreqs = output.modes.freqs

   >>> print("\nmodes_mw =\n", SNFmodes_mw)
   >>> print("\nmodes_c =\n", SNFmodes_c)
   >>> print("\nfreqs =\n", SNFfreqs)

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

Now, we can either use ``get_range`` to limit the range of the spectrum
in terms of frequencies, or use ``get_subset`` to directly select the modes::

   >>> modelist = range(0,2)
   >>> modes = res.modes.get_subset(modelist)
   >>> print("\nmodes_mw =\n", modes.modes_mw)
   >>> print("\nmodes_c =\n", modes.modes_c)
   >>> print("\nfreqs =\n", modes.freqs)

   modes_mw =
    [[ 0.41227257  0.          0.54138338 -0.         -0.         -0.2717917
     -0.41227257 -0.          0.54138338]
    [-0.57448154  0.          0.38852104 -0.         -0.         -0.19505052
      0.57448154 -0.          0.38852104]]
   
   modes_c =
    [[ 0.41066895  0.          0.53927756 -0.         -0.         -0.06795872
     -0.41066895 -0.          0.53927756]
    [-0.57224698  0.          0.38700981 -0.         -0.         -0.04877038
      0.57224698 -0.          0.38700981]]
   
   freqs =
    [1575.16798 3685.32274]

The same result is obtained with ``get_range``::

   >>> modes = res.modes.get_range(1000,3700)

We can also separate the modes with respect to different atoms,
where ``atomlist`` is a list of the corresponding atomic indices::

   >>> atomlist = [0,1]
   >>> modes = res.modes.get_fragment_modes(atomlist)
   >>> print("\nmodes_mw =\n", modes.modes_mw)

   modes_mw =
    [[ 0.56263072  0.          0.73882897 -0.         -0.         -0.37091567]
    [-0.79741252  0.          0.53928894 -0.         -0.         -0.27074104]
    [-0.7332129  -0.          0.57176051  0.36808798  0.         -0.        ]]

Modes - Gaussian System (g98.out)
"""""""""""""""""""""""""""""""""

First, execute the steps from the `Preparation`_ section.
Then, execute the steps from the `SNFResults calculation`_ section::

   >>> path = 'LocVib/src/VibTools/tests/test_data/H2O/'
   >>> res = VibTools.SNFResults(
   ...     outname=path+'snf.out',
   ...     restartname=path+'restart',
   ...     coordfile=path+'coord')
   >>> res.read()
   >>> output = res.snfoutput

With the results of the SNF calculation, we now write a ``g98.out`` file::

   >>> output.modes.write_g98out()

This generates the following ``g98.out`` file:

.. code-block:: console

    Entering Gaussian System
    *********************************************
    Gaussian 98:
    frequency fake output
    *********************************************
                            Standard orientation:
    --------------------------------------------------------------------
     Center     Atomic     Atomic              Coordinates (Angstroms)
     Number     Number      Type              X           Y           Z
    --------------------------------------------------------------------
       1          1             0       -2.403157   -1.117978   -0.002196
       2          8             0       -3.169188   -1.117978    0.595152
       3          1             0       -3.935218   -1.117978   -0.002196
    --------------------------------------------------------------------
        1 basis functions        1 primitive gaussians
        1 alpha electrons        1 beta electrons
    
     Harmonic frequencies (cm**-1), IR intensities (KM/Mole),
     Raman scattering activities (A**4/amu), Raman depolarization ratios,
     reduced masses (AMU), force constants (mDyne/A) and normal coordinates:
                          1                      2                     3
                         a                      a                      a
     Frequencies --  1575.1680              3685.3227              3796.1948
     Red. masses --     0.0000                 0.0000                 0.0000
     Frc consts  --     0.0000                 0.0000                 0.0000
     IR Inten    --     0.0000                 0.0000                 0.0000
     Raman Activ --     0.0000                 0.0000                 0.0000
     Depolar     --     0.0000                 0.0000                 0.0000
     Atom AN      X      Y      Z        X      Y      Z        X      Y      Z
       1   1     0.43   0.00   0.56    -0.59   0.00   0.40    -0.56  -0.00   0.43
       2   8    -0.00  -0.00  -0.07    -0.00  -0.00  -0.05     0.07   0.00  -0.00
       3   1    -0.43  -0.00   0.56     0.59  -0.00   0.40    -0.56   0.00  -0.43
    
    --------

Unitary Transformation of the Normal Modes
""""""""""""""""""""""""""""""""""""""""""

First, execute the steps from the `Preparation`_ section.
Then, execute the steps from the `SNFResults calculation`_ section::

   >>> path = 'test_data/H2O/'
   >>> res = VibTools.SNFResults(
   ...     outname=path+'snf.out',
   ...     restartname=path+'restart',
   ...     coordfile=path+'coord')
   >>> res.read()
   >>> mol = res.mol
   >>> natoms = mol.natoms
   >>> output = res.snfoutput
   >>> SNFmodes_mw = output.modes.modes_mw

Create the ``VibTools.Modes`` object and set the mass-weighted normal modes::

   >>> vimo = VibTools.VibModes(3*natoms-6,mol)
   >>> vimo_modes_mw = vimo.set_modes_mw(SNFmodes_mw)

Create an orthonormal (unitary) real matrix U (3 × 3);
UᵀU = I and Uᵀ = U⁻¹. 

The simplest example of this is a canonical basis matrix consisting of unit vectors::

   >>> tmatrix = np.asarray([[0,0,1],[0,1,0],[1,0,0]])

Apply the transformation function:

    Q_transformed = U Q 

    H_transformed = U H Uᵀ

where H is the diagonal Hessian matrix and Q the matrix of normal modes.

   >>> newmodes = vimo.transform(tmatrix)

Compare the original modes with the transformed ones::

   >>> print("\nmodes_mw =\n", SNFmodes_mw)
   >>> print("\nnewmodes =\n", newmodes.modes_mw)

   modes_mw =
    [[ 0.41227257  0.          0.54138338 -0.         -0.         -0.2717917
     -0.41227257 -0.          0.54138338]
    [-0.57448154  0.          0.38852104 -0.         -0.         -0.19505052
      0.57448154 -0.          0.38852104]
    [-0.536967   -0.          0.41872766  0.26956849  0.         -0.
     -0.536967    0.         -0.41872766]]
   
   newmodes =
    [[-0.536967    0.          0.41872766  0.26956849  0.          0.
     -0.536967    0.         -0.41872766]
    [-0.57448154  0.          0.38852104  0.          0.         -0.19505052
      0.57448154  0.          0.38852104]
    [ 0.41227257  0.          0.54138338  0.          0.         -0.2717917
     -0.41227257  0.          0.54138338]]


Useful Functions
^^^^^^^^^^^^^^^^

First, execute the steps from the `Preparation`_ section.
Then, execute the steps from the `SNFResults calculation`_ section::

   >>> res = VibTools.SNFResults(
   ...     outname=path+'snf.out',
   ...     restartname=path+'restart',
   ...     coordfile=path+'coord')
   >>> res.read()
   >>> mol = res.mol
   >>> natoms = mol.natoms
   >>> output = res.snfoutput
   >>> SNFmodes = output.modes
   >>> SNFmodes_mw = output.modes.modes_mw
   >>> SNFmodes_c = output.modes.modes_c
   >>> SNFfreqs = output.modes.freqs

Create a ``VibTools.Modes`` object::

   >>> vimo = VibTools.VibModes(3*natoms-6,mol)
   >>> vimo.set_modes_mw(SNFmodes_mw)
   >>> vimo.set_modes_c(SNFmodes_c)
   >>> vimo.set_freqs(SNFfreqs)

overlap
"""""""

The ``overlap`` function allows to calculate the overlap between two sets of modes.

If we use the same modes, we get a 100 percent overlap::

   >>> ov = vimo.overlap(SNFmodes)
   >>> print('ov =', ov)
   ov = [1. 1. 1.]

Now, we change the modes before calculating the overlap::

   >>> for i in range(3):
   >>>     vimo.modes_mw[i] = vimo.modes_mw[i]/(1.5*i+0.7)
   >>>     print(vimo.modes_mw[i])
   >>> ov = vimo.overlap(SNFmodes)
   >>> print('ov =', ov)
   ov = [2.04081633 0.20661157 0.07304602]

center
""""""

The ``center`` function allows the center of mass to be calculated::

   >>> modes_index = 0
   >>> cen = vimo.center(modes_index)
   >>> print('cen =', cen)
   cen = [-3.16918761 -1.1179783   0.04193042]


distance
""""""""

The ``distance`` function calculates the distance between two centers of mass.

We now calculate all possible distances of the given modes::

   >>> natoms = 3 # Number of atoms in H2O
   >>> dist = np.zeros((natoms,natoms))
   >>> for imode in range(natoms):
   >>>     for jmode in range(natoms):
   >>>         dist[imode,jmode] = vimo.distance(imode, jmode)
   >>> print('dist =\n', dist)
   dist =
    [[0.         0.02140059 0.00071894]
    [0.02140059 0.         0.02068165]
    [0.00071894 0.02068165 0.        ]]

As expected, the diagonal entries are zero because they are the distances to themselves.
