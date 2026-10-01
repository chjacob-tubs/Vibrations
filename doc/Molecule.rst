Molecule
--------

The ``Molecule.py`` module provides classes for working with molecular structures 
(e.g. from Turbomole or PySNF) in LocVib.

AbstractMolecule
^^^^^^^^^^^^^^^^

The ``AbstractMolecule`` class serves as a preparation for the `VibToolsMolecule`_ class and inherits its properties.

 .. _AbstractMolecule:

.. autoclass:: VibTools.AbstractMolecule

..   :members:

VibToolsMolecule
^^^^^^^^^^^^^^^^

 .. _VibToolsMolecule:

The ``VibToolsMolecule`` class is the main moleculare structure class in VibTools.
It relies on Open Babel and uses `AbstractMolecule`_.

.. autoclass:: VibTools.VibToolsMolecule
   :members:
   :show-inheritance:

Example
^^^^^^^

LocVib(VibTools) should already be installed as a loadable Python module.

Import package:

   >>> import VibTools
   
We can extract a coordinate file from the test folder ``/LocVib/tests/test_data/H2O``.

   >>> vtmole = VibTools.VibToolsMolecule()
   >>> vtmole.read_from_coord('test_data/H2O/coord')

Now, let's look at the properties of the loaded  molecule:

Show the total number of atoms:

   >>> print('natoms=', vtmole.natoms)

or

   >>> print('natoms=', vtmole.get_natoms())
   natoms= 3

Show the atom masses of the molecule:

   >>> print('atmasses=', vtmole.atmasses )

or

   >>> print('atmasses =', vtmole.get_atmasses())
   atmasses = [ 1.00782503 15.99491462  1.00782503]

Show the atomic numbers of the molecule:

   >>> print('atnums =', vtmole.atnums)

or

   >>> print('atnums =', vtmole.get_atnums())
   atnums = [1. 8. 1.]

Show the cartesian coordinates (x y z) of the molecule:

   >>> print('coordinates=', vtmole.coordinates)

or

   >>> print('coordinates =', vtmole.get_coordinates())
   coordinates = [[-2.40315717e+00 -1.11797830e+00 -2.19608983e-03]
   [-3.16918773e+00 -1.11797830e+00  5.95151641e-01]
   [-3.93521804e+00 -1.11797830e+00 -2.19608983e-03]]

We can save the data in LocVib format

   >>> filename = 'H2O_vt_data'
   >>> vtmole.write(filename)
   
Now, we should find the file ``H2O_vt_data`` in the current folder.
We can open the file e.g. with VIM:

.. code-block:: console

   >>> vi H2O_vt_data

   3
    
   H         -2.40316       -1.11798       -0.00220
   O         -3.16919       -1.11798        0.59515
   H         -3.93522       -1.11798       -0.00220

We can now read this file back into LocVib:

   >>>  vtmole = VibTools.VibToolsMolecule()
   >>>  vtmole.read(filename)

If we want to add more atoms to an existing molecule, we can use the following function.
Our test atom is here a H-atom (atomic number: 1) with the coordinates (-1 1 1):

   >>> vtmole.add_atoms([1],[[-1, 1, 1]])

The new coordinate and atomic number lists are as follows. They can be displayed with 
the commands above (natoms, atmasses, atnums, coordinates):

.. code-block:: console

   natoms = 4
   atmasses = [ 1.00782503 15.99491462  1.00782503  1.00782503]
   atnums = [1. 8. 1. 1.]
   coordinates = [[-2.40315717e+00 -1.11797830e+00 -2.19608983e-03]
    [-3.16918773e+00 -1.11797830e+00  5.95151641e-01]
    [-3.93521804e+00 -1.11797830e+00 -2.19608983e-03]
    [-1.00000000e+00  1.00000000e+00  1.00000000e+00]]

We can also take fragments of a molecule and process them analogously.
For this, we use the internal index list of the considered molecule.
If we continue to use the water example, we have a complete index list:

   >>> atomlist = [0,1,2]

Execute the ``get_fragment`` method to  extract a fragment:

   >>> frag = vtmole.get_fragment(atomlist)
   >>> print('atomic numbers:', frag.atnums)
   atomic numbers: [1. 8. 1.]

Now, we omit an index number:

   >>> atomlist = [0,2]
   >>> frag = vtmole.get_fragment(atomlist)
   >>> print('atomic numbers:', frag.atnums)
   atomic numbers: [1. 1.]

