"""
The module containing the classes related to hypersurfaces: potential energy surface, dipole moment surface, and other
properties surfaces.
"""

import numpy as np
import Misc


class Potential:
    """
    The class for containing and manipulating potential energy surfaces
    """
    def __init__(self, order=1, mol=None, modes=None, grids=None):
        """
        The class can be initialized with a PyADF/Molecule object, VibTools/VibModes object, and vibrations/Grid object.
        If the object are not given, an empty object is created, which can be used to read in an existing potential
        energy surface stored in a NumPy formatted binary file *.npy

        @param order: The dimensionality of the potential energy surface, 1 -- one-mode terms, 2 -- two-mode terms
        @type order: Integer
        @param mol: Object containing the molecular structure
        @type mol: PyADF/Molecule
        @param modes: Object containing the vibrational modes
        @type modes: VibTools/VibModes
        @param grids: Object containing the grids
        @type grids: Vibrations/Grid
        """
        if mol is None and modes is None and grids is None:
            self.natoms = 0
            self.nmodes = 0
            self.ngrid = 0
            self.empty = True

        else:
            self.nmodes = modes.nmodes
            self.ngrid = grids.get_number_of_grid_points()
            self.grids = grids.grids  # TODO this should be done via some method (using .copy()) not directly
            self.modes = modes
            self.empty = False

        self.order = order

        if self.order == 1:
            self.pot = np.zeros((self.nmodes, self.ngrid))

        elif self.order == 2:
            self.pot = np.zeros((self.nmodes, self.nmodes, self.ngrid, self.ngrid))

        else:
            raise Exception('Higher order potentials are not supported')

    def __str__(self):
        """
        Prints out some general informations about the potential energy surface
        """
        s = ''
        s += 'Potentials\n'
        s += 'Number of modes:        '+str(self.nmodes)
        s += 'Number of grid points:  '+str(self.ngrid)
        s += 'Order of the potential: '+str(self.order)

        return s

    def generate_harmonic(self, cmat=None):

        if not self.empty:
            if cmat is None:
                if self.order == 1:
                    for i in range(self.nmodes):
                        self.pot[i] = (self.grids[i] ** 2 * (self.modes.freqs[i] /
                                                             Misc.cm_in_au)**2) / 2.0
            else:
                if self.order == 1:
                    for i in range(self.nmodes):
                        self.pot[i] = (self.grids[i] ** 2 * cmat[i, i]) / 2.0

                elif self.order == 2:
                    for i in range(self.nmodes):
                        for j in range(i+1, self.nmodes):
                            for k in range(self.ngrid):
                                for l in range(self.ngrid):
                                    self.pot[i, j, k, l] = self.grids[i, k] * self.grids[j, l] * cmat[i, j]
                                    self.pot[j, i, l, k] = self.pot[i, j, k, l]

    def read_np(self, fname):
        """
        Reads in the existing potential energy surface from a NumPy formatted binary file *.npy

        @param fname: File name, without extension
        @type fname: String
        """

        tmparray = np.load(fname)

        if len(tmparray.shape) == 2:
            if self.order != 1:
                print 'Warning: order of the potential and shape of the input array do not match.\
                       Order will be corrected'
                self.order = 1
                print 'New order: ', self.order
                print ''

        elif len(tmparray.shape) == 4:
            if self.order != 2:
                print 'Warning: order of the potential and shape of the input array do not match.\
                       Order will be corrected'
                self.order = 2
                print 'New order: ', self.order
                print ''
        else:
            raise Exception('Input data shape mismatch, check shape of stored arrays')

        self.pot = tmparray.copy()
        self.nmodes = self.pot.shape[0]

        if self.order == 1:
            self.ngrid = self.pot.shape[1]

        elif self.order == 2:
            self.ngrid = self.pot.shape[3]

    def read_gamess(self, fname, order, ngrid, modeslist):
        """
        Reads in the GAMESS-formatted potential energy surface

        @param fname: The file containing the potential energy surface
        @type fname: String
        @param order: PES dimensionality
        @type order: Integer
        @param ngrid: Number of grid points
        @type ngrid: Integer
        @param modeslist: List of modes (in Gamess numbering -- starts from 7, frequency sorted) to be read in
        @type modeslist: List of Integers
        """

        self.nmodes = len(modeslist)
        self.order = int(order)
        self.ngrid = int(ngrid)

        print 'GAMESS-formatted potential from file %s will be read in' % fname
        print 'Order : %i' % self.order
        print 'NModes: %i' % self.nmodes
        print 'NGrid : %i' % self.ngrid
        print 'Modes : ', modeslist

        if self.order == 1:
            v1 = np.zeros((self.nmodes, self.ngrid))

        elif self.order == 2:
            v2 = np.zeros((self.nmodes, self.nmodes, self.ngrid, self.ngrid))

        else:
            raise Exception('Only 1 and 2-Mode potentials allowed')

        f = open(fname, 'r')
        lines = f.readlines()
        f.close()

        starts = []

        for i, l in enumerate(lines):
            if 'MODE=' in l:
                starts.append(i)

        for i in starts:
            ls = lines[i].split()
            ls1 = lines[i+1].split()  # ls + 1
            ls2 = lines[i+2].split()  # ls +2
            if len(ls) == 2 and self.order == 1:
                #1 mode potential
                mode = int(ls[1])
                if mode in modeslist:
                    gp = int(ls1[1])
                    pot = float(ls2[1])
                    mode = mode - modeslist[0]  # going back from gamess numbering to ours
                    v1[mode, gp-1] = pot   # -1 because of python numbering

            elif len(ls) == 3 and self.order == 2:
                #2 mode potential
                model = int(ls[1])
                moder = int(ls[2])
                if (model in modeslist) and (moder in modeslist):
                    gpl = int(ls1[1])
                    gpr = int(ls1[2])
                    pot = float(ls2[1])

                    model = model - modeslist[0]  # first active mode will have index 0 and so on
                    moder = moder - modeslist[0]

                    v2[model, moder, gpl-1, gpr-1] = pot
                    v2[moder, model, gpr-1, gpl-1] = pot

        if self.order == 1:
            self.pot = np.copy(v1)

        elif self.order == 2:
            self.pot = np.copy(v2)

        else:
            raise Exception('Something went wrong')

    def save(self, fname='pot'):
        """
        Saves the potential energy surface to a NumPy formatted binary file *.npy

        @param fname: Filename, without extension. A time-stamp is added.
        @type fname: String
        """
        from time import strftime
        fname = fname + '_v' + str(self.order) + '_' + strftime('%Y%m%d%H%M') + '.npy'
        np.save(fname, self.pot)


class Dipole:
    """
    The class representing dipole moment surfaces
    """

    def __init__(self, order=1, mol=None, modes=None, grids=None):
        """
        The class can be initialized with a PyADF/Molecule object, VibTools/VibModes object, and vibrations/Grid object.
        If the object are not given, an empty object is created, which can be used to read in an existing dipole
        moment surface stored in a NumPy formatted binary file *.npy

        @param order: The dimensionality of the dipole moment surface, 1 -- one-mode terms, 2 -- two-mode terms
        @type order: Integer
        @param mol: Object containing the molecular structure
        @type mol: PyADF/Molecule
        @param modes: Object containing the vibrational modes
        @type modes: VibTools/VibModes
        @param grids: Object containing the grids
        @type grids: Vibrations/Grid
        """
        if mol is None and modes is None and grids is None:
            self.natoms = 0
            self.nmodes = 0
            self.ngrid = 0
            self.empty = True

        else:
            self.nmodes = modes.nmodes
            self.ngrid = grids.get_number_of_grid_points()
            self.grids = grids.grids  # TODO this should be done via some method (using .copy()) not directly
            self.modes = modes
            self.empty = False

        self.order = order

        if self.order == 1:
            self.dm = np.zeros((self.nmodes, self.ngrid))

        elif self.order == 2:
            self.dm = np.zeros((self.nmodes, self.nmodes, self.ngrid, self.ngrid))

        else:
            raise Exception('Higher order dipole moment surfaces are not supported')

    def read_np(self, fname):
        """
        Reads in the existing dipole moment surface from a NumPy formatted binary file *.npy

        @param fname: File name, without extension
        @type fname: String
        """

        tmparray = np.load(fname)
        print tmparray.shape
        if len(tmparray.shape) == 3:
            if self.order != 1:
                print 'Warning: order of the potential and shape of the input array do not match.\
                       Order will be corrected'
                self.order = 1
                print 'New order: ', self.order
                print ''

        elif len(tmparray.shape) == 5:
            if self.order != 2:
                print 'Warning: order of the potential and shape of the input array do not match.\
                       Order will be corrected'
                self.order = 2
                print 'New order: ', self.order
                print ''
        else:
            raise Exception('Input data shape mismatch, check shape of stored arrays')

        self.dm = tmparray.copy()
        self.nmodes = self.dm.shape[0]

        if self.order == 1:
            self.ngrid = self.dm.shape[1]

        elif self.order == 2:
            self.ngrid = self.dm.shape[3]

    def save(self, fname='dm'):
        """
        Saves the dipole moment surface to a NumPy formatted binary file *.npy

        @param fname: Filename, without extension. A time-stamp is added.
        @type fname: String
        """
        from time import strftime
        fname = fname + '_' + str(self.order) + '_' + strftime('%Y%m%d%H%M') + '.npy'
        np.save(fname, self.dm)
