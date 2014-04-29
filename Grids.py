"""
The module containing the grid-related class.
"""

import numpy as np
import Misc


class Grid:
    """
    Class containing and manipulating the grids, which are used both for Potential Energy Surface (PES) or Dipole Moment
    Surface (DMS) evaluation, and for VSCF/VCI calculations.
    """

    def __init__(self, mol=None, modes=None):
        """
        The class can be initialized with objects referring to the molecular structure and the normal modes. By default,
        an empty grid is created, this may be used to read in an existing grid from *.npy file, see read_np method

        @param mol: molecule object
        @type mol: PyADF Molecule
        @param modes: vibrational modes object
        @type modes: VibTools Modes
        """

        if mol is None and modes is None:
            self.natoms = 0
            self.nmodes = 0
            self.ngrid = 0
            self.amp = 0
            self.grids = np.zeros(0)  # just an empty array for later

        else:

            self.natoms = mol.get_natoms()
            self.nmodes = modes.nmodes
            self.modes = modes
            self.ngrid = 0
            self.amp = 0
            self.grids = np.zeros(0)

    def __str__(self):

        s = ''
        s += 'Grids:\n'
        s += 'Number of modes:       ' + str(self.nmodes) + '\n'
        s += 'Number of grid points: ' + str(self.ngrid) + '\n'
        s += '\n'
        for i in range(self.nmodes):
            s += ' Mode ' + str(i) + '\n'
            s += '\n'
            st = ''
            for j in range(self.ngrid):
                st += ' ' + ('%6.2f' % self.grids[i, j]) + ' '

            s += ' Normal coordinates: \n'
            s += st

        return s

    def generate_grids(self, ngrid, amp):   # generate grid with ngrid number of points per mode and amp amplitude
        """
        Method generating the grids for a given molecule and vibrational modes

        @param ngrid: number of grid points
        @type ngrid: Integer
        @param amp: grid amplitude
        @type amp: Real+
        """
        grids = np.zeros((self.nmodes, ngrid))
        self.ngrid = ngrid
        self.amp = amp
        for i in range(self.nmodes):

            f = self.modes.freqs[i]
            fau = f * Misc.cm_in_au

            qrange = np.sqrt((2.0*(float(amp)+0.5))/fau)
            dq = 2.0 * qrange / (float(ngrid)-1.0)
            grid = [-qrange + j * dq for j in range(ngrid)]
            grids[i] = grid

        self.grids = np.copy(grids)

    def read_np(self, fname):
        """
        Read in an already existing grid from NumPy formatted binary file *.npy

        @param fname: file name
        @type fname: String
        """
        tmparray = np.load(fname)
        self.grids = tmparray.copy()
        self.nmodes = self.grids.shape[0]
        self.ngrid = self.grids.shape[1]

    def get_number_of_grid_points(self):
        return self.ngrid

    def get_number_of_modes(self):
        return self.nmodes

    def save_grids(self, fname='grids'):
        """
        Save the grid contained in the object to a NumPy formatted binary file *.npy

        @param fname: file name, without extension
        @type fname: String
        """
        from time import strftime
        fname = fname + '_' + strftime('%Y%m%d%H%M') + '.npy'
        np.save(fname, self.grids)
