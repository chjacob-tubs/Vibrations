import numpy as np
import Misc

class Grid:

    def __init__(self, mol=None, modes=None):

        # mol -- PyADF molecule
        # modes -- VibTools modes
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

        #printing in a neat form
        print 'Grids:\n'
        print 'Number of modes:       ', self.nmodes
        print 'Number of grid points: ', self.ngrid
        print ''
        for i in range(self.nmodes):
            print ' Mode ', i
            s = ''
            for j in range(self.ngrid):
                s += ' ' + ('%6.2f' % self.grids[i, j]) + ' '

            print ' Normal coordinates: '
            print s

        return ''

    def generate_grids(self, ngrid, amp):   # generate grid with ngrid number of points per mode and amp amplitude

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

    def read_np(self, fname):  # read numpy binary file (npy)
        tmparray = np.load(fname)
        self.grids = tmparray.copy()
        self.nmodes = self.grids.shape[0]
        self.ngrid = self.grids.shape[1]

    def get_number_of_grid_points(self):
        return self.ngrid

    def get_number_of_modes(self):
        return self.nmodes

    def save_grids(self, fname='grids'):

        from time import strftime
        fname = fname + '_' + strftime('%Y%m%d%H%M') + '.npy'
        np.save(fname, self.grids)
