import numpy as np

class Wavefunction:

    # class containing wave functions
    # and some methods to manipulate it

    def __init__(self, grids=None):  # initialize with grid

        if grids is None:
            self.nmodes = 0
            self.ngrid = 0
            self.nstates = 0
            self.wfns = np.zeros((self.nmodes, self.nstates, self.ngrid))

        else:
            self.nmodes = grids.get_number_of_modes()
            self.ngrid = grids.get_number_of_grid_points()
            self.nstates = self.ngrid
            self.wfns = np.zeros((self.nmodes, self.nstates, self.ngrid))

    def save_wavefunctions(self, fname='wavefunctions'):
        from time import strftime
        fname = fname + '_' + strftime('%Y%m%d%H%M') + '.npy'
        np.save(fname, self.wfns)

    def read_wavefunctions(self, fname='wavefunctions.npy'):

        tmpwfns = np.load(fname)
        self.nmodes = tmpwfns.shape[0]
        self.ngrid = tmpwfns.shape[2]
        self.nstates = tmpwfns.shape[1]
        self.wfns = tmpwfns.copy()