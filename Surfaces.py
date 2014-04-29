import numpy as np
import Misc

class Potential:

    def __init__(self, order=1, mol=None, modes=None, grids=None):

        if mol is None and modes is None and grids is None:
            self.natoms = 0
            self.nmodes = 0
            self.ngrid = 0
            self.empty = True

        else:
            self.nmodes = modes.nmodes
            self.ngrid = grids.get_number_of_grid_points()
            self.grids = grids.grids
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
        s = ''
        s += 'Potentials\n'
        s += 'Number of modes:        '+str(self.nmodes)
        s += 'Number of grid points:  '+str(self.ngrid)
        s += 'Order of the potential: '+str(self.order)

        return s

    def generate_harmonic(self,cmat=None):

        if not self.empty:
            if cmat is None:
                if self.order == 1:
                    for i in range(self.nmodes):
                        self.pot[i] = (self.grids[i] ** 2 * (self.modes.freqs[i] /
                                                               Misc.cm_in_au)**2) / 2.0
            else:
                if self.order == 1:
                    for i in range(self.nmodes):
                        self.pot[i] = (self.grids[i] ** 2 * cmat[i,i] ) / 2.0

                elif self.order == 2:
                    for i in range(self.nmodes):
                        for j in range(i+1,self.nmodes):
                            for k in range(self.ngrid):
                                for l in range(self.ngrid):
                                    self.pot[i,j,k,l] = self.grids[i,k] * self.grids[j,l] * cmat[i,j]
                                    self.pot[j,i,l,k] = self.pot[i,j,k,l]

    def read_np(self, fname):
        # read numpy binary file

        tmparray = np.load(fname)

        # check if orders are correct, warn when are different
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

        # fname - file with GAMESS formatted potential
        # order - 1 or 2 mode potential
        # ngrid - number of grid points per mode
        # modelist - list of modes to be included
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

    def save_potentials(self, fname='potentials'):

        from time import strftime
        fname = fname + '_' + strftime('%Y%m%d%H%M') + '.npy'
        np.save(fname, self.pot)