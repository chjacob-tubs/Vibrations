import numpy as np
import VibTools


def fancy_box(s):  # doing a fancy box around a string

    s = str(s)
    l = len(s)
    l += 10

    s1 = '+'+(l-2)*'-'+'+\n'
    s2 = '|'+3*'-'+' '+s+' '+3*'-'+'|\n'

    return s1+s2+s1


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
            fau = f * VibTools.Constants.cm_in_au
            
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
                                                               VibTools.Constants.cm_in_au)**2) / 2.0
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


class VSCF:

    # mother class for VSCF

    def __init__(self, grids, *potentials):  # always initialized with grids and (some) potentials

        self.nmodes = grids.get_number_of_modes()
        self.ngrid = grids.get_number_of_grid_points()
        self.nstates = self.ngrid

        #self.wavefunction = Wavefunction(grids)  # initialize an object to store wave function
        self.wavefunction = Wavefunction(grids)
        self.wfns = self.wavefunction.wfns
        self.eigv = np.zeros((self.nmodes, self.nstates))
        self.grids = grids.grids
        self.grid_object = grids
        self.dx = [x[1]-x[0] for x in self.grids]  # integration step

        self.solved = False  # simple switch to check, whether VSCF was already solved

    def _collocation(self, grid, potential):
        # some basis set parameters
        c = 0.7
        a = c**2/((grid[1]-grid[0])**2)
        # end of parameters

        wfn = np.zeros((self.ngrid, self.ngrid))

        matg = np.zeros((self.ngrid, self.ngrid))

        fac = (2.0*a/np.pi)**0.25

        for i in range(self.ngrid):
            for j in range(self.ngrid):
                wfn[i, j] = fac * np.exp(-a*(grid[j]-grid[i])**2)
                dr2 = (4.0*(a**2)*((grid[j]-grid[i])**2))-2.0*a
                matg[i, j] = -0.5*dr2*wfn[i, j]

        # create potential matrix from a vector
        matv = np.diagflat(potential)
        # inverse matrix of wfn (R in paper)
        invwfn = np.linalg.inv(wfn)

        # multiply G and inverted wfn 
        matgr = np.dot(matg, invwfn)

        # generate hamiltonian
        hamiltonian = matgr + matv

        # solve the Hamiltonian

        (eigval, eigvec) = np.linalg.eig(hamiltonian)

        # sort eigenvalues and eigenvectors with respect to the eigenvalues

        idx = eigval.argsort()
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]

        # find wave functions
        phi = np.zeros((self.ngrid, self.ngrid))

        for i in range(self.ngrid):
            for j in range(self.ngrid):
                phi[i, j] = eigvec[j, i]*wfn[j, j]

        # correct the sings and norm the wave function

        for i in range(self.ngrid):
            psum = phi[i, :].sum()

            if i % 2 == 0 and psum < 0:
                phi[i, :] = -phi[i, :]

            elif i % 2 != 0 and psum > 0:
                phi[i, :] = -phi[i, :]

            phi[i, :] = self._norm(phi[i, :], grid[1]-grid[0])

        return eigval, phi
        
    def _norm(self, phi, dx):

        wnorm = 0.0

        for i in range(self.ngrid):
            wnorm += dx*phi[i]**2

        normphi = np.array([x * 1.0/np.sqrt(wnorm) for x in phi])

        # check norm

        #p = 0.0
        #p = (normphi**2 * dx).sum()
        #if p <> 1.0:
        #    raise Exception('Something went wrong with wave function normalization')

        return normphi
    
    def get_wave_functions(self):

        return self.wfns

    def get_wave_function_object(self):

        return self.wavefunction

    def save_wave_functions(self, fname='wavefunctions'):
        
        from time import strftime
        fname = fname + '_' + strftime('%Y%m%d%H%M') + '.npy'
        np.save(fname, self.wfns)


class VSCFDiag(VSCF):

    # diagonal VSCF class

    def __init__(self, grids, *potentials):

        VSCF.__init__(self, grids, potentials)  # fist call the constructor of mother class
        
        if len(potentials) == 0:
            raise Exception('No potential given')

        elif len(potentials) > 1:
            print 'More than one potentials given, only the first will be used'

        self.v1 = potentials[0].pot
        if self.nmodes != self.v1.shape[0] or self.ngrid != self.v1.shape[1]:
            raise Exception('Potential and grid size mismatch')

        self.solved = False

    def solve(self):

        if self.solved:
            print 'Already solved, nothing to do. See results with print_results() method'

        else:

            for i in range(self.nmodes):
                print self.grids[i]
                print self.v1[i]
                (tmpeigv, tmpwfn) = self._collocation(self.grids[i], self.v1[i])
                self.eigv[i] = tmpeigv
                self.wfns[i] = tmpwfn
            
            self.solved = True

    def print_results(self):

        if self.solved:

            print 'Fundamental transitions:'
            for i in range(self.nmodes):
                print 'Mode %i, eigv: %f' % (i, (self.eigv[i, 1]-self.eigv[i, 0])/VibTools.Constants.cm_in_au)

            print 'Eigenvalues: '
            for i in range(self.nmodes):
                print 'Mode %i, eigv: %f' % (i, self.eigv[i, 0]/VibTools.Constants.cm_in_au)
        else:
            print 'VSCF not solved yet. Use solve() method first'

    def print_eigenvalues(self):

        for i in range(self.nmodes):
            print 'Mode %i' % i

            for j in range(self.nstates):
                print self.eigv[i, j], self.eigv[i, j]/VibTools.Constants.cm_in_au

    def save_wave_functions(self, fname='1D_wavefunctions'):

        from time import strftime
        fname = fname + '_' + strftime('%Y%m%d%H%M') + '.npy'
        np.save(fname, self.wfns)


class VSCF2D(VSCF):

    # 2D VSCF, requires 2D potentials

    def __init__(self, grids, wavefunctions, *potentials):

        VSCF.__init__(self, grids, potentials)
        
        if len(potentials) == 0:
            raise Exception('No potentials given')

        elif len(potentials) == 1:
            raise Exception('Only one set of  potentials given, go to VSCF_diag() class')
        elif len(potentials) > 2:
            print 'More than two sets potentials given. Only the two first will be used'

        self.v1 = potentials[0].pot
        self.v2 = potentials[1].pot
        self.wfns = wavefunctions.wfns.copy()  # initial wave functions

        self.dm1 = np.array([])
        self.dm2 = np.array([])

        if len(self.v2.shape) < 4:
            raise Exception('The second set should consist of two-dimensional potentials')
        if (self.nmodes != self.v1.shape[0]) or (self.ngrid != self.v1.shape[1]) \
           or (self.nmodes != self.v2.shape[0]) or (self.ngrid != self.v2.shape[2]):
            raise Exception('Potential and grid size mismatch')
        
        self.states = [[0]*self.nmodes]   # list of states for which the VSCF is solved, at first only gs considered
        self.energies = np.zeros(len(self.states))
        self.vscf_wfns = np.zeros((len(self.states), self.nmodes, self.nstates, self.ngrid))  # all vscf_wfns

    def calculate_intensities(self, *dipolemoments):

        if len(dipolemoments) == 0:
            raise Exception('No dipole moments given')

        elif len(dipolemoments) == 1:
            raise Exception('Only one set of dipole moments given, go to VSCF_diag class')
        elif len(dipolemoments) > 2:
            print 'More than two sets of dipole moments given, only the two first will be used'

        if not self.solved:
            raise Exception('Solve the VSCF first')

        self.dm1 = dipolemoments[0]
        self.dm2 = dipolemoments[1]

        # assuming that the first state is a ground state
        gs = self.states[0]
        print fancy_box('VSCF Intensities')
        print 'State        Energy /cm^-1    Intensity /km*mol^-1'
        for s in self.states[1:]:
            tmptm = np.array([0.0, 0.0, 0.0])
            stateindex = self.states.index(s)
            for i in range(self.nmodes):
                tmpd1 = np.array([0.0, 0.0, 0.0])
                tmpovrlp = 1.0
                for j in range(self.nmodes):
                    if j == i:
                        #calculate <psi|i|psi>
                        tmpd1[0] += (self.dx[i]*self.vscf_wfns[0, i, gs[i]]*self.vscf_wfns[stateindex, i, s[i]] *
                                     self.dm1[i, :, 0]).sum()
                        tmpd1[1] += (self.dx[i]*self.vscf_wfns[0, i, gs[i]]*self.vscf_wfns[stateindex, i, s[i]] *
                                     self.dm1[i, :, 1]).sum()
                        tmpd1[2] += (self.dx[i]*self.vscf_wfns[0, i, gs[i]]*self.vscf_wfns[stateindex, i, s[i]] *
                                     self.dm1[i, :, 2]).sum()

                    else:
                        if s[j] == gs[j]:
                            tmpovrlp *= (self.dx[j]*self.vscf_wfns[0, j, gs[j]] *
                                         self.vscf_wfns[stateindex, j, s[j]]).sum()
                            #tmpovrlp *= 1.0
                        else:
                            tmpovrlp = 0.0

                tmptm += tmpd1 * tmpovrlp
                #tmptm = tmptm + tmpd1
            for i in range(self.nmodes):
                tmpd2 = np.array([0.0, 0.0, 0.0])
                for j in range(i+1, self.nmodes):

                    for k in range(self.ngrid):
                        for l in range(self.ngrid):
                            tmpd2[0] += self.dx[i] * self.dx[j] * self.dm2[i, j, k, l, 0] \
                                * self.vscf_wfns[0, i, gs[i], k] \
                                * self.vscf_wfns[0, j, gs[j], l] \
                                * self.vscf_wfns[stateindex, i, s[i], k] \
                                * self.vscf_wfns[stateindex, j, s[j], l]
                            tmpd2[1] += self.dx[i] * self.dx[j] * self.dm2[i, j, k, l, 1] \
                                * self.vscf_wfns[0, i, gs[i], k] \
                                * self.vscf_wfns[0, j, gs[j], l] \
                                * self.vscf_wfns[stateindex, i, s[i], k] \
                                * self.vscf_wfns[stateindex, j, s[j], l]
                            tmpd2[2] += self.dx[i] * self.dx[j] * self.dm2[i, j, k, l, 2] \
                                * self.vscf_wfns[0, i, gs[i], k] \
                                * self.vscf_wfns[0, j, gs[j], l] \
                                * self.vscf_wfns[stateindex, i, s[i], k] \
                                * self.vscf_wfns[stateindex, j, s[j], l]
                    tmpovrlp = 1.0
                    for m in range(self.nmodes):
                        if m != i and m != j:
                            if s[m] == gs[m]:
                                tmpovrlp *= (self.dx[m]*self.vscf_wfns[0, m, gs[m]]
                                             * self.vscf_wfns[stateindex, m, s[m]]).sum()

                            else:
                                tmpovrlp = 0.0

                    tmptm += tmpd2 * tmpovrlp
                    #tmptm = tmptm + tmpd2
            factor = 2.5048
            intens = (tmptm[0]**2 + tmptm[1]**2 + tmptm[2]**2)*factor*(self.energies[stateindex]-self.energies[0])
            print '%s %7.1f %7.1f' % (s, self.energies[stateindex]-self.energies[0], intens)

    def get_groundstate_wfn(self):

        if self.states[0] == [0]*self.nmodes and self.solved:
            tmpwfn = Wavefunction(self.grid_object)
            tmpwfn.wfns = self.vscf_wfns[0]
            return tmpwfn
        else:
            raise Exception('Ground state not solved')

    def save_wave_functions(self, fname='2D_wavefunctions'):

        from time import strftime
        fname = fname + '_' + strftime('%Y%m%d%H%M') + '.npy'
        fname2 = 'States_' + fname
        np.save(fname, self.vscf_wfns)
        np.save(fname2, np.array(self.states))  # list of states in wave_function file

    def solve_singles(self):
        
        gs = [0]*self.nmodes
        states = [gs]
        
        for i in range(self.nmodes):
            vec = [0]*self.nmodes
            vec[i] = 1
            states.append(vec)

        self.solve(*states)

    def solve(self, *states):

        if len(states) == 0:
            states = [[0]*self.nmodes]  # if no states defined, only gs considered

        if self.solved and self.states == list(states):
            print 'Already solved, nothing to do. See results with print_results() method'

        else:
            print ''
            print fancy_box('Solving VSCF')
            self.states = list(states)  # use new states and do the VSCF
            self.energies = np.zeros(len(self.states))
            self.vscf_wfns = np.zeros((len(self.states), self.nmodes, self.nstates, self.ngrid))  # vscf opt wfns
            for i, s in enumerate(self.states):
                print fancy_box('Solving State: '+str(s))         
                (self.energies[i], self.vscf_wfns[i]) = self._solve_state(s)

            self.solved = True
            print ''
            print fancy_box('VSCF Done')
            print ''
            self.print_results()

    def print_results(self):

        if self.solved:
            print ''
            print fancy_box('VSCF Results')
            print 'States       Energy / cm^-1'
            print '---------------------------'
            for i, s in enumerate(self.states):
                print s, '%.1f' % self.energies[i]
            print ''
            print 'Assuming ', self.states[0], ' as the reference state'
            print '--------------------------------------------------'
            print 'Final states  Energy / cm^-1' 
            for i, s in enumerate(self.states):
                if i != 0:
                    print '-> ', s, '%.1f' % (self.energies[i]-self.energies[0])
        else:
            print 'VSCF not solved yet. Use solve() method first'

    def _solve_state(self, state):

        # solving 2D VSCF for a given state
        maxiter = 100
        eps = 1e-6
        etot = 0.0
        wfn = self.wfns.copy()
        tmpwfn = np.zeros(wfn.shape)
        eprev = 0.0
        for niter in range(maxiter):
            etot = 0.0
            print 'Iteration: %i ' % (niter+1)
            print 'Mode State   Eigv'
            for i in range(self.nmodes):
                diagpot = self.v1[i]
                # now get effective potential
                effpot = self._veffective(i, state, wfn)
                totalpot = diagpot+effpot
                
                # solve 1-mode problem
                (energies, wavefunctions) = self._collocation(self.grids[i], totalpot)
                tmpwfn[i] = wavefunctions
                # add energy
                etot += energies[state[i]]   # add optimized state-energy
                print '%4i %5i %8.1f' % (i, state[i], energies[state[i]]/VibTools.Constants.cm_in_au)

            #calculate correction
            emp1 = self._scfcorr(state, tmpwfn)

            print 'Sum of eigenvalues %.1f, SCF correction %.1f, total energy %.1f / cm^-1' \
                % (etot/VibTools.Constants.cm_in_au,
                  emp1/VibTools.Constants.cm_in_au,
                  (etot - emp1)/VibTools.Constants.cm_in_au)
            etot -= emp1
            print ''
            if abs(etot-eprev) < eps:
                break
            else:
                eprev = etot
                wfn = np.copy(tmpwfn)

            # get delta E

        return etot / VibTools.Constants.cm_in_au, wfn.copy()

    def _scfcorr(self, state, wfn):
        
        scfcorr = 0.0
        for i in range(self.nmodes):
            for j in range(i+1, self.nmodes):
                for gi in range(self.ngrid):
                    for gj in range(self.ngrid):

                        scfcorr += self.dx[i] * self.dx[j] * self.v2[i, j, gi, gj] * wfn[i, state[i], gi] ** 2 \
                            * wfn[j, state[j], gj]**2

        return scfcorr

    def _veffective(self, mode, state, wfn):

        veff = np.zeros(self.ngrid)
            
        for i in range(self.ngrid):
            for j in range(self.nmodes):
                if j != mode:
                    veff[i] += (wfn[j, state[j]]**2 * self.dx[j] * self.v2[mode, j, i, :]).sum()

        return veff
                    

class VCI:

    def __init__(self, grids, wavefunctions, *potentials):

        # initialize with object corresponding to grids, potentials and wavefunctions
        self.grids = grids.grids
        self.wfns = wavefunctions.wfns  # these are VSCF optimized wave functions
        
        self.nmodes = grids.nmodes
        self.ngrid = grids.ngrid

        self.states = []
        self.solved = False

        self.dx = [x[1]-x[0] for x in self.grids]   
        self.a = [(0.7**2.0)/((x[1]-x[0])**2.0) for x in self.grids]

        self.coefficients = np.zeros((self.nmodes, self.ngrid, self.ngrid))  # TODO number of VSCF states in wfn
        self._calculate_coeff()

        self.sij = np.zeros((self.nmodes, self.ngrid, self.ngrid))  # caculate Sij only once
        self._calculate_ovrlp_integrals()

        self.tij = np.zeros((self.nmodes, self.ngrid, self.ngrid))  # calculate Tij only once as well
        self._calculate_kinetic_integrals()

        self.energies = np.array([])
        self.energiesrcm = np.array([])
        self.vectors = np.array([])

        if len(potentials) != 2:
            raise Exception('Only two potentials, V1 and V2, accepted, so far')
        else:
            self.v1 = potentials[0].pot
            self.v2 = potentials[1].pot

        self.dm1 = np.array([])
        self.dm2 = np.array([])

        # TODO
        # 1. Check shapes of data stored in objects

    def solve(self):
        import itertools

        if len(self.states) == 0:
            print fancy_box('No CI space defined, by default singles space will be used')
            self.generate_states()

        # generate combination of states
        comb = [x for x in itertools.combinations_with_replacement(self.states, 2)]

        ncomb = len(comb)

        print fancy_box('Number of combinations : '+str(ncomb))

        # hamiltonian

        hamiltonian = np.zeros((len(self.states), len(self.states)))
        counter = 0

        for i in range(ncomb):
            counter += 1
            
            tmp = 0.0  # matrix element

            n = comb[i][0]  # left state
            m = comb[i][1]  # right state
            print 'Working on configruation %i out of %i' % (i+1, ncomb)
            print ' < %s | H | %s >' % (str(n), str(m))
            
            # 1-mode integrals
            for j in range(self.nmodes):
                tmpovrlp = 1.0
                tmpv1 = 0.0
                tmpt = 0.0

                for k in range(self.nmodes):

                    if k == j:
                        tmpv1 = self._v1_integral(j, n[j], m[j])
                        tmpt = self._kinetic_integral(j, n[j], m[j])

                    else:
                        if n[k] == m[k]:
                            s = self._ovrlp_integral(k, n[k], n[k])
                            tmpovrlp *= s

                        else:
                            tmpovrlp = 0.0

                tmpv1 *= tmpovrlp
                tmpt *= tmpovrlp

                tmp += tmpv1 + tmpt
            # 2-mode integrals
            for j in range(self.nmodes):

                for k in range(j+1, self.nmodes):
                    tmpovrlp = 1.0
                    tmpv2 = self._v2_integral(j, k, n[j], n[k], m[j], m[k])

                    for l in range(self.nmodes):

                        if l != j and l != k:
                            if n[l] == m[l]:
                            
                                tmpovrlp *= self._ovrlp_integral(l, n[l], n[l])

                            else:
                                tmpovrlp = 0.0

                    tmpv2 *= tmpovrlp

                    tmp += tmpv2
            nind = self.states.index(n)  # find the left state in the states vector
            mind = self.states.index(m)  # fin the right state
            hamiltonian[nind, mind] = tmp

            print 'Step %i/%i done, value %f stored' % (counter, ncomb, tmp)

        print fancy_box('Hamiltonian matrix constructed. Diagonalization...')
        w, v = np.linalg.eigh(hamiltonian, UPLO='U')

        self.energies = w
        self.vectors = v
        wcm = w / VibTools.Constants.cm_in_au
        self.energiesrcm = wcm  # energies in reciprocal cm
        print 'State %15s %15s' % ('E /cm^-1', 'DE /cm^-1')
        for i in range(len(self.states)):
            print "%s %10.4f %10.4f" % (self.states[i], wcm[i], wcm[i]-wcm[0])

        self.solved = True

    def print_states(self):
        print ''
        print fancy_box('CI Space')
        print self.states

    def generate_states(self, maxexc=1):
        # maxexc - maximal excitation, 1-Singles, 2-Doubles etc.
        if maxexc > 4:
            raise Exception('At most quadruple excitations supported')

        states = []
        gs = [0] * self.nmodes  
        states.append(gs)

        for i in range(self.nmodes):

            vec = [0] * self.nmodes
            vec[i] = 1
            states.append(vec)

            if maxexc > 1:
                vec = [0] * self.nmodes
                vec[i] = 2
                states.append(vec)

                for j in range(i+1, self.nmodes):
                    vec = [0] * self.nmodes
                    vec[i] = 1
                    vec[j] = 1
                    states.append(vec)

        if maxexc > 2:

            for i in range(self.nmodes):
                vec = [0]*self.nmodes
                vec[i] = 3
                states.append(vec)

                for j in range(i+1, self.nmodes):
                    vec = [0]*self.nmodes
                    vec[i] = 2
                    vec[j] = 1
                    states.append(vec)
                    vec = [0]*self.nmodes
                    vec[i] = 1
                    vec[j] = 2
                    states.append(vec)

                    for k in range(j+1, self.nmodes):
                        vec = [0]*self.nmodes
                        vec[i] = 1
                        vec[j] = 1
                        vec[k] = 1
                        states.append(vec)
        if maxexc > 3:

            for i in range(self.nmodes):
                vec = [0] * self.nmodes
                vec[i] = 4
                states.append(vec)

                for j in range(i+1, self.nmodes):
                    vec = [0]*self.nmodes
                    vec[i] = 3
                    vec[j] = 1
                    states.append(vec)
                    vec = [0]*self.nmodes
                    vec[i] = 1
                    vec[j] = 3
                    states.append(vec)
                    vec = [0]*self.nmodes
                    vec[i] = 2
                    vec[j] = 2
                    states.append(vec)
                    
                    for k in range(j+1, self.nmodes):
                        vec = [0]*self.nmodes
                        vec[i] = 1
                        vec[j] = 1
                        vec[k] = 2
                        states.append(vec)
                        vec = [0]*self.nmodes
                        vec[i] = 1
                        vec[j] = 2
                        vec[k] = 1
                        states.append(vec)
                        vec = [0]*self.nmodes
                        vec[i] = 2
                        vec[j] = 1
                        vec[k] = 1
                        states.append(vec)

                        for l in range(k+1, self.nmodes):
                            vec = [0]*self.nmodes
                            vec[i] = 1
                            vec[j] = 1
                            vec[k] = 1
                            vec[l] = 1
                            states.append(vec)

        self.states = states

    def calculate_intensities(self, *dipolemoments):

        if len(dipolemoments) == 0:
            raise Exception('No dipole moments given')

        elif len(dipolemoments) == 1:
            raise Exception('Only one set of dipole moments given, go to VSCF_diag class')
        elif len(dipolemoments) > 2:
            print 'More than two sets of dipole moments given, only the two first will be used'

        if not self.solved:
            raise Exception('Solve the VCI first')

        self.dm1 = dipolemoments[0]
        self.dm2 = dipolemoments[1]

        # assuming that the first state is a ground state

        for i in range(1, len(self.states)):
            totaltm = 0.0
            for istate in range(len(self.states)):
                ci = self.vectors[istate, 0]  # initial state's coefficient

                for fstate in range(len(self.states)):
                    cf = self.vectors[fstate, i]  # final state's coefficient

                    tmptm = np.array([0.0, 0.0, 0.0])

                    for j in range(self.nmodes):
                        tmpd1 = np.array([0.0, 0.0, 0.0])
                        tmpovrlp = 1.0
                        jistate = self.states[istate][j]
                        jfstate = self.states[fstate][j]

                        for k in range(self.nmodes):
                            kistate = self.states[istate][k]
                            kfstate = self.states[fstate][k]

                            if k == j:
                                #  calculate <psi|u|psi>
                                tmpd1[0] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                             * self.dm1[j, :, 0]).sum()
                                tmpd1[1] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                             * self.dm1[j, :, 1]).sum()
                                tmpd1[2] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                             * self.dm1[j, :, 2]).sum()

                            else:
                                if self.states[istate][k] == self.states[fstate][k]:
                                    tmpovrlp *= (self.dx[k] * self.wfns[k, kistate] * self.wfns[k, kfstate]).sum()
                                else:
                                    tmpovrlp = 0.0

                        tmptm += tmpd1 * tmpovrlp

                    for j in range(self.nmodes):
                        jistate = self.states[istate][j]
                        jfstate = self.states[fstate][j]
                        for k in range(j+1, self.nmodes):
                            tmpd2 = np.array([0.0, 0.0, 0.0])
                            kistate = self.states[istate][k]
                            kfstate = self.states[fstate][k]

                            for l in range(self.ngrid):
                                for m in range(self.ngrid):
                                    tmpd2[0] += self.dx[j] * self.dx[k] * self.dm2[j, k, l, m,0] \
                                        * self.wfns[j, jistate, l] * self.wfns[j, jfstate, l] \
                                        * self.wfns[k, kistate, m] * self.wfns[k, kfstate, m]
                                    tmpd2[1] += self.dx[j] * self.dx[k] * self.dm2[j, k, l, m,1] \
                                        * self.wfns[j, jistate, l] * self.wfns[j, jfstate, l] \
                                        * self.wfns[k, kistate, m] * self.wfns[k, kfstate, m]
                                    tmpd2[2] += self.dx[j] * self.dx[k] * self.dm2[j, k, l, m,2] \
                                        * self.wfns[j, jistate, l] * self.wfns[j, jfstate, l] \
                                        * self.wfns[k, kistate, m] * self.wfns[k, kfstate, m]
                            tmpovrlp = 1.0

                            for n in range(self.nmodes):
                                if n != j and n != k:
                                    nistate = self.states[istate][n]
                                    nfstate = self.states[fstate][n]

                                    if nistate == nfstate:
                                        tmpovrlp *= (self.dx[n] * self.wfns[n, nistate] * self.wfns[n, nfstate]).sum()

                                    else:
                                        tmpovrlp = 0.0

                            tmptm += tmpd2 * tmpovrlp

                    totaltm += tmptm * ci * cf

            factor = 2.5048
            intens = (totaltm[0]**2 + totaltm[1]**2 + totaltm[2]**2) * factor * (self.energiesrcm[i]
                                                                                 - self.energiesrcm[0])
            print '%7.1f %7.1f' % (self.energiesrcm[i] - self.energiesrcm[0], intens)

    def _v1_integral(self, mode, lstate, rstate):  # calculate integral of type < mode(lstate) | V1 | mode(rstate) >

        s = (self.dx[mode] * self.wfns[mode, lstate] * self.wfns[mode, rstate] * self.v1[mode]).sum()

        return s

    def _v2_integral(self, mode1, mode2, lstate1, lstate2, rstate1, rstate2):
     # < mode1(lstate1) mode2(lstate2) | V2 | mode1(rstate1),mode2(rstate2)>

        s = 0.0

        for i in range(self.ngrid):
            si = self.dx[mode1] * self.wfns[mode1, lstate1, i] * self.wfns[mode1, rstate1, i]

            for j in range(self.ngrid):

                sj = self.dx[mode2] * self.wfns[mode2, lstate2, j] * self.wfns[mode2, rstate2, j]
                s += si * sj * self.v2[mode1, mode2, i, j]

        return s

    def _ovrlp_integral(self, mode, lstate, rstate):    # overlap integral < mode(lstates) | mode(rstate) >

        s = (self.dx[mode] * self.wfns[mode, lstate] * self.wfns[mode, rstate] * 1.0).sum()

        return s
       
    def _kinetic_integral(self, mode, lstate, rstate):  # kinetic energy integral < mode(lstate) | T | mode(rstate) >

        t = 0.0

        for i in range(self.ngrid):
            for j in range(self.ngrid):
                t += self.coefficients[mode, lstate, i] * self.coefficients[mode, rstate, j] * self.tij[mode, i, j]

        return t

    def _dgs_kinetic_integral(self, mode, i, j):  # integral between two DGs of the same modal centered at qi and qj

        a = self.a[mode]
        qi = self.grids[mode, i]
        qj = self.grids[mode, j]
        bij = (a + a)**0.5
        cij = (a*a)/(bij**2.0) * ((qi-qj)**2.0)
        ovrlp = self.sij[mode, i, j]

        return ovrlp * (a*a/(bij**2.0)) * (1.0-2.0*cij)

    def _dgs_ovrlp_integral(self, mode, i, j):  # overlap integral between two DGs of the same modal

        a = self.a[mode]
        qi = self.grids[mode, i]
        qj = self.grids[mode, j]
        aij = (4.0*a*a/(np.pi**2.0))**0.25
        bij = (a + a)**0.5
        cij = (a*a)/(bij**2.0)*((qi-qj)**2.0)
        return np.sqrt(np.pi)*(aij/bij)*np.exp(-cij)

    @staticmethod
    def _chi(q, a, qi):  # definition of a basis set function

        return ((2.0*a)/np.pi)**0.25*np.exp(-a*(q-qi)**2.0)

    def _calculate_coeff(self):  # calculates basis set coefficients, using grid and wave function values
        for i in range(self.nmodes):  # for each mode

            for j in range(self.ngrid):  # for each state

                #now calculate coefficients
                chi = np.zeros((self.ngrid, self.ngrid))
                for k in range(self.ngrid):
                    for l in range(self.ngrid):
                        chi[k, l] = self._chi(self.grids[i, l], self.a[i], self.grids[i, k])

                c = np.linalg.solve(chi, self.wfns[i, j])
                self.coefficients[i, j] = np.copy(c)

    def _calculate_kinetic_integrals(self):

        for i in range(self.nmodes):  # for each mode

            for j in range(self.ngrid):
                for k in range(self.ngrid):

                    self.tij[i, j, k] = self._dgs_kinetic_integral(i, j, k)

    def _calculate_ovrlp_integrals(self):

        for i in range(self.nmodes):  # for each mode

            for j in range(self.ngrid):
                for k in range(self.ngrid):

                    self.sij[i, j, k] = self._dgs_ovrlp_integral(i, j, k)
