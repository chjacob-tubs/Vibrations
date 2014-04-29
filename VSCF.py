import numpy as np
import Misc

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
                print 'Mode %i, eigv: %f' % (i, (self.eigv[i, 1]-self.eigv[i, 0])/Misc.cm_in_au)

            print 'Eigenvalues: '
            for i in range(self.nmodes):
                print 'Mode %i, eigv: %f' % (i, self.eigv[i, 0]/Misc.cm_in_au)
        else:
            print 'VSCF not solved yet. Use solve() method first'

    def print_eigenvalues(self):

        for i in range(self.nmodes):
            print 'Mode %i' % i

            for j in range(self.nstates):
                print self.eigv[i, j], self.eigv[i, j]/Misc.cm_in_au

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
        print Misc.fancy_box('VSCF Intensities')
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
            print Misc.fancy_box('Solving VSCF')
            self.states = list(states)  # use new states and do the VSCF
            self.energies = np.zeros(len(self.states))
            self.vscf_wfns = np.zeros((len(self.states), self.nmodes, self.nstates, self.ngrid))  # vscf opt wfns
            for i, s in enumerate(self.states):
                print Misc.fancy_box('Solving State: '+str(s))         
                (self.energies[i], self.vscf_wfns[i]) = self._solve_state(s)

            self.solved = True
            print ''
            print Misc.fancy_box('VSCF Done')
            print ''
            self.print_results()

    def print_results(self):

        if self.solved:
            print ''
            print Misc.fancy_box('VSCF Results')
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
                print '%4i %5i %8.1f' % (i, state[i], energies[state[i]]/Misc.cm_in_au)

            #calculate correction
            emp1 = self._scfcorr(state, tmpwfn)

            print 'Sum of eigenvalues %.1f, SCF correction %.1f, total energy %.1f / cm^-1' \
                % (etot/Misc.cm_in_au,
                  emp1/Misc.cm_in_au,
                  (etot - emp1)/Misc.cm_in_au)
            etot -= emp1
            print ''
            if abs(etot-eprev) < eps:
                break
            else:
                eprev = etot
                wfn = np.copy(tmpwfn)

            # get delta E

        return etot / Misc.cm_in_au, wfn.copy()

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
                    


