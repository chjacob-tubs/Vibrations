"""
Module containing all Vibrational SCF classes
G{classtree VSCF}
"""
import numpy as np
import Misc
import Wavefunctions


class VSCF:
    """
    The mother class for VSCF.
    G{classtree}
    """

    def __init__(self, *potentials):  # always initialized with (some) potentials, grids are stored in the potentials
        """
        The class must be initialized with grids and potentials

        @param potentials: The potentials
        @type potentials: Vibrations/Potential
        """
        if len(potentials) > 0:
            self.nmodes = potentials[0].grids.nmodes
            self.ngrid = potentials[0].ngrid
            self.nstates = self.ngrid
            self.wavefunction = Wavefunctions.Wavefunction(potentials[0].grids)
            self.eigv = np.zeros((self.nmodes, self.nstates))
            self.grids = potentials[0].grids
            self.dx = [x[1]-x[0] for x in self.grids.grids]  # integration step
            self.solved = False  # simple switch to check, whether VSCF was already solved
        else:
            raise Exception("No potential given.")


    def _collocation(self, grid, potential):
        """
        The collocation method, see Chem. Phys. Lett., 153(1988), 98. for details.
        """
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

        #(eigval, eigvec) = np.linalg.eigh(hamiltonian,UPLO='U')
        (eigval, eigvec) = np.linalg.eig(hamiltonian)
        eigval = np.real(eigval)
        eigvec = np.real(eigvec)

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
        """
        Returns the wave functions as numpy.array
        """

        return self.wavefunction.wfns

    def get_wave_function_object(self):
        """
        Returns the wave functions as Vibrations/Wavefunction object
        """
        return self.wavefunction

    def save_wave_functions(self, fname='wavefunctions'):
        """
        Saves the wave functions to a NumPy formatted binary file *.npy

        @param fname: File name, without extension, time stamp is added
        @type fname: String
        """
        from time import strftime
        fname = fname + '_' + strftime('%Y%m%d%H%M') + '.npy'
        np.save(fname, self.wavefunction.wfns)


class VSCFDiag(VSCF):
    """
    The class for the diagonal VSCF -- using only 1-mode potentials
    """

    def __init__(self,*potentials):

        
        if len(potentials) == 0:
            raise Exception('No potential given')

        elif len(potentials) > 1:
            print 'More than one potentials given, only the first will be used'

        VSCF.__init__(self, potentials[0])  # fist call the constructor of mother class
        self.v1 = potentials[0]

        self.solved = False

    def solve(self):
        """
        Solves the diagonal VSCF
        """
        if self.solved:
            print 'Already solved, nothing to do. See results with print_results() method'

        else:

            for i in range(self.nmodes):  # go over each mode
                #print self.grids[i]
                #print self.v1.data[i]
                v1ind = self.v1.indices.index(i)  #  find the index of the mode i in the  potential
                # TODO take into account that the mode can be not present in the potential, use try etc.
                (tmpeigv, tmpwfn) = self._collocation(self.grids.grids[i], self.v1.data[v1ind])
                self.eigv[i] = tmpeigv
                self.wavefunction.wfns[i] = tmpwfn
            
            self.solved = True

    def print_results(self):
        """
        Prints the results
        """
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
        """
        Prints the eigenvalues
        """
        for i in range(self.nmodes):
            print 'Mode %i' % i

            for j in range(self.nstates):
                print self.eigv[i, j], self.eigv[i, j]/Misc.cm_in_au

    def save_wave_functions(self, fname='1D_wavefunctions'):
        """
        Saves the wave function to a NumPy formatted binary file *.npy

        @param fname: File name, without extension, time stamp added
        @type fname: String
        """
        from time import strftime
        fname = fname + '_' + strftime('%Y%m%d%H%M') + '.npy'
        np.save(fname, self.wavefunction.wfns)


class VSCF2D(VSCF):
    """
    The class for the 2-dimensional VSCF -- containing the mean-field potential for modes coupling
    """
    def __init__(self, *potentials):


        
        if len(potentials) == 0:
            raise Exception('No potentials given')

        elif len(potentials) == 1:
            raise Exception('Only one set of  potentials given, go to VSCF_diag() class')
        elif len(potentials) > 2:
            print 'More than two sets potentials given. Only the two first will be used'

        VSCF.__init__(self, potentials[0])
        import copy
        self.v1 = copy.copy(potentials[0])
        self.v2 = copy.copy(potentials[1])
        self.grids = self.v1.grids
        self.dm1 = np.array([])
        self.dm2 = np.array([])

        #if len(self.v2.shape) < 4:
        #    raise Exception('The second set should consist of two-dimensional potentials')
        #if (self.nmodes != self.v1.shape[0]) or (self.ngrid != self.v1.shape[1]) \
        #   or (self.nmodes != self.v2.shape[0]) or (self.ngrid != self.v2.shape[2]):
        #    raise Exception('Potential and grid size mismatch')
        
        self.states = [[0]*self.nmodes]   # list of states for which the VSCF is solved, at first only gs considered
        self.energies = []
        self.vscf_wavefunctions = []   # list of Wavefunction objects (for each state)

    def calculate_intensities(self, *dipolemoments):
        """
        Calculates VSCF intensities with dipole moment surfaces

        @param dipolemoments: dipole moment surfaces
        @type dipolemoments: numpy.array
        """


        if not self.solved:
            raise Exception('Solve the VSCF first')

        if len(dipolemoments) == 0:
            raise Exception('No dipole moments given.')
        elif len(dipolemoments) == 1:
            print 'Only one dipole moment surface given, the 2D counterpart will be set to 0.'
        elif len(dipolemoments) > 2:
            print 'More than two sets of dipole moments given, only the two first will be used.'

#       if dipolemoments[0].order == 1:
#           self.dm1 = dipolemoments[0].dm
#       else:
#           raise Exception('The 1-D DMS should be given as the first one.')
#       if len(dipolemoments) > 1 and dipolemoments[1].order == 2:
#           self.dm2 = dipolemoments[1].dm
#       elif len(dipolemoments) == 1:
#           self.dm2 = np.zeros((self.nmodes, self.nmodes, self.ngrid, self.ngrid, 3))
#       else:
#           raise Exception('The order of the second dipole moment surface does not match.')

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
                        ind = self.dm1.indices.index(j)
                        #calculate <psi|i|psi>
                        tmpd1[0] += (self.dx[i]*self.vscf_wavefunctions[0].wfns[i, gs[i]]*self.vscf_wavefunctions[stateindex].wfns[i, s[i]] *
                                     self.dm1.data[ind][:, 0]).sum()
                        tmpd1[1] += (self.dx[i]*self.vscf_wavefunctions[0].wfns[i, gs[i]]*self.vscf_wavefunctions[stateindex].wfns[i, s[i]] *
                                     self.dm1.data[ind][:, 1]).sum()
                        tmpd1[2] += (self.dx[i]*self.vscf_wavefunctions[0].wfns[i, gs[i]]*self.vscf_wavefunctions[stateindex].wfns[i, s[i]] *
                                     self.dm1.data[ind][:, 2]).sum()

                    else:
                        if s[j] == gs[j]:
                            tmpovrlp *= (self.dx[j]*self.vscf_wavefunctions[0].wfns[j, gs[j]] *
                                         self.vscf_wavefunctions[stateindex].wfns[j, s[j]]).sum()
                            #tmpovrlp *= 1.0
                        else:
                            tmpovrlp = 0.0

                tmptm += tmpd1 * tmpovrlp
                #tmptm = tmptm + tmpd1
            for i in range(self.nmodes):
                for j in range(i+1, self.nmodes):
                    tmpd2 = np.array([0.0, 0.0, 0.0])

                    for k in range(self.ngrid):
                        ind = self.dm2.indices.index((i,j))
                        for l in range(self.ngrid):
                            tmpd2[0] += self.dx[i] * self.dx[j] * self.dm2.data[ind][k, l, 0] \
                                * self.vscf_wavefunctions[0].wfns[i, gs[i], k] \
                                * self.vscf_wavefunctions[0].wfns[j, gs[j], l] \
                                * self.vscf_wavefunctions[stateindex].wfns[i, s[i], k] \
                                * self.vscf_wavefunctions[stateindex].wfns[j, s[j], l]
                            tmpd2[1] += self.dx[i] * self.dx[j] * self.dm2.data[ind][k, l, 1] \
                                * self.vscf_wavefunctions[0].wfns[i, gs[i], k] \
                                * self.vscf_wavefunctions[0].wfns[j, gs[j], l] \
                                * self.vscf_wavefunctions[stateindex].wfns[i, s[i], k] \
                                * self.vscf_wavefunctions[stateindex].wfns[j, s[j], l]
                            tmpd2[2] += self.dx[i] * self.dx[j] * self.dm2.data[ind][k, l, 2] \
                                * self.vscf_wavefunctions[0].wfns[i, gs[i], k] \
                                * self.vscf_wavefunctions[0].wfns[j, gs[j], l] \
                                * self.vscf_wavefunctions[stateindex].wfns[i, s[i], k] \
                                * self.vscf_wavefunctions[stateindex].wfns[j, s[j], l]
                    tmpovrlp = 1.0
                    for m in range(self.nmodes):
                        if m != i and m != j:
                            if s[m] == gs[m]:
                                tmpovrlp *= (self.dx[m]*self.vscf_wavefunctions[0].wfns[m, gs[m]]
                                             * self.vscf_wavefunctions[stateindex].wfns[ m, s[m]]).sum()

                            else:
                                tmpovrlp = 0.0

                    tmptm += tmpd2 * tmpovrlp
                    #tmptm = tmptm + tmpd2
            factor = 2.5048
            intens = (tmptm[0]**2 + tmptm[1]**2 + tmptm[2]**2)*factor*(self.energies[stateindex]-self.energies[0])
            print '%s %7.1f %7.1f' % (s, self.energies[stateindex]-self.energies[0], intens)

    def get_groundstate_wfn(self):
        """
        Returns the ground state wave function, which can be used for VCI calculations
        """
        if self.states[0] == [0]*self.nmodes and self.solved:
            return self.vscf_wavefunctions[0]
        else:
            raise Exception('Ground state not solved')

    def save_wave_functions(self, fname='2D_wavefunctions'):
        """
        Saves the wave functions to a NumPy formatted binary file *.npy

        @param fname: File name
        @type fname: String
        """
        from time import strftime
        fname = fname + '_' + strftime('%Y%m%d%H%M') + '.npy'
        fname2 = 'States_' + fname
        np.save(fname, self.vscf_wfns)
        np.save(fname2, np.array(self.states))  # list of states in wave_function file

    def solve_singles(self):
        """
        Solves the VSCF for the ground state and all singly-excited states
        """
        gs = [0]*self.nmodes
        states = [gs]
        
        for i in range(self.nmodes):
            vec = [0]*self.nmodes
            vec[i] = 1
            states.append(vec)

        self.solve(*states)

    def solve(self, *states):
        """
        Solves the VSCF for given states

        @param *states: considered states, the first given is assumed to be the ground state
        @type *states: List of Integer
        """
        if len(states) == 0:
            states = [[0]*self.nmodes]  # if no states defined, only gs considered

        if self.solved and self.states == list(states):
            print 'Already solved, nothing to do. See results with print_results() method'

        else:
            print ''
            print Misc.fancy_box('Solving VSCF')
            self.states = list(states)  # use new states and do the VSCF
            self.energies = []
            self.vscf_wavefunctions = []
            for i, s in enumerate(self.states):
                print Misc.fancy_box('Solving State: '+str(s))         
                (energy, wfn) = self._solve_state(s)
                self.energies.append(energy)
                wfn_obj = Wavefunctions.Wavefunction(self.v1.grids)
                wfn_obj.wfns = wfn
                self.vscf_wavefunctions.append(wfn_obj)

            self.solved = True
            print ''
            print Misc.fancy_box('VSCF Done')
            print ''
            self.print_results()

    def print_results(self):
        """
        Prints the results
        """
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
        actualwfns = np.zeros((self.nmodes,self.ngrid,self.ngrid))
        tmpwfns = np.zeros((self.nmodes,self.ngrid,self.ngrid))

        # first generate a diagonal wave function as a reference
        for i in range(self.nmodes):
            (modeen, modewfn) = self._collocation(self.grids.grids[i], self.v1.data[self.v1.indices.index(i)])
            actualwfns[i] = modewfn


        eprev = 0.0
        for niter in range(maxiter):
            etot = 0.0
            print 'Iteration: %i ' % (niter+1)
            print 'Mode State   Eigv'
            for i in range(self.nmodes):
                diagpot = self.v1.data[self.v1.indices.index(i)]
                # now get effective potential
                effpot = self._veffective(i, state, actualwfns)
                totalpot = diagpot+effpot
                # solve 1-mode problem
                (energies, wavefunction) = self._collocation(self.grids.grids[i], totalpot)
                tmpwfns[i] = wavefunction
                # add energy
                etot += energies[state[i]]   # add optimized state-energy
                print '%4i %5i %8.1f' % (i, state[i], energies[state[i]]/Misc.cm_in_au)

            #calculate correction
            actualwfns = tmpwfns
            emp1 = self._scfcorr(state, actualwfns)
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

            # get delta E


        return etot / Misc.cm_in_au, actualwfns.copy()

    def _scfcorr(self, state, wfns):
        
        scfcorr = 0.0
        for i in range(self.nmodes):
            for j in range(i+1, self.nmodes):
                if (i,j) in self.v2.indices or (j,i) in self.v2.indices: #first check if there's the pot for two modes
                    try:
                        ind = self.v2.indices.index((i,j))
                    except:
                        ind = selv.v2.indices.index((j,i))
                    for gi in range(self.ngrid):
                        for gj in range(self.ngrid):

                            scfcorr = scfcorr +  self.dx[i] * self.dx[j] * self.v2.data[ind][gi, gj] * wfns[i, state[i], gi] ** 2 * wfns[j, state[j], gj]**2

        return scfcorr

    def _veffective(self, mode, state, wfn):

        veff = np.zeros(self.ngrid)
        for i in range(self.ngrid):
            for j in range(self.nmodes):
                if j != mode:
             
                    if (mode,j) in self.v2.indices or (j,mode) in self.v2.indices: #first check if there's the pot for two modes
                        try:
                            ind = self.v2.indices.index((mode,j))
                        except:
                            ind = self.v2.indices.index((j,mode))
                        #for k in range(self.ngrid):
                        #    tmpsum += (wfn[j, state[j],k]**2 * self.dx[j] * self.v2.data[ind][i, k])
                        if mode < j:
                            veff[i] += (wfn[j, state[j]]**2 * self.dx[j] * self.v2.data[ind][i,:]).sum()
                        elif mode > j:
                            veff[i] += (wfn[j, state[j]]**2 * self.dx[j] * self.v2.data[ind][:,i]).sum()
                #veff[i] += tmpsum 

        return veff
                    


