# This file is a part of 
# Vibrations - a Python Code for Anharmonic Vibrational Calculations
# wiht User-defined Vibrational Coordinates
# Copyright (C) 2014-2016 by Pawel T. Panek, and Christoph R. Jacob.
#
# Vibrations is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Vibrations is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Vibrations.  If not, see <http://www.gnu.org/licenses/>.

"""
Module related to the VCI class for Vibrational Confinuration Interaction calculations
"""

import numpy as np
import scipy
import scipy.linalg
import Misc
import Surfaces
import fints


def multichoose(n, k):
    """
    General algorithm for placing k balls in n buckets. Here will be used
    for generating VCI states

    @param n: Number of buckets, here number of modes
    @type n: Integer
    @param k: Number of balls, here excitation quanta
    @type k: Integer
    """
    if not k:
        return [[0]*n]
    if not n:
        return []
    if n == 1:
        return [[k]]
    return [[0]+val for val in multichoose(n-1, k)] + \
            [[val[0]+1]+val[1:] for val in multichoose(n, k-1)]


class VCI(object):
    """
    The class performing and storing VCI calculations
    """
    def __init__(self, wavefunctions, *potentials):
        """
        The class must be initialized with grids, some, e.g. VSCF, wave functions, and potentials
        @param wavefunctions: The object containing the reference wave function, e.g. VSCF wfn
        @type wavefunctions: Vibrations/Wavefunction
        @param potentials: The potentials
        @type potentials: Vibrations/Potential
        """

        self.grids = potentials[0].grids.grids.copy()  #
        self.wfns = wavefunctions.wfns.copy()  # these are VSCF optimized wave functions
        self.nmodes = int(potentials[0].grids.nmodes)
        self.ngrid = int(potentials[0].grids.ngrid)

        self.states = []
        self.combinations = None
        self.solved = False

        self.dx = [x[1]-x[0] for x in self.grids]
        self.a = [(0.7**2.0)/((x[1]-x[0])**2.0) for x in self.grids]

        self.coefficients = np.zeros((self.nmodes, self.ngrid, self.ngrid))  #
        self._calculate_coeff()

        self.sij = np.zeros((self.nmodes, self.ngrid, self.ngrid))  # calculate Sij only once
        self._calculate_ovrlp_integrals()

        self.tij = np.zeros((self.nmodes, self.ngrid, self.ngrid))  # calculate Tij only once as well
        self._calculate_kinetic_integrals()
        
        self.store_ints = True  # prescreen the precomputed 1-mode integrals
        self.integrals = {}  # dictionary storing the precomputed parts of the  1-mode integrals, {(mode,lstate,rstate):val}
        self.store_potints = True # store 1-,2-,3-, 4-mode integrals
        self.int1d = {}  # 1-d integrals involving potentials
        self.int2d = {}  # 2-d -,,-
        self.int3d = {}  # 3-d -,,-
        self.int4d = {}  # 4-d -,,-

        self.fortran = True # use Fortran integrals (should be faster)

        self.energies = np.array([])
        self.energiesrcm = np.array([])
        self.H = np.array([])
        self.vectors = np.array([])
        self.intensities = None

        self.maxpot = None
        import copy
        if len(potentials) == 3:
            self.v1_indices = (potentials[0].indices)
            self.v1_data = (potentials[0].data)
            self.v2_indices = (potentials[1].indices)
            self.v2_data = (potentials[1].data)
            self.v3_indices = (potentials[2].indices)
            self.v3_data = (potentials[2].data)
            self.maxpot = 3
        elif len(potentials) == 4:
            self.v1_indices = (potentials[0].indices)
            self.v1_data = (potentials[0].data)
            self.v2_indices = (potentials[1].indices)
            self.v2_data = (potentials[1].data)
            self.v3_indices = (potentials[2].indices)
            self.v3_data = (potentials[2].data)
            self.v4_indices = (potentials[3].indices)
            self.v4_data = (potentials[3].data)
            self.maxpot = 4
        elif len(potentials) == 2:
            self.v1_indices = (potentials[0].indices)
            self.v1_data = (potentials[0].data)
            self.v2_indices = (potentials[1].indices)
            self.v2_data = (potentials[1].data)
            self.maxpot = 2
        else:
            raise Exception('Only two- or three-mode potentials accepted')
   

    def calculate_transition(self,c):
        order = self.order_of_transition(c)
        #print 'Solving the transition: ',c
        if order == 0:
            tmp = self.calculate_diagonal(c)
        elif order == 1:
            tmp = self.calculate_single(c)
        elif order == 2:
            tmp = self.calculate_double(c)
        elif order == 3 and self.maxpot > 2:
            tmp = self.calculate_triple(c)
        elif order == 4 and self.maxpot > 3:
            tmp = self.calculate_quadriple(c)
        else:
            tmp = 0.0

        if abs(tmp) < 1e-8: 
            tmp = 0.0

        nind = self.states.index(c[0])  # find the indices of the vectors
        mind = self.states.index(c[1])
        return (nind,mind,tmp)

    def __call__(self,c):
        tmp = self.calculate_transition(c)
        return tmp


    #@do_cprofile
    def calculate_diagonal(self, c):
        """
        Calculates a diagonal element of the VCI matrix
        :param c: Configuration
        :return: Value of the diagonal element, in a.u.
        """
        tmp = 0.0
        n = c[0]
        m = c[1]
        
        if self.maxpot == 2:
            for i in xrange(self.nmodes):
                tmpv1 = self._v1_integral(i, n[i], m[i])
                tmpt = self._kinetic_integral(i, n[i], m[i])
                tmp += tmpv1 + tmpt

                for j in xrange(i+1, self.nmodes):
                    tmpv2 = self._v2_integral(i, j, n[i], n[j], m[i], m[j])
                    tmp += tmpv2 

        elif self.maxpot == 3:
            for i in xrange(self.nmodes):
                tmpv1 = self._v1_integral(i, n[i], m[i])
                tmpt = self._kinetic_integral(i, n[i], m[i])
                tmp += tmpv1 + tmpt

                for j in xrange(i+1, self.nmodes):
                    tmpv2 = self._v2_integral(i, j, n[i], n[j], m[i], m[j])
                    tmp += tmpv2 
                    for k in xrange(j+1,self.nmodes):
                        tmpv3 = self._v3_integral(i, j, k, n[i], n[j], n[k], m[i], m[j], m[k])
                        tmp += tmpv3
        elif self.maxpot == 4:
            for i in xrange(self.nmodes):
                tmpv1 = self._v1_integral(i, n[i], m[i])
                tmpt = self._kinetic_integral(i, n[i], m[i])
                tmp += tmpv1 + tmpt

                for j in xrange(i+1, self.nmodes):
                    tmpv2 = self._v2_integral(i, j, n[i], n[j], m[i], m[j])
                    tmp += tmpv2 
                    for k in xrange(j+1,self.nmodes):
                        tmpv3 = self._v3_integral(i, j, k, n[i], n[j], n[k], m[i], m[j], m[k])
                        tmp += tmpv3
                        for l in xrange(k+1,self.nmodes):
                            tmpv4 = self._v4_integral(i,j,k,l,n[i],n[j],n[k],n[l],m[i],m[j],m[k],m[l])
                            tmp += tmpv4

        return tmp

    #@do_cprofile
    def calculate_single(self, c):
        """
        Calculates an element corresponding to a single transition
        :param c: Configuration
        :return: Value of the element, a.u.
        """
        tmp = 0.0
        n = c[0]
        m = c[1]
        i = [x != y for x, y in zip(n, m)].index(True)  # give me the index of the element that differs two vectors

        tmpv1 = self._v1_integral(i, n[i], m[i])
        tmpt = self._kinetic_integral(i, n[i], m[i])
        tmp += tmpv1 + tmpt

        if self.maxpot == 2:
            for j in xrange(self.nmodes):
                if j != i:
                    tmpv2 = self._v2_integral(i, j, n[i], n[j], m[i], m[j])
                    tmp += tmpv2
        elif self.maxpot == 3:
            for j in xrange(self.nmodes):
                if j != i:
                    tmpv2 = self._v2_integral(i, j, n[i], n[j], m[i], m[j])
                    tmp += tmpv2
                    for k in xrange(j+1,self.nmodes):
                        if k != j and k != i:
                            tmpv3 = self._v3_integral(i, j, k, n[i], n[j], n[k], m[i], m[j], m[k])
                            tmp += tmpv3
        elif self.maxpot == 4:
            for j in xrange(self.nmodes):
                if j != i:
                    tmpv2 = self._v2_integral(i, j, n[i], n[j], m[i], m[j])
                    tmp += tmpv2
                    for k in xrange(j+1,self.nmodes):
                        if k != j and k != i:
                            tmpv3 = self._v3_integral(i, j, k, n[i], n[j], n[k], m[i], m[j], m[k])
                            tmp += tmpv3
                            for l in xrange(k+1, self.nmodes):
                                if l != j and l!= i and l != k:
                                    tmpv4 = self._v4_integral(i,j,k,l,n[i],n[j],n[k],n[l],m[i],m[j],m[k],m[l])
                                    tmp += tmpv4

        return tmp

    #@do_cprofile
    def calculate_double(self, c):
        """
        Calculates an element corresponding to a double transition
        :param c: Configuration
        :return: Value of the element, a.u.
        """
        tmp = 0.0
        n = c[0]
        m = c[1]

        indices = [ind for ind, e in enumerate([x != y for x, y in zip(n, m)]) if e]

        i = indices[0]
        j = indices[1]

        tmpv2 = self._v2_integral(i, j, n[i], n[j], m[i], m[j])
        tmp += tmpv2

        if self.maxpot == 3 :
            for k in xrange(self.nmodes):
                if k != i and k != j:
                    tmpv3 = self._v3_integral(i, j, k, n[i], n[j], n[k], m[i], m[j], m[k])
                    tmp += tmpv3
        elif self.maxpot == 4:
            for k in xrange(self.nmodes):
                if k != i and k != j:
                    tmpv3 = self._v3_integral(i, j, k, n[i], n[j], n[k], m[i], m[j], m[k])
                    tmp += tmpv3
                    for l in xrange(k+1,self.nmodes):
                        if l != i and l != j and l != k:
                            tmpv4 = self._v4_integral(i,j,k,l,n[i],n[j],n[k],n[l],m[i],m[j],m[k],m[l])
                            tmp += tmpv4


        return tmp
    
    #@do_cprofile
    def calculate_triple(self, c):
        """
        Calculates an element corresponding to a triple transition
        :param c: Configuration
        :return: Value of the element, a.u.
        """
        tmp = 0.0
        n = c[0]
        m = c[1]

        indices = [ind for ind, e in enumerate([x != y for x, y in zip(n, m)]) if e]
        i = indices[0]
        j = indices[1]
        k = indices[2]

        tmpv3 = self._v3_integral(i, j, k, n[i], n[j], n[k], m[i], m[j], m[k])
        tmp += tmpv3
        
        if self.maxpot == 4 :
            for l in xrange(self.nmodes):
                if l != i and l!= j and l!=k:
                    tmpv4 = self._v4_integral(i,j,k,l,n[i],n[j],n[k],n[l],m[i],m[j],m[k],m[l])
                    tmp += tmpv4


        return tmp

    def calculate_quadriple(self, c):
        """
        Calculates an element corresponding to a quadriple transition
        :param c: Configuration
        :return: Value of the element, a.u.
        """
        tmp = 0.0
        n = c[0]
        m = c[1]
        
        indices = [ind for ind, e in enumerate([x != y for x, y in zip(n, m)]) if e]
        i = indices[0]
        j = indices[1]
        k = indices[2]
        l = indices[3]
        tmpv4 = self._v4_integral(i,j,k,l,n[i],n[j],n[k],n[l],m[i],m[j],m[k],m[l])
        tmp += tmpv4

        return tmp
        

    @staticmethod
    def order_of_transition(c):
        """
        Gives the order of the transition (how many modes are changed upon the transition)
        :param c: combination, a tuple of two vectors, left and right, representing the transition
        :return: order of the transition, 1 for singles, 2 for doubles etc.
        """
        return np.count_nonzero(c[0]-c[1])

    def nex_state(self, vci_state):
        """
        Returns the contributions of different types of states (0 - ground state,
        1 - fundamentals, 2 - doubly excited etc.) for a given VCI state.
        """
        nex_contrib = [0.0] * (self.smax+1)
        for j, s in enumerate(self.states):
            nex_contrib[sum(s)] += vci_state[j]**2
        return nex_contrib

    def nexmax_state(self, vci_state):
        nex_contrib = self.nex_state(vci_state)
        nex_contrib_max = max(enumerate(nex_contrib), key=lambda x: x[1])[0]
        return nex_contrib_max

    def idx_fundamentals(self):
        if self.solved :
            idx = []

            nex_contrib_fund = np.zeros_like(self.energies)
            for i in xrange(len(self.energies)):
                nex_contrib_fund[i] = self.nex_state(self.vectors[:,i])[1]

            idx = np.argsort(nex_contrib_fund)
            idx = np.sort(idx[-self.nmodes:])

            return list(idx)
        else:
            return None

    def print_results(self, which=1, maxfreq=4000, short=False):
        """
        Prints VCI results, can be limited to the states mostly contributed from given type of transitions (1 - singles,
        etc.), and to the maximal energy (usually 4000cm^-1 is the range of interest)
        :param which: transitions to which states should be included, 1 for singles, 2 for SD, etc.
        :param maxfreq: frequency threshold
        :return: void
        """
        if self.solved:
            print Misc.fancy_box('Results of the VCI')
            print 'State %14s %10s %10s' % ('Contrib', 'E /cm^-1', 'DE /cm^-1')
            for i in xrange(len(self.energies)):  # was self.states, is self.energies
                en = self.energiesrcm[i] - self.energiesrcm[0]
                if en < maxfreq:
                    nex_contrib = self.nex_state(self.vectors[:,i])
                    nex_contrib_max = max(enumerate(nex_contrib), key=lambda x: x[1])[0]

                    state = self.states[(self.vectors[:, i]**2).argmax()]
                    if nex_contrib_max <= which:
                        if not short:
                            print "%s %10.4f %10.4f %10.4f" % (state, (self.vectors[:, i]**2).max(), 
                                                               self.energiesrcm[i], en)
                        else:
                            print "%s %10.4f %10.4f %10.4f" % (self.print_short_state(state), (self.vectors[:, i]**2).max(), 
                                                               self.energiesrcm[i], en)
        else:
            print Misc.fancy_box('Solve the VCI first')
        print

    def print_short_state(self,state):
        s = ''
        ds = [state.index(x) for x in state if x]
        s += str(len(ds))
        s += ': '
        for d in ds:
            s += str(d)+ '(' + str(state[d]) + ')' + ' '

        return s


    def print_contributions(self, mincon=0.1,which=1, maxfreq=4000):
        """
        Prints VCI results, can be limited to the states mostly contributed from given type of transitions (1 - singles,
        etc.), and to the maximal energy (usually 4000cm^-1 is the range of interest)
        :param mincon: contribution threshold, 0.1 by default
        :param which: transitions to which states should be included, 1 for singles, 2 for SD, etc.
        :param maxfreq: frequency threshold
        :return: void
        """
        if self.solved:
            print Misc.fancy_box('Results of the VCI')
            print 'State %15s %15s %15s' % ('Contrib', 'E /cm^-1', 'DE /cm^-1')

            for i in xrange(len(self.energies)): #was self.states is self.energies
                en = self.energiesrcm[i] - self.energiesrcm[0]
                if en < maxfreq:
                    nex_contrib = self.nex_state(self.vectors[:,i])
                    nex_contrib_max = max(enumerate(nex_contrib), key=lambda x: x[1])[0]

                    if nex_contrib_max <= which:
                        print "State %3i      energy = %10.4f,  excitation energy = %10.4f" % (i, self.energiesrcm[i], en)
                        for j, contr in enumerate(self.vectors[:, i]):
                            
                            if contr**2 >= mincon:
                                print " %9.4f   %s  %s" % (contr, self.states[j], self.print_short_state(self.states[j]))

                        print 15*' ' + "GS: %6.2f, Fundamentals: %6.2f " % (nex_contrib[0], nex_contrib[1]),
                        for j in range(2, self.smax+1):
                            print "%2i: %6.2f" % (j, nex_contrib[j]),
                        print
                        print

                                
    def print_states(self):
        """
        Prints the vibrational states used in the VCI calculations
        """
        print ''
        print Misc.fancy_box('VCI Space')
        print 'There are %i states: ' %len(self.states)
        for i,s in enumerate(self.states):
            print i, s
        print

    def generate_states_nmax(self, nexc=None, smax=None):
        """
        Generates the states for VCI calculations in a way that
        VCI[1] means that at most 1 state is excited at a time to the
        excitation state nmax. For combination states VCI[2] etc. 
        all the states where sum of exc. quanta is smaller than nmax 
        are included.
        
        @param nexc: Maximal number of modes excited simultaneously
        @type nexc: Integer
        @param smax: Maximal sum of excitation quanta
        @type smax: Integer
        """
        import itertools

        if not nexc:
            nexc = 1  # singles by default
        if not smax:
            smax = 4  # up to the fourth excited state

        res = [[0] * self.nmodes]

        for i in xrange(1, smax+1):
            res += multichoose(self.nmodes, i)

        res = filter(lambda x: len(filter(None, x)) < nexc + 1, res)
        self.states = np.array(res)

        self.nmax = nexc
        self.smax = smax

    def generate_states(self, maxexc=1):
        self.generate_states_nmax(maxexc, maxexc)
        self.nmax = maxexc
        self.smax = maxexc


    def filter_combinations(self):
        """
        Filters out the combinations (transitions) that do not contribute due to the max potential dimensionality

        """
        if self.combinations:
            res = [c for c in self.combinations if sum([x != y for (x, y) in zip(c[0], c[1])]) < self.maxpot+1]
            self.combinations = res

    def combgenerator(self):
        """
        Generator returning combinations of states that contribute due to available potentials
        """
        nstates = len(self.states)
        for i in xrange(nstates):
            if i % 500 == 0:
                 print 'combgenerator', i, 'of', nstates
            sdiff = np.count_nonzero(self.states[i:,:] - self.states[i,:], axis=1)
            for j in np.nonzero(sdiff < self.maxpot+1)[0] :
                 yield (self.states[i], self.states[j+i], i, j+i)
    
    def combgenerator_nofilter(self):
        """
        Generates combinations of states without prescreening
        """
        nstates = len(self.states)
        for i in xrange(nstates):
            for j in xrange(i, nstates):
                yield (self.states[i], self.states[j], i, j)

    def calculate_transition_moments(self, *properties):
        """
        Calculates VCI transition moments for given properties
        """
        if not self.solved:
            raise Exception('Solve the VCI first')

        maxprop = len(properties)
        self.prop1 = None
        self.prop2 = None
        self.transitions = None

        if maxprop == 0:
            raise Exception('No property surfaces given')
        elif maxprop == 1:
            self.prop1 = properties[0]
        elif maxprop == 2:
            self.prop1 = properties[0]
            self.prop2 = properties[1]
        else:
            raise Exception('Too many properties, only up to second order used')

        tensors = self.prop1
        tensors2 = None
        if self.prop2:
            tensors2 = self.prop2


        totaltens = []
        tmptens = []
        for t in tensors:
            tmptens.append(np.zeros(t.prop[0]))
            totaltens.append(np.zeros(t.prop[0]))

        nstates = len(self.states)
        transitions = [[] for i in range(nstates)]

        for i in xrange(1, nstates):  # loop over all VCI states except the ground state
            totaltens = []
            for t in tensors:
                totaltens.append(np.zeros(t.prop[0]))
            for istate in xrange(nstates):
                ci = self.vectors[istate, 0]  # initial state's coefficient

                for fstate in xrange(nstates):
                    cf = self.vectors[fstate, i]  # final state's coefficient
                    if ci and cf:
                        for tt in tmptens:
                            tt *= 0.0
                        
                        order = self.order_of_transition((self.states[istate], self.states[fstate]))
                        if order == 0:
                            for j in xrange(self.nmodes):
                                jistate = self.states[istate][j]
                                jfstate = self.states[fstate][j]
                                try:
                                    s1 = self.integrals[(j,jistate,jfstate)]
                                except:
                                    s1 = (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate])
                                    self.integrals[(j,jistate,jfstate)] = s1
                                
                                for ti, t in enumerate(tensors):
                                    ind = t.indices.index(j)
                                    for o in range(t.prop[0]):
                                        tmptens[ti][o] += np.dot(t.data[ind][:,o],s1)

                                if tensors2:
                                    for k in xrange(j+1, self.nmodes):
                                        if (j,k) in tensors2[0].indices:
                                            kistate = self.states[istate][k]
                                            kfstate = self.states[fstate][k]
                                            try:
                                                s2 = self.integrals[(k,kistate,kfstate)]
                                            except:
                                                s2 = (self.dx[k] * self.wfns[k, kistate] * self.wfns[k, kfstate])
                                                self.integrals[(k,kistate,kfstate)] = s2
                                             
                                            for ti, t in enumerate(tensors2):
                                                ind = t.indices.index((j,k)) 
                                                for o in range(t.prop[0]):
                                                    tmptens[ti][o] += np.einsum('i,j,ij', s1, s2, t.data[ind][:,:,o])

                        elif order == 1:
                            n = self.states[istate]
                            m = self.states[fstate]
                            j = [x != y for x, y in zip(n, m)].index(True)  # give me the index of the element that differs two vectors
                            jistate = self.states[istate][j]
                            jfstate = self.states[fstate][j]
                            try:
                                s1 = self.integrals[(j,jistate,jfstate)]
                            except:
                                s1 = (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate])
                                self.integrals[(j,jistate,jfstate)] = s1
                            for ti, t in enumerate(tensors):
                                                                                      
                                ind = t.indices.index(j)
                                for o in range(t.prop[0]):
                                    tmptens[ti][o] += np.dot(t.data[ind][:,o],s1)
                            if tensors2:
                                for k in xrange(self.nmodes):
                                    if j!=k and ((j,k) in tensors2[0].indices or (k,j) in tensors2[0].indices):
                                        kistate = self.states[istate][k]
                                        kfstate = self.states[fstate][k]
                                        try:
                                            s2 = self.integrals[(k,kistate,kfstate)]
                                        except:
                                            s2 = (self.dx[k] * self.wfns[k, kistate] * self.wfns[k, kfstate])
                                            self.integrals[(k,kistate,kfstate)] = s2
                                        
                                        for ti, t in enumerate(tensors2):
                                            try:
                                                ind = t.indices.index((j,k))
                                            except:
                                                ind = t.indices.index((k,j))
                                            for o in range(t.prop[0]):
                                                tmptens[ti][o] += np.einsum('i,j,ij', s1, s2, t.data[ind][:,:,o])
                        elif tensors2 and  order == 2:
                            n = self.states[istate]
                            m = self.states[fstate]
                            j,k = [ind for ind, e in enumerate([x != y for x, y in zip(n, m)]) if e]
                            jistate = n[j]
                            jfstate = m[j]
                            kistate = n[k]
                            kfstate = m[k]
                            try:
                                s1 = self.integrals[(j,jistate,jfstate)]
                            except:
                                s1 = (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate])
                                self.integrals[(j,jistate,jfstate)] = s1
                                try:
                                    s2 = self.integrals[(k,kistate,kfstate)]
                                except:
                                    s2 = (self.dx[k] * self.wfns[k, kistate] * self.wfns[k, kfstate])
                                    self.integrals[(k,kistate,kfstate)] = s2
                                if (j,k) in t.indices or (k,j) in t.indices: 
                                    for ti, t in enumerate(tensors2):
                                        try:
                                            ind = t.indices.index((j,k))
                                        except:
                                            ind = t.indices.index((k,j))
                                        for o in range(t.prop[0]):
                                            tmptens[ti][o] += np.einsum('i,j,ij', s1, s2, t.data[ind][:,:,o])
                        for tn,tt in enumerate(totaltens):
                            tt += tmptens[tn]   * ci * cf
            transitions[i]=totaltens
        return transitions
    
    #@do_cprofile
    def calculate_IR(self, *dipolemoments):
        
        print Misc.fancy_box('VCI IR Intensities')

        self.dm1 = None
        self.dm2 = None
        if len(dipolemoments) == 0:
            raise Exception('No dipole moments given')
        elif len(dipolemoments) == 1:
            print 'Only one set of dipole moments given.'
            print
            self.dm1 = dipolemoments[0]
            transm = self.calculate_transition_moments([self.dm1])
        elif len(dipolemoments) == 2:
            print 'Two sets of dipole moments given.'
            print
            self.dm1 = dipolemoments[0]
            self.dm2 = dipolemoments[1]
            transm = self.calculate_transition_moments([self.dm1],[self.dm2])
        elif len(dipolemoments) > 2:
            print 'More than two sets of dipole moments given, only the two first will be used'
            print
            self.dm1 = dipolemoments[0]
            self.dm2 = dipolemoments[1]
            transm = self.calculate_transition_moments([self.dm1],[self.dm2])
       
        
        nstates = len(self.states)
        self.intensities=np.zeros(nstates)
        print '%7s %7s' %('Freq.','Int.')
        print '%7s %7s' %('[cm^-1]','[km*mol^-1]')
        for i in range(1,nstates):
            totaltm = transm[i]
            for tens in totaltm:
                intens = (tens[0]**2 + tens[1]**2 + tens[2]**2) * Misc.intfactor * (self.energiesrcm[i]
                                                                                     - self.energiesrcm[0])
            self.intensities[i] = intens
            print '%7.1f %12.6f' % (self.energiesrcm[i] - self.energiesrcm[0], intens)
        

    def calculate_intensities(self, *dipolemoments):
        """
        Calculates VCI intensities using the dipole moment surfaces

        @param dipolemoments: dipole moment surfaces, so far only 1- and 2-mode DMS supported
        @type dipolemoments: Surfaces.Dipole
        """

        if not self.solved:
            raise Exception('Solve the VCI first')
        self.dm1 = None
        self.dm2 = None
        if len(dipolemoments) == 0:
            raise Exception('No dipole moments given')
        elif len(dipolemoments) == 1:
            print 'Only one set of dipole moments given, the second will be taken as 0.'
            self.dm1 = dipolemoments[0]
        elif len(dipolemoments) == 2:
            print 'Two sets of dipole moments given.'
            self.dm1 = dipolemoments[0]
            self.dm2 = dipolemoments[1]
        elif len(dipolemoments) > 2:
            print 'More than two sets of dipole moments given, only the two first will be used'
            self.dm1 = dipolemoments[0]
            self.dm2 = dipolemoments[1]

        self.intensities = np.zeros(len(self.states))

        # assuming that the first state is a ground state
        totaltm = np.zeros(3)
        tmptm = np.zeros(3)
        tmpd1 = np.zeros(3)
        tmpd2 = np.zeros(3)
        nstates = len(self.states)

        for i in xrange(1, nstates):
            totaltm *= 0.0
            for istate in xrange(nstates):
                ci = self.vectors[istate, 0]  # initial state's coefficient

                for fstate in xrange(nstates):
                    cf = self.vectors[fstate, i]  # final state's coefficient
                    if ci and cf:
                        tmptm *= 0.0

                        for j in xrange(self.nmodes):
                            tmpd1 *= 0.0
                            tmpovrlp = 1.0
                            jistate = self.states[istate][j]
                            jfstate = self.states[fstate][j]

                            for k in xrange(self.nmodes):

                                if k == j:
                                    #  calculate <psi|u|psi>
                                    ind = self.dm1.indices.index(k)
                                    tmpd1[0] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                                 * self.dm1.data[ind][:, 0]).sum()
                                    tmpd1[1] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                                 * self.dm1.data[ind][:, 1]).sum()
                                    tmpd1[2] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                                 * self.dm1.data[ind][:, 2]).sum()

                                else:
                                    if self.states[istate][k] == self.states[fstate][k]:
                                        tmpovrlp *= 1.0
                                    else:
                                        tmpovrlp = 0.0

                            tmptm += tmpd1 * tmpovrlp
                        if self.dm2:
                            for j in xrange(self.nmodes):
                                jistate = self.states[istate][j]
                                jfstate = self.states[fstate][j]
                                for k in xrange(j+1, self.nmodes):
                                    tmpd2 *= 0.0
                                    kistate = self.states[istate][k]
                                    kfstate = self.states[fstate][k]

                                    ind = self.dm2.indices.index((j, k))

                                    for l in xrange(self.ngrid):
                                        for m in xrange(self.ngrid):
                                            tmpd2[0] += self.dx[j] * self.dx[k] * self.dm2.data[ind][l, m, 0] \
                                                * self.wfns[j, jistate, l] * self.wfns[j, jfstate, l] \
                                                * self.wfns[k, kistate, m] * self.wfns[k, kfstate, m]
                                            tmpd2[1] += self.dx[j] * self.dx[k] * self.dm2.data[ind][l, m, 1] \
                                                * self.wfns[j, jistate, l] * self.wfns[j, jfstate, l] \
                                                * self.wfns[k, kistate, m] * self.wfns[k, kfstate, m]
                                            tmpd2[2] += self.dx[j] * self.dx[k] * self.dm2.data[ind][l, m, 2] \
                                                * self.wfns[j, jistate, l] * self.wfns[j, jfstate, l] \
                                                * self.wfns[k, kistate, m] * self.wfns[k, kfstate, m]
                                    tmpovrlp = 1.0

                                    for n in xrange(self.nmodes):
                                        if n != j and n != k:
                                            nistate = self.states[istate][n]
                                            nfstate = self.states[fstate][n]

                                            if nistate == nfstate:
                                                tmpovrlp *= 1.0

                                            else:
                                                tmpovrlp = 0.0

                                    tmptm += tmpd2 * tmpovrlp
                        totaltm += tmptm * ci * cf
            print np.square(totaltm).sum()
            intens = (totaltm[0]**2 + totaltm[1]**2 + totaltm[2]**2) * Misc.intfactor * (self.energiesrcm[i]
                                                                                 - self.energiesrcm[0])
            self.intensities[i] = intens
            print '%7.1f %12.6f' % (self.energiesrcm[i] - self.energiesrcm[0], intens)


    def calculate_raman(self, *pols):
        """
        Calculates VCI Raman activities using the polarizability surfaces
        """

        if not self.solved:
            raise Exception('Solve the VCI first')
        self.pol1 = None
        self.pol2 = None
        if len(pols) == 0:
            raise Exception('No dipole moments given')
        elif len(pols) == 1:
            print 'Only one set of dipole moments given, the second will be taken as 0.'
            self.pol1 = pols[0]
            self.maxpol = 1
        elif len(pols) > 1:
            print 'More than one sets of properties given, only the first will be used'
            self.pol1 = pols[0]
            self.maxpol = 1


        self.intensities = np.zeros(len(self.states))

        totalpol = np.zeros(6)
        tmptm = np.zeros(6)
        tmpp1 = np.zeros(6)
        nstates = len(self.states)

        for i in xrange(1, nstates):
            totalpol *= 0.0
            for istate in xrange(nstates):
                ci = self.vectors[istate, 0]  # initial state's coefficient

                for fstate in xrange(nstates):
                    cf = self.vectors[fstate, i]  # final state's coefficient
                    if ci and cf:   
                        order = self.order_of_transition((self.states[istate], self.states[fstate]))
                        tmpp1 *= 0.0
                        if order == 0:
                            for j in xrange(self.nmodes):
                                jistate = self.states[istate][j]
                                jfstate = self.states[fstate][j]
                                ind = self.pol1.indices.index(j)
                                tmpp1[0] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                             * self.pol1.data[ind][:, 0]).sum()
                                tmpp1[1] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                             * self.pol1.data[ind][:, 1]).sum()
                                tmpp1[2] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                             * self.pol1.data[ind][:, 2]).sum()
                                tmpp1[3] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                             * self.pol1.data[ind][:, 3]).sum()
                                tmpp1[4] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                             * self.pol1.data[ind][:, 4]).sum()
                                tmpp1[5] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                             * self.pol1.data[ind][:, 5]).sum()
                        elif order == 1:
                            n = self.states[istate]
                            m = self.states[fstate]
                            j = [x != y for x, y in zip(n, m)].index(True)  # give me the index of the element that differs two vectors
                            jistate = self.states[istate][j]
                            jfstate = self.states[fstate][j]
                            ind = self.pol1.indices.index(j)
                            tmpp1[0] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                         * self.pol1.data[ind][:, 0]).sum()
                            tmpp1[1] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                         * self.pol1.data[ind][:, 1]).sum()
                            tmpp1[2] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                         * self.pol1.data[ind][:, 2]).sum()
                            tmpp1[3] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                         * self.pol1.data[ind][:, 3]).sum()
                            tmpp1[4] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                         * self.pol1.data[ind][:, 4]).sum()
                            tmpp1[5] += (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate]
                                         * self.pol1.data[ind][:, 5]).sum()
                        
                        totalpol += tmpp1 * ci * cf
    
            a2 = 1.0/3.0 * (totalpol[0] + totalpol[3] + totalpol[5])
            a2 = a2**2  * Misc.Bohr_in_Angstrom**4
            g2 = 1.0/2.0 * ((totalpol[0]-totalpol[3])**2 + (totalpol[3]-totalpol[5])**2
                           +(totalpol[5]-totalpol[0])**2 + 6.0 * totalpol[1]**2
                           + 6.0 * totalpol[2]**2 + 6.0 * totalpol[4]**2)
            g2 *= Misc.Bohr_in_Angstrom**4
            f = self.energies[i] - self.energies[0]
            intens = 2 * f *  (45.0 * a2 + 7.0 * g2)
            self.intensities[i] = intens
            print '%7.1f %12.6f' % (self.energiesrcm[i] - self.energiesrcm[0], intens)

    #@do_cprofile
    def calculate_roa(self, pollen,polvel,gtenvel, aten, lwl):
        """
        Calculate ROA backscattering, instead of explicit property tensors,
        just use the results instance, so that everything is generated "on the fly"
        """
        self.lwl = lwl

        tensors = [pollen, polvel, gtenvel, aten]


        self.intensities = np.zeros(len(self.states))

        totaltens = []
        tmptens = []
        for t in tensors:
            tmptens.append(np.zeros(t.prop[0]))
            totaltens.append(np.zeros(t.prop[0]))

        nstates = len(self.states)

        for i in xrange(1, nstates):
            for tt in totaltens:
                tt *= 0.0


            for istate in xrange(nstates):
                ci = self.vectors[istate, 0]  # initial state's coefficient

                for fstate in xrange(nstates):
                    cf = self.vectors[fstate, i]  # final state's coefficient
                    if ci and cf:
                        for tt in tmptens:
                            tt *= 0.0

                        order = self.order_of_transition((self.states[istate], self.states[fstate]))
                        if order == 0:
                            for j in xrange(self.nmodes):
                                jistate = self.states[istate][j]
                                jfstate = self.states[fstate][j]
                                
                                for ti, t in enumerate(tensors):
                                    ind = t.indices.index(j)
                                    for o in range(t.prop[0]):
                                        try:
                                            s1 = self.integrals[(j,jistate,jfstate)]
                                        except:
                                            s1 = (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate])
                                            self.integrals[(j,jistate,jfstate)] = s1

                                        tmptens[ti][o] += np.dot(t.data[ind][:,o],s1)

                        elif order == 1:
                            n = self.states[istate]
                            m = self.states[fstate]
                            j = [x != y for x, y in zip(n, m)].index(True)  # give me the index of the element that differs two vectors
                            jistate = self.states[istate][j]
                            jfstate = self.states[fstate][j]
                            for ti, t in enumerate(tensors):
                                ind = t.indices.index(j)
                                for o in range(t.prop[0]):
                                    try:
                                        s1 = self.integrals[(j,jistate,jfstate)]
                                    except:
                                        s1 = (self.dx[j] * self.wfns[j, jistate] * self.wfns[j, jfstate])
                                        self.integrals[(j,jistate,jfstate)] = s1

                                    tmptens[ti][o] += np.dot(t.data[ind][:,o],s1)
                        
                        for tn,tt in enumerate(totaltens):
                            tt += tmptens[tn] * ci * cf

            # calculate the invariants
            # bG
            pi = tensors.index(polvel)
            gi = tensors.index(gtenvel)
            pol = totaltens[pi]
            gt = totaltens[gi]
            bG = 0.5 * (3*pol[0]*gt[0] - pol[0]*gt[0] +
                        3*pol[1]*gt[1] - pol[0]*gt[4] +
                        3*pol[2]*gt[2] - pol[0]*gt[8] +
                        3*pol[1]*gt[3] - pol[3]*gt[0] +
                        3*pol[3]*gt[4] - pol[3]*gt[4] +
                        3*pol[4]*gt[5] - pol[3]*gt[8] +
                        3*pol[2]*gt[6] - pol[5]*gt[0] +
                        3*pol[4]*gt[7] - pol[5]*gt[4] +
                        3*pol[5]*gt[8] - pol[5]*gt[8])
            bG = bG*(Misc.Bohr_in_Angstrom**4) * (1 / Misc.cvel) * 1e6
            # bA
            pi = tensors.index(pollen)
            ai = tensors.index(aten)
            pol = totaltens[pi]
            at = totaltens[ai]
            bA = 0.5 * self.lwl * (  (pol[3]-pol[0])*at[11]
                                   + (pol[0]-pol[5])*at[6]
                                   + (pol[5]-pol[3])*at[15]
                                   + pol[1]*(at[19]-at[20]+at[8]-at[14]) 
                                   + pol[2]*(at[25]-at[21]+at[3]-at[4]) 
                                   + pol[4]*(at[10]-at[24]+at[12]-at[5])
                                  )
            bA = bA*(Misc.Bohr_in_Angstrom**4) * (1 / Misc.cvel) * 1e6
            
            roa =  1e-6*96.0*(bG + (1.0/3.0)*bA)
            f = self.energies[i]-self.energies[0]
            roa *= 2*f
            self.intensities[i]=roa


    def _v1_integral(self, mode, lstate, rstate):  # calculate integral of type < mode(lstate) | V1 | mode(rstate) >
        ind = self.v1_indices.index(mode)

        try:
            return self.int1d[(mode,lstate,rstate)]

        except:

            try:
                s1 = self.integrals[(mode,lstate,rstate)]
            except:
                s1 = (self.dx[mode] * self.wfns[mode, lstate] * self.wfns[mode, rstate])
                if self.store_ints:
                    self.integrals[(mode,lstate,rstate)] = s1
            
            if self.fortran:
                s = fints.v1int(self.v1_data[ind],s1)
            else:
                s = (s1 * self.v1_data[ind]).sum()
            if self.store_potints:
                self.int1d[(mode,lstate,rstate)] = s

            return s
   
    def v1_integral(self,mode,lstate,rstate):
        return self._v1_integral(mode,lstate,rstate)


    def _v2_integral(self, mode1, mode2, lstate1, lstate2, rstate1, rstate2):
        s = 0.0

        if (mode1, mode2) in self.v2_indices or (mode2, mode1) in self.v2_indices:
            try:
                return self.int2d[(mode1,lstate1,rstate1,mode2,lstate2,rstate2)]
            except:

                try:
                    ind = self.v2_indices.index((mode1, mode2))
                except:
                    ind = self.v2_indices.index((mode2, mode1))

                if mode1 < mode2: 
                    try:
                        s1 = self.integrals[(mode1,lstate1,rstate1)]
                    except:
                        s1 = (self.dx[mode1] * self.wfns[mode1, lstate1] * self.wfns[mode1, rstate1])
                        if self.store_ints:
                            self.integrals[(mode1,lstate1,rstate1)] = s1
                    try:
                        s2 = self.integrals[(mode2,lstate2,rstate2)]
                    except:
                        s2 = (self.dx[mode2] * self.wfns[mode2, lstate2] * self.wfns[mode2, rstate2])
                        if self.store_ints:
                            self.integrals[(mode2,lstate2,rstate2)] = s2

                    if self.fortran:
                        s = fints.v2int(self.v2_data[ind],s1,s2)
                    else:
                        s1 = s1.transpose()
                        s = (s1.dot(self.v2_data[ind]).dot(s2)).sum()

                else:
                    try:
                        s1 = self.integrals[(mode1,lstate1,rstate1)]
                    except:
                        s1 = (self.dx[mode1] * self.wfns[mode1, lstate1] * self.wfns[mode1, rstate1])
                        if self.store_ints:
                            self.integrals[(mode1,lstate1,rstate1)] = s1
                    try:
                        s2 = self.integrals[(mode2,lstate2,rstate2)]
                    except:
                        s2 = (self.dx[mode2] * self.wfns[mode2, lstate2] * self.wfns[mode2, rstate2])
                        if self.store_ints:
                            self.integrals[(mode2,lstate2,rstate2)] = s2
                    
                    if self.fortran:
                        s = fints.v2int(self.v2_data[ind],s2,s1)
                    else:
                        s1 = s1.transpose()
                        s = (s1.dot(self.v2_data[ind].transpose()).dot(s2)).sum()

                if self.store_potints: 
                    self.int2d[(mode1,lstate1,rstate1,mode2,lstate2,rstate2)] = s

        return s


    def v2_integral(self, mode1, mode2, lstate1, lstate2, rstate1, rstate2):
        return self._v2_integral(mode1, mode2, lstate1, lstate2, rstate1, rstate2)
    
    #@do_cprofile
    def _v3_integral(self, mode1, mode2, mode3, lstate1, lstate2, lstate3,
                     rstate1, rstate2, rstate3):
        s = 0.0
        modes = list((mode1,mode2,mode3))
        lstates = list((lstate1,lstate2,lstate3))
        rstates = list((rstate1,rstate2,rstate3))
        ind = zip(modes,lstates,rstates)
        ind.sort()
        (modes,lstates,rstates)=zip(*ind)
        mode1 = modes[0]
        mode2 = modes[1]
        mode3 = modes[2]
        lstate1 = lstates[0]
        lstate2 = lstates[1]
        lstate3 = lstates[2]
        rstate1 = rstates[0]
        rstate2 = rstates[1]
        rstate3 = rstates[2]
        if (mode1,mode2,mode3) in self.v3_indices:
            ind = self.v3_indices.index((mode1,mode2,mode3))
        else:
            return 0.0
       
        try:
            return self.int3d[(mode1,lstate1,rstate1,mode2,lstate2,rstate2,mode3,lstate3,rstate3)]
        except:

            try:
                si = self.integrals[(mode1,lstate1,rstate1)]
            except:
                si = (self.dx[mode1] * self.wfns[mode1, lstate1] * self.wfns[mode1, rstate1])
                if self.store_ints:
                    self.integrals[(mode1,lstate1,rstate1)] = si
            try:
                sj = self.integrals[(mode2,lstate2,rstate2)]
            except:
                sj = (self.dx[mode2] * self.wfns[mode2, lstate2] * self.wfns[mode2, rstate2])
                if self.store_ints:
                    self.integrals[(mode2,lstate2,rstate2)] = sj
            try:
                sk = self.integrals[(mode3,lstate3,rstate3)]
            except:
                sk = (self.dx[mode3] * self.wfns[mode3, lstate3] * self.wfns[mode3, rstate3])
                if self.store_ints:
                    self.integrals[(mode3,lstate3,rstate3)] = sk
            
            if self.fortran:
                s = fints.v3int(self.v3_data[ind],si,sj,sk)
            else:
                s = np.einsum('i,j,k,ijk',si,sj,sk,self.v3_data[ind])  # einstein summation rules!
            if self.store_potints:
               self.int3d[(mode1,lstate1,rstate1,mode2,lstate2,rstate2,mode3,lstate3,rstate3)] = s
            return s

    def _v4_integral(self, mode1, mode2, mode3, mode4, lstate1, lstate2, lstate3,
                     lstate4, rstate1, rstate2, rstate3, rstate4):

        modes = list((mode1,mode2,mode3,mode4))
        lstates = list((lstate1,lstate2,lstate3,lstate4))
        rstates = list((rstate1,rstate2,rstate3,rstate4))
        ind = zip(modes,lstates,rstates)
        ind.sort()
        (modes,lstates,rstates)=zip(*ind)

        if (modes[0],modes[1],modes[2],modes[3]) in self.v4_indices:
            potind = self.v4_indices.index((modes[0],modes[1],modes[2],modes[3]))
        else:
            return 0.0

        mode1 = modes[0]
        mode2 = modes[1]
        mode3 = modes[2]
        mode4 = modes[3]
        lstate1 = lstates[0]
        lstate2 = lstates[1]
        lstate3 = lstates[2]
        lstate4 = lstates[3]
        rstate1 = rstates[0]
        rstate2 = rstates[1]
        rstate3 = rstates[2]
        rstate4 = rstates[3]

        try:
            return self.int4d[(mode1,lstate1,rstate1,mode2,lstate2,rstate2,mode3,lstate3,rstate3,mode4,lstate4,rstate4)]

        except:

            try:
                si = self.integrals[(mode1,lstate1,rstate1)]
            except:
                si = (self.dx[mode1] * self.wfns[mode1, lstate1] * self.wfns[mode1, rstate1])
                self.integrals[(mode1,lstate1,rstate1)] = si
            try:
                sj = self.integrals[(mode2,lstate2,rstate2)]
            except:
                sj = (self.dx[mode2] * self.wfns[mode2, lstate2] * self.wfns[mode2, rstate2])
                self.integrals[(mode2,lstate2,rstate2)] = sj
            try:
                sk = self.integrals[(mode3,lstate3,rstate3)]
            except:
                sk = (self.dx[mode3] * self.wfns[mode3, lstate3] * self.wfns[mode3, rstate3])
                self.integrals[(mode3,lstate3,rstate3)] = sk
            try:
                sl = self.integrals[(mode4,lstate4,rstate4)]
            except:
                sl = (self.dx[mode4] * self.wfns[mode4, lstate4] * self.wfns[mode4, rstate4])
                self.integrals[(mode4,lstate4,rstate4)] = sl
            
            if self.fortran:
                s = fints.v4int(self.v4_data[potind],si,sj,sk,sl)
            else:
                s = np.einsum('i,j,k,l,ijkl',si,sj,sk,sl,self.v4_data[potind])
            if self.store_potints:
                self.int4d[(mode1,lstate1,rstate1,mode2,lstate2,rstate2,mode3,lstate3,rstate3,mode4,lstate4,rstate4)] = s
            return s


    def _ovrlp_integral(self, mode, lstate, rstate):    # overlap integral < mode(lstates) | mode(rstate) >

        s = (self.dx[mode] * self.wfns[mode, lstate] * self.wfns[mode, rstate] * 1.0).sum()

        return s

    def _kinetic_integral(self, mode, lstate, rstate):  # kinetic energy integral < mode(lstate) | T | mode(rstate) >

        t = 0.0
        return np.einsum('i,j,ij', self.coefficients[mode, lstate],
                         self.coefficients[mode, rstate],self.tij[mode])
        for i in xrange(self.ngrid):
            for j in xrange(self.ngrid):
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

        for i in xrange(self.nmodes):  # for each mode

            for j in xrange(self.ngrid):  # for each state

                chi = np.zeros((self.ngrid, self.ngrid))
                for k in xrange(self.ngrid):
                    for l in xrange(self.ngrid):
                        chi[k, l] = self._chi(self.grids[i, l], self.a[i], self.grids[i, k])

                c = np.linalg.solve(chi, self.wfns[i, j])
                self.coefficients[i, j] = np.copy(c)

    def _calculate_kinetic_integrals(self):

        for i in xrange(self.nmodes):  # for each mode

            for j in xrange(self.ngrid):
                for k in xrange(self.ngrid):

                    self.tij[i, j, k] = self._dgs_kinetic_integral(i, j, k)

    def _calculate_ovrlp_integrals(self):

        for i in xrange(self.nmodes):  # for each mode

            for j in xrange(self.ngrid):
                for k in xrange(self.ngrid):

                    self.sij[i, j, k] = self._dgs_ovrlp_integral(i, j, k)
    
    #@Misc.do_cprofile
    def solve(self, parallel=False, diag='Direct'):
        """
        General solver for the VCI
        """

        if len(self.states) == 0:
            print Misc.fancy_box('No VCI states defined, by default singles will be used')
            self.generate_states()

        nstates = len(self.states)
        #s = 'There are %i states' %nstates
        print Misc.fancy_box('There are %i states') % (nstates)
        #print Misc.fancy_box(s)
        #for s in self.states:
        #    print s
        #print
        #self.print_states()

        #self.H = np.zeros((nstates, nstates))
        import scipy.sparse
        self.H = scipy.sparse.lil_matrix((nstates, nstates))

        if not parallel:
            import time
            counter = 1
            for c in self.combgenerator():
                order = self.order_of_transition(c)
                if order == 0:
                    tmp = self.calculate_diagonal(c)
                elif order == 1:
                    tmp = self.calculate_single(c)
                elif order == 2:
                    tmp = self.calculate_double(c)
                elif order == 3 and self.maxpot > 2:
                    tmp = self.calculate_triple(c)
                elif order == 4 and self.maxpot > 3:
                    tmp = self.calculate_quadriple(c)
                else:
                    tmp = 0.0

                if abs(tmp) < 1e-8: 
                    tmp = 0.0
                nind = c[2]  # find the indices of the vectors
                mind = c[3]
                self.H[nind, mind] = tmp
                self.H[mind, nind] = tmp
                counter += 1

        else:
            import dill
            import time
            import pathos.multiprocessing as mp
            ncores = 10
            pool = mp.ProcessingPool(nodes=ncores)
            # ntrans = sum(1 for _ in self.combgenerator())
            # ch,e = divmod(ntrans,ncores*4)
            # if e:
            #    ch += 1
            ch = 85
            print 'Ncores:' ,ncores
            print 'Chunksize: ',ch
            # print 'Transitions: ',ntrans
            results =  pool.map(self.calculate_transition, self.combgenerator(),chunksize=ch)
            for r in results:
                self.H[r[0],r[1]] = r[2]
                self.H[r[1],r[0]] = r[2]

        self.H = self.H.tocsr()

        if diag=='Direct':
            print Misc.fancy_box('Hamiltonian matrix constructed. Diagonalization...')
            w, v = np.linalg.eigh(self.H.toarray(), UPLO='U')
            self.energies = w
            self.vectors = v
            self.energiesrcm = self.energies / Misc.cm_in_au
            self.solved = True
            self.print_results()
        elif diag=='Iterative':
            print Misc.fancy_box('Hamiltonian matrix constructed. Iterative diagonalization.')
            k = (np.sort(self.H.diagonal()-self.H.diagonal()[0])/Misc.cm_in_au < 4000).nonzero()[0][-1]
            k += 50
            if k > self.H.shape[0]:
                k = self.H.shape[0]-1
            from scipy.sparse.linalg import eigsh
            w, v = eigsh(self.H, k=k, which='SA')
            self.energies = w
            self.vectors = v
            self.energiesrcm = self.energies / Misc.cm_in_au
            self.solved = True
            self.print_results()

        else:
            print Misc.fancy_box('Hamiltonian contstructed and saved to file. No diag.')
            np.save('Hessian.npy',self.H)


    def save_vectors(self, fname=None):
        """
        Save results, eigenvectors and eigenvalues, of VCI to .npz file
        """
        if self.solved:
            if not fname:
                fname = 'VCI_%i_%i_results.npz' %(self.nmax,self.smax)
            np.savez_compressed(fname, vec=self.vectors, enrcm=self.energiesrcm, states=self.states)
        else:
            print 'Solve VCI first'

    def read_vectors(self,fname=None):
        if fname:
            npzfile = np.load(fname)
            if np.all(self.states==npzfile['states']):
                self.vectors = npzfile['vec']
                self.energiesrcm = npzfile['enrcm']
                self.energies = self.energiesrcm * Misc.cm_in_au
                self.states = npzfile['states']
                self.solved = True
            else:
                print 'The results do not fit the defined CI space'

