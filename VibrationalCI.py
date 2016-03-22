"""
The module reeated to the VCI class for Vibrational Confinuration Interaction calculations
"""

import numpy as np
import Misc
import time

import cProfile

def do_cprofile(func):
        def profiled_func(*args, **kwargs):
            profile = cProfile.Profile()
            try:
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                return result
            finally:
                profile.print_stats()
        return profiled_func

def timefunc(f):
    def f_timer(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        print f.__name__, 'took', end - start, 'time'
        return result
    return f_timer

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

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

        @param grids: The object containing the grids
        @type grids: Vibrations/Grid
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

        self.integrals = {}  # dictionary storing the precomputed 1-mode integrals, {(mode,lstate,rstate):val}

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
                                #pass

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
                    #print i,j,tmpv2
                    tmp += tmpv2
        elif self.maxpot == 3:
            for j in xrange(self.nmodes):
                if j != i:
                    tmpv2 = self._v2_integral(i, j, n[i], n[j], m[i], m[j])
                    #print i,j,tmpv2
                    tmp += tmpv2
                    for k in xrange(j+1,self.nmodes):
                        if k != j and k != i:
                            tmpv3 = self._v3_integral(i, j, k, n[i], n[j], n[k], m[i], m[j], m[k])
                            #print i,j,k,tmpv3
                            tmp += tmpv3
        elif self.maxpot == 4:
            for j in xrange(self.nmodes):
                if j != i:
                    tmpv2 = self._v2_integral(i, j, n[i], n[j], m[i], m[j])
                    #print i,j,tmpv2
                    tmp += tmpv2
                    for k in xrange(j+1,self.nmodes):
                        if k != j and k != i:
                            tmpv3 = self._v3_integral(i, j, k, n[i], n[j], n[k], m[i], m[j], m[k])
                            #print i,j,k,tmpv3
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

        indices = [ind for ind, e in enumerate([x != y for x, y in zip(n, m)]) if e is True]
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

        indices = [ind for ind, e in enumerate([x != y for x, y in zip(n, m)]) if e is True]
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
        
        indices = [ind for ind, e in enumerate([x != y for x, y in zip(n, m)]) if e is True]
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
        return sum([x != y for (x, y) in zip(c[0], c[1])])

    def solve_old(self):
        """
        Runs the VCI calculations
        """
        import itertools

        if len(self.states) == 0:
            print Misc.fancy_box('No CI space defined, by default singles space will be used')
            self.generate_states()

        # generate combination of states
        comb = [x for x in itertools.combinations_with_replacement(self.states, 2)]

        ncomb = len(comb)

        print Misc.fancy_box('Number of combinations : '+str(ncomb))

        # hamiltonian

        hamiltonian = np.zeros((len(self.states), len(self.states)))
        counter = 0

        for i in xrange(ncomb):
            counter += 1

            tmp = 0.0  # matrix element

            n = comb[i][0]  # left state
            m = comb[i][1]  # right state
            print 'Working on configuration %i out of %i' % (i+1, ncomb)
            print ' < %s | H | %s >' % (str(n), str(m))

            # 1-mode integrals
            for j in xrange(self.nmodes):
                tmpovrlp = 1.0
                tmpv1 = 0.0
                tmpt = 0.0

                for k in xrange(self.nmodes):

                    if k == j:
                        tmpv1 = self._v1_integral(j, n[j], m[j])
                        tmpt = self._kinetic_integral(j, n[j], m[j])

                    else:
                        if n[k] == m[k]:
                            s = self._ovrlp_integral(k, n[k], n[k])
                            tmpovrlp *= 1.0

                        else:
                            tmpovrlp = 0.0

                tmpv1 *= tmpovrlp
                tmpt *= tmpovrlp

                tmp += tmpv1 + tmpt
            # 2-mode integrals
            for j in xrange(self.nmodes):

                for k in xrange(j+1, self.nmodes):
                    tmpovrlp = 1.0
                    for l in xrange(self.nmodes):

                        if l != j and l != k:
                            if n[l] == m[l]:

                                #tmpovrlp *= self._ovrlp_integral(l, n[l], n[l])
                                tmpovrlp *= 1.0

                            else:
                                tmpovrlp = 0.0

                    if abs(tmpovrlp) > 1e-6:
                        tmpv2 = self._v2_integral(j, k, n[j], n[k], m[j], m[k])
                        tmpv2 *= tmpovrlp
                        #print j,k,tmpv2
                        tmp += tmpv2

            # 3-mode integrals
            if self.maxpot > 2:
                for j in xrange(self.nmodes):
                    for k in xrange(j+1, self.nmodes):
                        for l in xrange(k+1, self.nmodes):
                            tmpovrlp = 1.0
                            for o in xrange(self.nmodes):
                                if o != j and o != k and o != l:
                                    if n[o] == m[o]:
                                        #tmpovrlp *= self._ovrlp_integral(o, n[o], n[o])
                                        tmpovrlp *= 1.0
                                    else:
                                        tmpovrlp = 0.0
                            if abs(tmpovrlp) > 1e-6:
                                tmpv3 = self._v3_integral(j, k, l, n[j], n[k], n[l], m[j], m[k], m[l])
                                tmpv3 *= tmpovrlp
                                #print j,k,l,tmpv3
                                tmp += tmpv3

            nind = self.states.index(n)  # find the left state in the states vector
            mind = self.states.index(m)  # fin the right state
            hamiltonian[nind, mind] = tmp

            print 'Step %i/%i done, value %f stored' % (counter, ncomb, tmp)

        print Misc.fancy_box('Hamiltonian matrix constructed. Diagonalization...')
        w, v = np.linalg.eigh(hamiltonian, UPLO='U')

        self.energies = w
        self.vectors = v
        wcm = w / Misc.cm_in_au
        self.energiesrcm = wcm  # energies in reciprocal cm
        print 'State %15s %15s %15s' % ('Contrib', 'E /cm^-1', 'DE /cm^-1')
        for i in xrange(len(self.states)):
            print "%s %10.4f %10.4f %10.4f" % (self.states[(v[:, i]**2).argmax()],
                                               (v[:, i]**2).max(), wcm[i], wcm[i]-wcm[0])

        self.solved = True
        self.H = hamiltonian.copy()

    def print_results(self, which=1, maxfreq=4000):
        """
        Prints VCI results, can be limited to the states mostly contributed from given type of transitions (1 - singles,
        etc.), and to the maximal energy (usually 4000cm^-1 is the range of interest)
        :param which: transitions to which states should be included, 1 for singles, 2 for SD, etc.
        :param maxfreq: frequency threshold
        :return: void
        """
        if self.solved:
            print Misc.fancy_box('Results of the VCI')
            print 'State %15s %15s %15s' % ('Contrib', 'E /cm^-1', 'DE /cm^-1')
            for i in xrange(len(self.energies)):  # was self.states, is self.energies
                state = self.states[(self.vectors[:, i]**2).argmax()]
                en = self.energiesrcm[i] - self.energiesrcm[0]
                if sum([x > 0 for x in state]) < which+1 and sum(state) < which+1:
                    if en < maxfreq:
                        print "%s %10.4f %10.4f %10.4f" % (state, (self.vectors[:, i]**2).max(), self.energiesrcm[i],
                                                   en)

        else:
            print Misc.fancy_box('Solve the VCI first')

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
                state = self.states[(self.vectors[:, i]**2).argmax()]
                #state = self.states[i]
                en = self.energiesrcm[i] - self.energiesrcm[0]
                if sum([x > 0 for x in state]) < which+1:
                    if en < maxfreq:
                        contrsum = 0.0
                        print "%i %s %10.4f %10.4f %10.4f" % (i, state, (self.vectors[:, i]**2).max(), self.energiesrcm[i],
                                                   en)
                        for j,contr in enumerate(self.vectors[:, i]**2):
                            
                            if contr >= mincon:
                                print "   %s %10.4f" %(self.states[j],contr)
                                contrsum += contr
                        print 15*' '+"%10.4f" %contrsum

                                
    def print_states(self):
        """
        Prints the vibrational states used in the VCI calculations
        """
        print ''
        print Misc.fancy_box('CI Space')
        print self.states

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
        self.states = res

        self.nmax = nexc
        self.smax = smax

        # generate combination of states
        #self.combinations = [x for x in itertools.combinations_with_replacement(self.states, 2)]

        # filter the combinations to take into account the maximal dimensionality of the potentials
        #self.filter_combinations()

    def generate_states(self, maxexc=1):
        self.generate_states_nmax(maxexc, maxexc)
        self.nmax = maxexc
        self.smax = maxexc

    def generate_states_old(self, maxexc=1):
        """
        Generates the states for the VCI calcualtions

        @param maxexc: Maximal excitation quanta, 1 -- Singles, 2 -- Doubles, etc.
        @type maxexc: Integer
        """
        import itertools

        if maxexc > 4:
            raise Exception('At most quadruple excitations supported')

        states = []
        gs = [0] * self.nmodes
        states.append(gs)

        for i in xrange(self.nmodes):

            vec = [0] * self.nmodes
            vec[i] = 1
            states.append(vec)

            if maxexc > 1:
                vec = [0] * self.nmodes
                vec[i] = 2
                states.append(vec)

                for j in xrange(i+1, self.nmodes):
                    vec = [0] * self.nmodes
                    vec[i] = 1
                    vec[j] = 1
                    states.append(vec)

        if maxexc > 2:

            for i in xrange(self.nmodes):
                vec = [0]*self.nmodes
                vec[i] = 3
                states.append(vec)

                for j in xrange(i+1, self.nmodes):
                    vec = [0]*self.nmodes
                    vec[i] = 2
                    vec[j] = 1
                    states.append(vec)
                    vec = [0]*self.nmodes
                    vec[i] = 1
                    vec[j] = 2
                    states.append(vec)

                    for k in xrange(j+1, self.nmodes):
                        vec = [0]*self.nmodes
                        vec[i] = 1
                        vec[j] = 1
                        vec[k] = 1
                        states.append(vec)
        if maxexc > 3:

            for i in xrange(self.nmodes):
                vec = [0] * self.nmodes
                vec[i] = 4
                states.append(vec)

                for j in xrange(i+1, self.nmodes):
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

                    for k in xrange(j+1, self.nmodes):
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

                        for l in xrange(k+1, self.nmodes):
                            vec = [0]*self.nmodes
                            vec[i] = 1
                            vec[j] = 1
                            vec[k] = 1
                            vec[l] = 1
                            states.append(vec)

        self.states = states
        self.combinations = [x for x in itertools.combinations_with_replacement(self.states, 2)]
        self.filter_combinations()

    def filter_combinations(self):
        """
        Filters out the combinations (transitions) that do not contribute due to the max potential dimensionality

        """
        #res = []
        #for c in combinations:
            #print c, sum([x!=y for (x,y) in zip(c[0], c[1])])
            #if sum ([x!=y for (x,y) in zip(c[0],c[1])]) < maxp+1:
                #res.append(c)
        if self.combinations:
            res = [c for c in self.combinations if sum([x != y for (x, y) in zip(c[0], c[1])]) < self.maxpot+1]
            self.combinations = res

    def combgenerator(self):
        """
        Generator returning combinations of states that contribute due to available potentials
        """
        nstates = len(self.states)
        for i in xrange(nstates):
            for j in xrange(i, nstates):
                if sum([x != y for (x, y) in zip(self.states[i], self.states[j])]) < self.maxpot+1:
                    yield (self.states[i], self.states[j])
    
    def combgenerator_nofilter(self):
        """
        Generates combinations of states without prescreening
        """
        nstates = len(self.states)
        for i in xrange(nstates):
            for j in xrange(i, nstates):
                yield (self.states[i], self.states[j])

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

        # TODO
        # 1. Check size of the data etc.

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
                    if abs(ci) > 1e-8 and abs(cf) > 1e-8:
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
                                        #tmpovrlp *= (self.dx[k] * self.wfns[k, kistate] * self.wfns[k, kfstate]).sum()
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
                                                #tmpovrlp *= (self.dx[n] * self.wfns[n,
                                                            # nistate] * self.wfns[n, nfstate]).sum()
                                                tmpovrlp *= 1.0

                                            else:
                                                tmpovrlp = 0.0

                                    tmptm += tmpd2 * tmpovrlp
                        totaltm += tmptm * ci * cf
            factor = 2.5048
            intens = (totaltm[0]**2 + totaltm[1]**2 + totaltm[2]**2) * factor * (self.energiesrcm[i]
                                                                                 - self.energiesrcm[0])
            self.intensities[i] = intens
            print '%7.1f %7.1f' % (self.energiesrcm[i] - self.energiesrcm[0], intens)


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
        elif len(pols) == 2:
            print 'Two sets of dipole moments given.'
            self.pol1 = pols[0]
            self.pol2 = pols[1]
            self.maxpol = 2
        elif len(pols) > 2:
            print 'More than two sets of dipole moments given, only the two first will be used'
            self.pol1 = pols[0]
            self.pol2 = pols[1]
            self.maxpol = 2

        # TODO
        # 1. Check size of the data etc.

        self.intensities = np.zeros(len(self.states))

        # assuming that the first state is a ground state
        totalpol = np.zeros(6)
        tmptm = np.zeros(6)
        tmpp1 = np.zeros(6)
        tmpp2 = np.zeros(6)
        nstates = len(self.states)

        for i in xrange(1, nstates):
            totalpol *= 0.0
            for istate in xrange(nstates):
                ci = self.vectors[istate, 0]  # initial state's coefficient

                for fstate in xrange(nstates):
                    cf = self.vectors[fstate, i]  # final state's coefficient

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
            #print a2,g2
            f = self.energies[i] - self.energies[0]
            intens = 2 * f *  (45.0 * a2 + 7.0 * g2)
            self.intensities[i] = intens
            print '%7.1f %7.1f' % (self.energiesrcm[i] - self.energiesrcm[0], intens)

    def _v1_integral(self, mode, lstate, rstate):  # calculate integral of type < mode(lstate) | V1 | mode(rstate) >
        ind = self.v1_indices.index(mode)

        try:
            s1 = self.integrals[(mode,lstate,rstate)]
        except:
            s1 = (self.dx[mode] * self.wfns[mode, lstate] * self.wfns[mode, rstate])
            self.integrals[(mode,lstate,rstate)] = s1

        #s = (self.dx[mode] * self.wfns[mode, lstate] * self.wfns[mode, rstate] * self.v1_data[ind]).sum()
        s = (s1 * self.v1_data[ind]).sum()
        #s = (self.dx[mode] * self.wfns[mode, lstate] * self.wfns[mode, rstate] * self.v1[[mode]]).sum()

        return s
   
    def v1_integral(self,mode,lstate,rstate):
        return self._v1_integral(mode,lstate,rstate)

    def _v2_integral_old(self, mode1, mode2, lstate1, lstate2, rstate1, rstate2):
     # < mode1(lstate1) mode2(lstate2) | V2 | mode1(rstate1),mode2(rstate2)>

        s = 0.0

        if (mode1, mode2) in self.v2_indices or (mode2, mode1) in self.v2_indices:
            try:
                ind = self.v2_indices.index((mode1, mode2))
            except:
                ind = self.v2_indices.index((mode2, mode1))

            for i in xrange(self.ngrid):
                si = self.dx[mode1] * self.wfns[mode1, lstate1, i] * self.wfns[mode1, rstate1, i]

                for j in xrange(self.ngrid):

                    sj = self.dx[mode2] * self.wfns[mode2, lstate2, j] * self.wfns[mode2, rstate2, j]
                    s += si * sj * self.v2_data[ind][i, j]

            if s > 1e-6:
                print s

        return s

    def _v2_integral_new(self, mode1, mode2, lstate1, lstate2, rstate1, rstate2):
        s = 0.0

        if (mode1, mode2) in self.v2_indices or (mode2, mode1) in self.v2_indices:
            try:
                ind = self.v2_indices.index((mode1, mode2))
            except:
                ind = self.v2_indices.index((mode2, mode1))
            if mode1 < mode2: 
                try:
                    s1 = self.integrals[(mode1,lstate1,rstate1)]
                except:
                    s1 = (self.dx[mode1] * self.wfns[mode1, lstate1] * self.wfns[mode1, rstate1])
                    self.integrals[(mode1,lstate1,rstate1)] = s1
                try:
                    s2 = self.integrals[(mode2,lstate2,rstate2)]
                except:
                    s2 = (self.dx[mode2] * self.wfns[mode2, lstate2] * self.wfns[mode2, rstate2])
                    self.integrals[(mode2,lstate2,rstate2)] = s2
                
            
                #s1 = (self.dx[mode1] * self.wfns[mode1, lstate1] * self.wfns[mode1, rstate1]).transpose()
                #s2 = (self.dx[mode2] * self.wfns[mode2, lstate2] * self.wfns[mode2, rstate2])
                #return np.einsum('i,j,ij',s1,s2,self.v2_data[ind])
                s1 = s1.transpose()
                s = (s1.dot(self.v2_data[ind]).dot(s2)).sum()
                #s = (s1.dot(self.v2[mode1,mode2]).dot(s2)).sum()
            else:
                try:
                    s1 = self.integrals[(mode1,lstate1,rstate1)]
                except:
                    s1 = (self.dx[mode1] * self.wfns[mode1, lstate1] * self.wfns[mode1, rstate1])
                    self.integrals[(mode1,lstate1,rstate1)] = s1
                try:
                    s2 = self.integrals[(mode2,lstate2,rstate2)]
                except:
                    s2 = (self.dx[mode2] * self.wfns[mode2, lstate2] * self.wfns[mode2, rstate2])
                    self.integrals[(mode2,lstate2,rstate2)] = s2
                #s1 = (self.dx[mode1] * self.wfns[mode1, lstate1] * self.wfns[mode1, rstate1]).transpose()
                #s2 = (self.dx[mode2] * self.wfns[mode2, lstate2] * self.wfns[mode2, rstate2])
                #return np.einsum('i,j,ji',s1,s2,self.v2_data[ind])
                s1 = s1.transpose()
                s = (s1.dot(self.v2_data[ind].transpose()).dot(s2)).sum()
                #s = (s1.dot(self.v2[mode1, mode2]).dot(s2)).sum()


        return s

    def _v2_integral(self, mode1, mode2, lstate1, lstate2, rstate1, rstate2):
        return self._v2_integral_new(mode1, mode2, lstate1, lstate2, rstate1, rstate2)
        #return self._v2_integral_old(mode1, mode2, lstate1, lstate2, rstate1, rstate2)

    def v2_integral(self, mode1, mode2, lstate1, lstate2, rstate1, rstate2):
        return self._v2_integral_new(mode1, mode2, lstate1, lstate2, rstate1, rstate2)
    
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
#       elif (mode1, mode3, mode2) in self.v3_indices:
#           ind = self.v3_indices.index((mode1, mode3, mode2))
#       elif (mode2, mode1, mode3) in self.v3_indices:
#           ind = self.v3_indices.index((mode2, mode1, mode3))
#       elif (mode2, mode3, mode1) in self.v3_indices:
#           ind = self.v3_indices.index((mode2, mode3, mode1))
#       elif (mode3, mode1, mode2) in self.v3_indices:
#           ind = self.v3_indices.index((mode3, mode1, mode2))
#       elif (mode3, mode2, mode1) in self.v3_indices:
#           ind = self.v3_indices.index((mode3, mode2, mode1))
        else:
            return 0.0
        
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

        #si = self.dx[mode1] * self.wfns[mode1,lstate1] * self.wfns[mode1,rstate1]
        #sj = self.dx[mode2] * self.wfns[mode2,lstate2] * self.wfns[mode2,rstate2]
        #sk = self.dx[mode3] * self.wfns[mode3,lstate3] * self.wfns[mode3,rstate3]

        return np.einsum('i,j,k,ijk',si,sj,sk,self.v3_data[ind])  # einstein summation rules!


        for i in xrange(self.ngrid):
            si = self.dx[mode1] * self.wfns[mode1, lstate1, i] * self.wfns[mode1, rstate1, i]

            for j in xrange(self.ngrid):
                sj = self.dx[mode2] * self.wfns[mode2, lstate2, j] * self.wfns[mode2, rstate2, j]

                for k in xrange(self.ngrid):
                    sk = self.dx[mode3] * self.wfns[mode3, lstate3, k] * self.wfns[mode3, rstate3, k]
                    s += si * sj * sk * self.v3_data[ind][i, j, k]
                    #s += si * sj * sk * self.v3[mode1,mode2,mode3,i,j,k]


        return s

    def _v4_integral(self, mode1, mode2, mode3, mode4, lstate1, lstate2, lstate3,
                     lstate4, rstate1, rstate2, rstate3, rstate4):

        modes = list((mode1,mode2,mode3,mode4))
        lstates = list((lstate1,lstate2,lstate3,lstate4))
        rstates = list((rstate1,rstate2,rstate3,rstate4))
        ind = zip(modes,lstates,rstates)
        ind.sort()
        (modes,lstates,rstates)=zip(*ind)
        #print modes,lstates,rstates
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
        #s = []

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

        #si = self.dx[mode1] * self.wfns[mode1,lstate1] * self.wfns[mode1,rstate1]
        #sj = self.dx[mode2] * self.wfns[mode2,lstate2] * self.wfns[mode2,rstate2]
        #sk = self.dx[mode3] * self.wfns[mode3,lstate3] * self.wfns[mode3,rstate3]
        #sl = self.dx[mode4] * self.wfns[mode4,lstate4] * self.wfns[mode4,rstate4]
        #for m,l,r in ind:
        #    s.append(self.dx[m] * self.wfns[m,l] * self.wfns[m,r])
        
        #return np.einsum('i,j,k,l,ijkl',s[0],s[1],s[2],s[3],self.v4_data[potind])
        tmp = np.einsum('i,j,k,l,ijkl',si,sj,sk,sl,self.v4_data[potind])
        #print tmp
        return tmp
        #return 0.0


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

                #now calculate coefficients
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
    
    @do_cprofile
    def solve(self, parallel=False, diag='Direct'):
        """
        General solver for the VCI
        """

        if len(self.states) == 0:
            print Misc.fancy_box('No VCI states defined, by default singles will be used')
            self.generate_states()

        nstates = len(self.states)
        #ncomb = len(list(self.combgenerator()))

        print Misc.fancy_box('There are %i states') % (nstates)

        self.H = np.zeros((nstates, nstates))

        if not parallel:
            import time
            counter = 1
            for c in self.combgenerator():
                order = self.order_of_transition(c)
                print 'Point %i, of order %i' %(counter,order)
                print ' ',c 
                if order == 0:
                    tmp = self.calculate_diagonal(c)
                    print '  Diagonal element'
                elif order == 1:
                    tmp = self.calculate_single(c)
                    print '  Single transition'
                elif order == 2:
                    tmp = self.calculate_double(c)
                    print '  Double transition'
                elif order == 3 and self.maxpot > 2:
                    tmp = self.calculate_triple(c)
                    print '  Triple transition'
                elif order == 4 and self.maxpot > 3:
                    tmp = self.calculate_quadriple(c)
                    print '  Quadruple transition'
                else:
                    tmp = 0.0

                if abs(tmp) < 1e-8: 
                    tmp = 0.0
                #print 'Value %f stored' %tmp 
                nind = self.states.index(c[0])  # find the indices of the vectors
                mind = self.states.index(c[1])
                #print 'At incides ',nind,mind
                self.H[nind, mind] = tmp
                self.H[mind, nind] = tmp
                counter += 1

        else:
            #import sys, pickle
            #sys.modules['cPickle']=pickle
            #from multiprocessing import Pool
            #from pathos.multiprocessing import Pool
            #pool = Pool(maxtasksperchild=10)
            #pool = Pool()
            #results = pool.map(self,self.combinations)
            #results = pool.imap(self.calculate_transition,self.combinations,chunksize=ncomb/12)
            #pool.close()
            #pool.join()
            #results = list(results)
            import dill
            import time
            import pathos.multiprocessing as mp
            ncores = 12
            pool = mp.ProcessingPool(nodes=ncores)
            ntrans = sum(1 for _ in self.combgenerator())
            ch,e = divmod(ntrans,ncores*4)
            if e:
                ch += 1
            #ch = 85
            print 'Ncores:' ,ncores
            print 'Chunksize: ',ch
            print 'Transitions: ',ntrans
            results =  pool.map(self.calculate_transition, self.combgenerator(),chunksize=ch)
            for r in results:
                self.H[r[0],r[1]] = r[2]
                self.H[r[1],r[0]] = r[2]

        if diag=='Direct':
            print Misc.fancy_box('Hamiltonian matrix constructed. Diagonalization...')
            w, v = np.linalg.eigh(self.H, UPLO='U')
            #w, v = np.linalg.eig(self.H)
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

    @do_cprofile
    def solve_loop(self, parallel=False):
        """
        General solver for the VCI, without transitions prescreening
        """

        if len(self.states) == 0:
            print Misc.fancy_box('No VCI states defined, by default singles will be used')
            self.generate_states()

        nstates = len(self.states)

        print Misc.fancy_box('There are %i states') % (nstates)

        self.H = np.zeros((nstates, nstates))

        if not parallel:

            for i in xrange(nstates):
                for j in xrange(i,nstates):
                    c = (self.states[i],self.states[j])
                    order = self.order_of_transition(c)

                    if order == 0:
                        tmp = self.calculate_diagonal(c)
                        self.H[i,j] = tmp
                    elif order == 1:
                        tmp = self.calculate_single(c)
                        self.H[i,j] = tmp
                    elif order == 2:
                        tmp = self.calculate_double(c)
                        self.H[i,j] = tmp
                    elif order == 3 and self.maxpot > 2:
                        tmp = self.calculate_triple(c)
                        self.H[i,j] = tmp
                    elif order == 4 and self.maxpot > 3:
                        tmp = self.calculate_quadriple(c)
                        self.H[i,j] = tmp

        else:
            import dill
            import time
            import pathos.multiprocessing as mp
            ncores = 12
            pool = mp.ProcessingPool(nodes=ncores)
            ntrans = nstates * (nstates-1) / 2
            ch,e = divmod(ntrans,ncores*4)
            if e:
                ch += 1
            #ch = 85
            print 'Ncores:' ,ncores
            print 'Chunksize: ',ch
            print 'Transitions: ',ntrans
            results =  pool.map(self.calculate_transition, self.combgenerator_nofilter(),chunksize=ch)
            for r in results:
                self.H[r[0],r[1]] = r[2]

        np.save('Hessian.npy',self.H)
        print Misc.fancy_box('Hamiltonian matrix constructed. Diagonalization...')
        w, v = np.linalg.eigh(self.H, UPLO='U')

        self.energies = w
        self.vectors = v
        self.energiesrcm = self.energies / Misc.cm_in_au
        self.solved = True
        self.print_results()

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


