"""
The module containing the classes related to hypersurfaces: potential energy surface, dipole moment surface, and other
properties surfaces.
"""

import numpy as np
import Misc


class Surface:
    """
    General class containing and manipulating some property surface
    """

    def __init__(self, grids=None, order=1, prop=(1,)):
        """
        An object is initialized with existing grids, the order of the surface (how many modes are involved), and
        the shape of the property: (1,) for energy, (3,) for dipole moment, (6,) for polarizability

        @param grids: Vibrations grid object
        @param order: Order (or dimensionality) of the surface
        @type order: Integer
        @param prop: Shape of the property, e.g. (1,) for energy
        @type prop: Tuple of integer
        """

        self.grids = grids

        if self.grids is None:  # an empty object
            self.empty = True
            self.ngrid = 0

        else:
            import copy
            self.empty = False
            self.ngrid = grids.ngrid
            self.grids = copy.copy(grids)

        self.order = order
        self.prop = prop

        self.indices = []  # indices of the modes corresponding to the data stored in self.data
        self.data = []  # data (potential, dipole moment, etc.)

    def __str__(self):
        s = ''
        if self.empty:
            s += '!Empty surface\n'

        s += 'Order (how many modes involved): ' + str(self.order) + '\n'
        s += 'Shape of the property: ' + str(self.prop) + '\n'
        return s

    def delete(self, *lind):
        """
        Deleting a surface of given index (ind)
        
        @param ind: List of tuples of modes
        """
        for ind in lind:
            type(ind)
            ind = list(ind)
            ind.sort()
            try:
                i = self.indices.index(tuple(ind))
                self.indices.pop(i)
                self.data.pop(i)
            except:
                print 'Surface of index ',ind,' not found'

    def zero(self):
        """
        Zeroing all surfaces
        """
        for i,e in enumerate(self.data):
            self.data[i] *=0.0

    def __getitem__(self, item):
        """
        Gets a surface/element for given indices, Nindices == 2*Nmodes => for a given grid point
        :param item: Indices of the modes
        :return: Property for given indices
        """
        if len(item) == self.order:
            ind = list(item)
            newind = ind[:]
            newind.sort() #  the data in the object is sorted according to the indices
            #whichelement = self.indices.index(tuple(newind))
            #print whichelement
            try:
                #return np.transpose(self.data[self.indices.index(tuple(newind))],sorted(range(len(ind)), key=lambda k: ind[k]))
                return np.transpose(self.data[self.indices.index(tuple(newind))],axes=(tuple([newind.index(i) for i in ind])))
            except:
                raise Exception('Surface not found')

        elif len(item) == 2 * self.order:
            ind = list(item)
            newind = zip(ind[:self.order],ind[self.order:])
            newind.sort()
            newind2 = [x[0] for x in newind] + [x[1] for x in newind]
            modes = newind2[:self.order]
            points = newind2[self.order:]
            try:
                #return  self.data[self.indices.index(tuple(newind[:self.order]))][tuple(newind[self.order:])]
                return  self.data[self.indices.index(tuple(modes))][tuple(points)]
            except:
                raise Exception('Point not found')
        else:
            pass



class Potential(Surface):

    def __init__(self, grids=None, order=1):
        Surface.__init__(self, grids, order, prop=(1,))
        self.index = 0  # will be used for the iterator

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.index < len(self.indices)
        except IndexError:
            raise StopIteration
        self.index += 1
        res = self.indices[self.index][:]
        res.append(self.data[self.index])
        return tuple(res)


    def read_np(self, fname):
        """
        Reads in the existing potential energy surface from a NumPy formatted binary file *.npy

        @param fname: File name, without extension
        @type fname: String
        """

        tmparray = np.load(fname)

        if len(tmparray.shape) == 2:
            if self.order != 1:
                raise Exception('Shape mismatch')

        elif len(tmparray.shape) == 4:
            if self.order != 2:
                raise Exception('Shape mismatch')

        elif len(tmparray.shape) == 6:
            if self.order != 3:
                raise Exception('Shape mismatch')
                
        elif len(tmparray.shape) == 8:
            if self.order != 4:
                raise Exception('Shape mismatch')
        else:
            raise Exception('Input data shape mismatch, check shape of stored arrays')




        if self.order == 1:
            for i in range(tmparray.shape[0]):
                self.indices.append(i)
                self.data.append(tmparray[i,:])

        elif self.order == 2:
            for i in range(tmparray.shape[0]):
                for j in range(i+1,tmparray.shape[0]):
                    self.indices.append((i,j))
                    self.data.append(tmparray[i,j,:,:])
        
        elif self.order == 3:
            for i in range(tmparray.shape[0]):
                for j in range(i+1, tmparray.shape[0]):
                    for k in range(j+1, tmparray.shape[0]):
                        if not np.all(tmparray[i,j,k,:,:,:]==0.0):
                            self.indices.append((i,j,k))
                            self.data.append(tmparray[i,j,k,:,:,:])

        elif self.order == 4:
            nmodes = tmparray.shape[0]
            for i in range(nmodes):
                for j in range(i+1, nmodes):
                    for k in range(j+1, nmodes):
                        for l in range(k+1, nmodes):
                            if not np.all(tmparray[i,j,k,l,:,:,:,:] == 0.0):
                                self.indices.append((i,j,k,l))
                                self.data.append(tmparray[i,j,k,l,:,:,:,:])

    def save(self, fname='pot.npy'):
        """
        Saving the surface to npy file
        """

        shape = [self.grids.nmodes] * self.order + [self.ngrid] * self.order
        shape = tuple(shape)
        print shape
        tmparray = np.zeros(shape)

        for i,ind in enumerate(self.indices):
            tmparray[ind] = self.data[i]

        np.save(fname,tmparray)

    def generate_harmonic(self, cmat=None):

        if not self.empty:
            if cmat is None:
                if self.order == 1:
                    for i in range(self.grids.nmodes):
                        self.indices.append(i)
                        potential = (self.grids.grids[i] ** 2 * (self.grids.modes.freqs[i] *
                                                             Misc.cm_in_au)**2) / 2.0
                        self.data.append(potential)
            else:
                if self.order == 1:
                    for i in range(self.grids.nmodes):
                        self.indices.append(i)
                        potential = (self.grids.grids[i] ** 2 * cmat[i, i]) / 2.0
                        self.data.append(potential)

                elif self.order == 2:
                    for i in range(self.grids.nmodes):
                        for j in range(i+1, self.grids.nmodes):
                            if cmat[i,j] != 0.0:
                                self.indices.append((i,j))
                                potential = np.zeros((self.ngrid,self.ngrid))
                                for k in range(self.grids.ngrid):
                                    for l in range(self.grids.ngrid):
                                        potential[k, l] = self.grids.grids[i, k] * self.grids.grids[j, l] * cmat[i, j]
                                        potential[l, k] = potential[k, l]

                                self.data.append(potential)


class Polarizability(Surface):
    """ 
    Class for polarizability tensors. 
    """
    def __init__(self, grids=None, gauge='len', order=1):
        Surface.__init__(self, grids, order, prop=(6,))
        self.gauge = gauge

    def generate_harmonic(self, res):
        """
        Method generating harmonic polarizability surface, using its first derivative from
        harmonic calculations.
        res -- VibTools results instance
        """
        pol_deriv_nm = res.get_tensor_deriv_nm('pol'+self.gauge, ncomp=6, modes=self.grids.modes)
        print pol_deriv_nm.shape

        if not self.empty:
            if self.order == 1:
                for i in range(self.grids.nmodes):
                    pol = np.zeros((self.grids.ngrid, 6))
                    self.indices.append(i)
                    for j in range(6):
                        pol[:,j] = self.grids.grids[i] * pol_deriv_nm[i,j]
                    self.data.append(pol)
            else:
                raise Exception('In the harmonic approximation only 1-mode polarizability tensors are available')

class Gtensor(Surface):
    
    def __init__(self,grids=None, gauge='vel', order=1):
        Surface.__init__(self, grids, order, prop=(9,))
        self.gauge = gauge

    def generate_harmonic(self, res):

        gten_deriv_nm = res.get_tensor_deriv_nm('gten'+self.gauge, modes=self.grids.modes)
        if not self.empty:
            if self.order == 1:
                for i in range(self.grids.nmodes):
                    gten = np.zeros((self.grids.ngrid, 9))
                    self.indices.append(i)
                    for j in range(9):
                        gten[:,j] = self.grids.grids[i] * gten_deriv_nm[i,j]
                    self.data.append(gten)
            else:
                raise Exception('In the harmonic approximation only 1-mode polarizability tensors are available')

class Atensor(Surface):
    
    def __init__(self,grids=None, order=1):
        Surface.__init__(self, grids, order, prop=(27,))

    def generate_harmonic(self, res):

        aten_deriv_nm = res.get_tensor_deriv_nm('aten', modes=self.grids.modes)
        if not self.empty:
            if self.order == 1:
                for i in range(self.grids.nmodes):
                    aten = np.zeros((self.grids.ngrid, 27))
                    self.indices.append(i)
                    for j in range(27):
                        aten[:,j] = self.grids.grids[i] * aten_deriv_nm[i,j]
                    self.data.append(aten)
            else:
                raise Exception('In the harmonic approximation only 1-mode polarizability tensors are available')

class Dipole(Surface):

    def __init__(self, grids=None, order=1):
        Surface.__init__(self, grids, order, prop=(3,))

    def generate_harmonic(self,res):
        """
        Method generating harmonic dipole moment surface, using its first derivative from harmonic calculations
        res -- the VibTools results instance
        """
        dm_deriv_nm =  res.get_tensor_deriv_nm('dipole',modes=self.grids.modes)

        if not self.empty:
            if self.order == 1:
                for i in range(self.grids.nmodes):
                    dm = np.zeros((self.grids.ngrid, 3))
                    self.indices.append(i)
                    for j in range(3):
                        dm[:,j] = self.grids.grids[i] * dm_deriv_nm[i,j]  * Misc.au_in_Debye * np.sqrt(Misc.me_in_amu)
                    self.data.append(dm)
            else:
                raise Exception('In the harmonic approximation only 1-mode dipole moments are available')

                                                

    def read_np(self, fname):
        """
        Reads in the existing dipole moment surface from a NumPy formatted binary file *.npy

        @param fname: File name
        @type fname: String
        """

        tmparray = np.load(fname)

        if len(tmparray.shape) == 3:
            if self.order != 1:
               raise Exception('Shape mismatch')

        elif len(tmparray.shape) == 5:
            if self.order != 2:
              raise Exception('Shape mismatch')
        else:
            raise Exception('Input data shape mismatch, check shape of stored arrays')




        if self.order == 1:
            for i in range(tmparray.shape[0]):
                self.indices.append(i)
                self.data.append(tmparray[i,:,:])

        elif self.order == 2:
            for i in range(tmparray.shape[0]):
                for j in range(i+1,tmparray.shape[0]):
                    self.indices.append((i,j))
                    self.data.append(tmparray[i,j,:,:,:])





class Dipole_old:
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


class Potential_old:
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
        self.pot = []
        self.indces = []


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
                        self.indces.append(i)
                        potential = (self.grids[i] ** 2 * (self.modes.freqs[i] /
                                                             Misc.cm_in_au)**2) / 2.0
                        self.pot.append(potential)
            else:
                if self.order == 1:
                    for i in range(self.nmodes):
                        self.indces.append(i)
                        potential = (self.grids[i] ** 2 * cmat[i, i]) / 2.0
                        self.pot.append(potential)

                elif self.order == 2:
                    for i in range(self.nmodes):
                        for j in range(i+1, self.nmodes):
                            self.indces.append((i,j))
                            potential = np.zeros((self.ngrid,self.ngrid))
                            for k in range(self.ngrid):
                                for l in range(self.ngrid):
                                    potential[k, l] = self.grids[i, k] * self.grids[j, l] * cmat[i, j]
                                    potential[l, k] = potential[k, l]

                            self.pot.append(potential)

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

        self.nmodes = tmparray.shape[0]


        if self.order == 1:
            self.ngrid = tmparray.shape[1]
            for i in range(self.nmodes):
                self.indces.append(i)
                self.pot.append(tmparray[i,:])

        elif self.order == 2:
            self.ngrid = tmparray.shape[3]
            for i in range(self.nmodes):
                for j in range(i+1,self.nmodes):
                    self.indces.append((i,j))
                    self.pot.append(tmparray[i,j,:,:])



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

