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
Module containing classes related to property hypersurfaces: 
potential energy surface, dipole moment surface, and other
properties surfaces.
"""

import numpy as np
import Misc


class Surface(object):
    """
    Class containing and manipulating a generic property surface
    """

    def __init__(self, grids=None, order=1, prop=(1,)):
        """
        An object is initialized with existing grids, the order of the surface (how many modes are involved), and
        the shape of the property: (1,) for energy, (3,) for dipole moment, (6,) for polarizability

        @param grids: Vibrations grid object
        @param order: Order (or dimensionality) of the surface
        @type order: Integer
        @param prop: Shape of the property, e.g. (1,) for energy
        @type prop: Tuple of Integer
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
        
        @param lind: List of tuples of modes
        """
        for ind in lind:
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
        if self.order == 1:
            if type(item) is int:
                return self.data[self.indices.index(item)]
            elif len(item) == 2*self.order:
                return self.data[self.indices.index(item[0])][item[1]]

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
                #pass
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
                    if not np.all(tmparray[i,j,:,:]==0.0):
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
        Saves the PES to NumPy formatted binary file *.npy
        """

        shape = [self.grids.nmodes] * self.order + [self.ngrid] * self.order
        shape = tuple(shape)
        print shape
        tmparray = np.zeros(shape)

        for i,ind in enumerate(self.indices):
            tmparray[ind] = self.data[i]

        np.save(fname,tmparray)

    def generate_harmonic(self, cmat=None):
        """
        Generates the harmonic PES, for localized modes
        also the harmonic 2-mode potentials can be generated.
        In this case cmat corresponds to coupling matrix given in Hartree.
        """

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

    def generate_from_function(self, func):
        """
        Generates PES by calling a given function.

        The function f((i,j,..), (qi,qj,...)) has a tuple with the mode indices as
        its first argument and a tuple of the displacement values (possibly numpy
        arrays) as its second argument.

        Example for 1D:
        def func(i, qi) :
            return (qi ** 2 * freq[i]) / 2.0

        Example for 2D:
        def func(ij, qij) :
            pot = .
        """

        if not self.empty:
            if self.order == 1:
                for i in range(self.grids.nmodes):
                    self.indices.append(i)
                    potential = func(i, self.grids.grids[i])
                    self.data.append(potential)

            elif self.order == 2:
                for i in range(self.grids.nmodes):
                    for j in range(i+1, self.grids.nmodes):
                        self.indices.append((i,j))
                        #f = np.vectorize(lambda x,y: func(i,j, y,x))
                        #potential = f(self.grids.grids[i, :], self.grids.grids[j, :].reshape((self.grids.ngrid,1)))
                        potential = np.zeros((self.grids.ngrid, self.grids.ngrid))
                        for n in range(self.grids.ngrid) :
                            for m in range(self.grids.ngrid) :
                                potential[n,m] = func(i,j, self.grids.grids[i, n], self.grids.grids[j, m])
                        self.data.append(potential)


class Polarizability(Surface):
    """ 
    The polarizability tensors. 
    """
    def __init__(self, grids=None, gauge='len', order=1):
        Surface.__init__(self, grids, order, prop=(6,))
        self.gauge = gauge

    def generate_harmonic(self, res):
        """
        Generates the harmonic polarizability surface, using its first derivative from
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
    """
    The G tensor for ROA intensities
    """
    
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
    """
    The A tensor for ROA intensities
    """
    
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
        Generates the harmonic dipole moment surface, using its first derivative from harmonic calculations
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

