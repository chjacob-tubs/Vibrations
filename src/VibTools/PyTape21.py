# -*- coding: utf-8 -*-
#
# This file is part of the
# LocVib 1.3 suite of tools for the analysis for vibrational spectra.
# Copyright (C) 2009-2023 by Christoph R. Jacob and others.
#
#    LocVib is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    LocVib is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with LocVib.  If not, see <http://www.gnu.org/licenses/>.
#
# In scientific publications using the LocVib tools, please cite:
#   Ch. R. Jacob, J. Chem. Phys 130 (2009), 084106.
#
# The most recent version of LocVib is available at
#   http://www.christophjacob.eu/software

import kf
import numpy
# import math

# from .Constants import *

from .Molecule import VibToolsMolecule
from .Modes import VibModes
from .Results import Results


class ADFResults(Results):

    def __init__(self, t21file='TAPE21', xyzfile='mol.xyz'):
        """
        ADFResults constructor.
        """
        self.mol = VibToolsMolecule()
        self.xyzfilename = xyzfile
        self.t21filename = t21file

    def read(self):
        self.mol.read(filename=self.xyzfilename, filetype='xyz')

        f = kf.kffile(self.t21filename)

        natoms = f.read('Geometry', 'nr of atoms')[0]
        if not (self.mol.natoms == natoms):
            raise Exception('Inconsistent number of atoms')

        inpatm = f.read('Geometry',
                        'atom order index').reshape((natoms, 2),
                                                    order='Fortran')[:, 0]

        nmodes = natoms*3
        modes = VibModes(nmodes, self.mol)

        freqs = f.read('Freq', 'Frequencies')
        normalmodes = f.read('Freq', 'Normalmodes')
        normalmodes = normalmodes.reshape((nmodes, nmodes))

        # the normal modes in T21 are in internal order,
        # convert to input order
        normalmodes_inputorder = numpy.zeros_like(normalmodes)
        for iinp in range(natoms):
            ia = inpatm[iinp] - 1
            normalmodes_inputorder[:, 3*iinp] = normalmodes[:, 3*ia]
            normalmodes_inputorder[:, 3*iinp+1] = normalmodes[:, 3*ia+1]
            normalmodes_inputorder[:, 3*iinp+2] = normalmodes[:, 3*ia+2]

        modes.set_modes_c(normalmodes_inputorder)
        modes.set_freqs(freqs)

        self.natoms = natoms
        self.nmodes = nmodes
        self.modes = modes

        f.close()
