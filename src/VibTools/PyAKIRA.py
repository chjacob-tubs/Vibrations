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

from .Molecule import VibToolsMolecule
from .Modes import VibModes
from .Results import Results


class AKIRAResults(Results):

    def __init__(self, iterationsfile='akira_iterations.out',
                 coordfile='coord'):
        """
        AKIRAResults constructor.
        """
        self.mol = VibToolsMolecule()

        self.iterationsfile = iterationsfile
        self.coordfile = coordfile

        self.modes = None
        self.irints = None

    def read(self):
        import numpy
        import re

        self.mol.read_from_coord(filename=self.coordfile)

        # read akira iterations
        f = open(self.iterationsfile)
        lines = f.readlines()
        f.close()
        for i, lin in enumerate(lines):
            if re.search('Final results from module Davidson', lin):
                start = i
        nbas = int(lines[start+3].split()[-1])
        lines = lines[start+8:start+8+nbas]
        modes = []
        for lin in lines:
            spl = lin.split()
            if spl[10] == 'YES':
                modes.append([float(spl[2]), float(spl[8])])

        self.modes = VibModes(len(modes), self.mol)
        self.modes.set_freqs(numpy.array([m[0] for m in modes]))
        self.irints = numpy.array([m[1] for m in modes])

    def get_ir_intensity(self):
        return self.irints
