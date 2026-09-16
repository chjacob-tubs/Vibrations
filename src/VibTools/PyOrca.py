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


class OrcaResults(Results):

    def __init__(self):
        self.mol = VibToolsMolecule()
        self.modes = None
        self.nmodes = None
        self.natoms = None

    def get_freqs(self, output):
        # import numpy
        import re
        f = open(output)
        lines = f.readlines()
        f.close()

        for i, l in enumerate(lines):
            if re.search('vibrational_freq', l):
                start = i
            if re.search('normal_modes', l):
                end = i

        freqs = list(map(lambda L: float(L.split()[1]),
                     lines[start+2+6:end-1]))
        return freqs

    def read(self, coords, output):
        import numpy
        import re

        self.mol.read(filename=coords)
        natoms = self.mol.natoms
        # read akira iterations
        f = open(output)
        lines = f.readlines()
        f.close()

        for i, l in enumerate(lines):
            if re.search('normal_modes', l):
                start = i
            if re.search('atoms', l):
                end = i

        nmodes = natoms*3
        modes = VibModes(nmodes-6, self.mol)
        normalmodes = numpy.zeros((nmodes-6, 3*natoms))

        freqs = list(map(lambda L: float(L.split()[1]),
                     lines[start-nmodes-1+6:start-1]))

        startnum = start+3
        j = 0
        while startnum + natoms * 3 < end:
            for column in range(len(lines[startnum].split())-1):
                if j < 6:
                    j += 1
                    continue
                # mode = []
                column = column+1
                for i in range(natoms):
                    normalmodes[j-6][i*3 + 0] = lines[i*3+startnum
                                                      + 0].split()[column]
                    normalmodes[j-6][i*3 + 1] = lines[i*3+startnum
                                                      + 1].split()[column]
                    normalmodes[j-6][i*3 + 2] = lines[i*3+startnum
                                                      + 2].split()[column]
                j += 1
            startnum = startnum + natoms * 3 + 1

        freqs = numpy.asarray(freqs)
        normalmodes = numpy.asarray(normalmodes)

        modes.set_modes_c(normalmodes)
        modes.set_freqs(freqs)

        self.natoms = natoms
        self.nmodes = nmodes
        self.modes = modes
