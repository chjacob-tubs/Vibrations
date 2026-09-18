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
"""
Importing results of VASP calculations.
"""
import numpy
import math

from . import Constants

from .Molecule import VibToolsMolecule
from .Modes import VibModes
from .Results import Results


class VASPoutput:
    """
    Class that handles VASP output (OUTCAR) file.
    """
    def __init__(self, filename='full-output.vasp'):
        """
        The constructor of VASPoutput.
        """
        self.filename = filename
        self.dipgrad = None
        self.hessian = None

# UPDATE below

    def read(self, mol):
        """
        Reading in the output file
        """
        natoms = mol.natoms
        self.dipgrad = numpy.zeros((natoms, 3, 3))
        f = open('full-output.vasp', 'r')
        line = f.readline()
        while line != '':
            if 'BORN EFFECTIVE CHARGES' in line:
                break
            line = f.readline()
        if line == '':
            raise Exception('Born effective charges not found in output file')
        line = f.readline()
        line = f.readline()
        line = f.readline()
        bec = []
        ion = []
        while line != '' and line != '\n':
            if 'ion' not in line:
                line = list(map(float, line.split()[1:]))
                ion.append(line)
                line = f.readline()
            else:
                bec.append(ion)
                ion = []
                line = f.readline()
        bec.append(ion)
        ion = []
        self.dipgrad = numpy.array(bec)
        print(self.dipgrad.shape)
        f.close()

        # dipgrad = ' '.join(lines[start+1:end])
        # .replace('D', 'E').replace('d', 'E').split()
        # self.dipgrad = numpy.array([float(d) for d in dipgrad])
        # self.dipgrad.shape = (natoms,3,3)
# up to here
    def read_hessian(self, mol):
        """
        Reading in the hessian martix.
        """
        natoms = mol.natoms
        self.hessian = numpy.zeros((natoms*3, natoms*3))
        self.modes = VibModes(3*natoms, mol)

        f = open(self.filename, 'r')
        line = f.readline()
        while line != '':
            if 'SECOND DERIVATIVES' in line:
                break
            line = f.readline()
        if line == '':
            raise Exception('hessian not found in output file')
        line = f.readline()
        line = f.readline()
        line = f.readline()
        tmp = []  # temporary list for the lines
        while line != '':
            if line != ' \n':
                tmp.append(line)
                line = f.readline()
            else:
                break
        for i, L in enumerate(tmp):
            L = L.split()
            L.pop(0)
            tmp[i] = list(map(float, L))
        self.hessian = numpy.array(tmp)
        # symmetrizing the hessian
        self.hessian = -(self.hessian+self.hessian.transpose())/2.0
        f.close()
        for i, atmass in enumerate(mol.atmasses):
            self.hessian[:, 3*i:3*i+3] = (self.hessian[:, 3*i:3*i+3] /
                                          math.sqrt(atmass))
            self.hessian[3*i:3*i+3, :] = (self.hessian[3*i:3*i+3, :] /
                                          math.sqrt(atmass))
        evals, evecs = numpy.linalg.eigh(self.hessian)
        # evals are in eV/Angs**2 *amu
        # convert to cm^-1 [evals are in Hartree / (Bohr**2 * amu) ]
        evals = (evals * Constants.eV_in_Joule
                 / (1e-10**2 * Constants.amu_in_kg))
        # in J/(m*m*kg) = 1/s**2
        # print numpy.sqrt(abs(evals))/(2.0*math.pi)/cvel_ms*1e-2
        # print '-'*30
        for i in range(evals.size):
            if evals[i] > 0.0:
                evals[i] = math.sqrt(evals[i])
            else:
                evals[i] = -math.sqrt(-evals[i])

        evals = evals / (2.0*math.pi)  # frequency in 1/s
        evals = evals / Constants.cvel_ms * 1e-2  # wavenumber in 1/cm
        # numpy.set_printoptions(precision=4)
        # print evals
        self.modes.set_freqs(evals)
        self.modes.set_modes_mw(evecs.transpose())


class VASPResults(Results):
    """
    Class that reads in and contains VASP results.
    """

    def __init__(self, output='full-output.vasp', coordfile='coord'):
        """
        Constructor for VASPResults

        All arguments are optional, leaving out an argument will
        choose default settings.

        @param coordfile: path to Turbomole formated coordinates
        file (default: pwd/coord)
        @type coordfile: str
        @param output: path to VASP output file (OUTCAR)
        (default: full-output.vasp)
        @type output: str
        """
        self.mol = VibToolsMolecule()
        self.coordfile = coordfile

        self.output = VASPoutput(filename=output)

    def _get_modes(self):
        return self.output.modes

    modes = property(_get_modes)

    def read(self):
        """
        Reading the results.
        """
        self.mol.read_from_coord(filename=self.coordfile)

        self.output.read(self.mol)
        self.output.read_hessian(self.mol)

    def get_tensor_deriv_c(self, tens, ncomp=None):
        if tens == 'dipole':
            return self.output.dipgrad
        else:
            raise Exception('Not implemented')
