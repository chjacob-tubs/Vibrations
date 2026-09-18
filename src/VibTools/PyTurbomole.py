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

import numpy
import math

from . import Constants


from .Molecule import VibToolsMolecule
from .Modes import VibModes
from .Results import Results


class AOForceOutputFile:

    def __init__(self, filename='aoforce.out'):
        """
        AOForceOutputFile constructor.
        """
        self.filename = filename
        self.modes = None

    def read(self, mol):
        natoms = mol.natoms
        self.modes = VibModes(3*natoms, mol)

        f = open(self.filename, 'r')
        lines = f.readlines()
        f.close()

        start = -1
        end = -1
        for i, l in enumerate(lines):
            if 'NORMAL MODES and VIBRATIONAL FREQUENCIES' in l:
                start = i
            elif (start > -1) and ('zero point VIBRATIONAL energy' in l):
                end = i
                break

        if start == -1:
            raise Exception('dipole gradients not found in control file')

        lines = lines[start+1:end]

        freqlines = [L[14:] for L in lines
                     if L.strip().startswith('frequency')]
        freqs = ' '.join(freqlines).replace('i', '-')
        freqs = numpy.array([float(f) for f in freqs.split()])
        self.modes.set_freqs(freqs)

        masslines = [L[22:] for L in lines
                     if L.strip().startswith('reduced mass')]
        redmass = ' '.join(masslines)
        redmass = numpy.array([float(f) for f in redmass.split()])

        lines = [L[18:] for L in lines]
        lines = [L[2:] for L in lines if L.startswith('x')
                 or L.startswith('y') or L.startswith('z')]

        normalmodes = numpy.zeros((3*natoms, 3*natoms))

        ncol = len(lines[0].split())
        icol = -ncol
        for i, l in enumerate(lines):
            irow = i % (3*natoms)

            if irow == 0:
                icol += ncol
            else:
                line = [float(fl) for fl in l.split()]
                normalmodes[icol:icol+ncol, irow] = line

        self.modes.set_modes_c(normalmodes)


class TMControlFile:

    def __init__(self, filename='control'):
        """
        TMControlFile constructor.
        """
        self.filename = filename
        self.dipgrad = None
        self.hessian = None

    def read(self, mol):
        natoms = mol.natoms
        self.dipgrad = numpy.zeros((natoms, 3, 3))

        f = open(self.filename, 'r')
        lines = f.readlines()
        f.close()

        start = -1
        end = -1
        for i, l in enumerate(lines):
            if l.startswith('$dipgrad') and ('cartesian dipole gradients'
                                             in l):
                start = i
            elif (start > -1) and l.startswith('$'):
                end = i
                break

        if start == -1:
            raise Exception('dipole gradients not found in control file')

        dipgrad = ' '.join(lines[start+1:end]).replace('D', 'E').\
                  replace('d', 'E').split()
        self.dipgrad = numpy.array([float(d) for d in dipgrad])
        self.dipgrad.shape = (natoms, 3, 3)

    def read_hessian(self, mol):
        natoms = mol.natoms
        self.hessian = numpy.zeros((natoms*3, natoms*3))
        self.modes = VibModes(3*natoms, mol)

        f = open(self.filename, 'r')
        lines = f.readlines()
        f.close()

        start = -1
        end = -1
        for i, l in enumerate(lines):
            if l.startswith('$hessian'):
                start = i
            elif (start > -1) and l.startswith('$'):
                end = i
                break

        if start == -1:
            raise Exception('dipole gradients not found in control file')

        lines = lines[start+1:end]
        ncol = len(lines[0].split())-2
        lines = [L[6:] for L in lines]

        nrow = 3*natoms // ncol
        if 3*natoms % ncol:
            nrow = nrow+1

        for i in range(3*natoms):
            row = ' '.join(lines[i*nrow:(i+1)*nrow])
            row = numpy.array([float(f) for f in row.split()])

            self.hessian[i, :] = row

        # mass-weight Hessian
        for i, atmass in enumerate(mol.atmasses):
            self.hessian[:, 3*i:3*i+3] = (self.hessian[:, 3*i:3*i+3]
                                          / math.sqrt(atmass))
            self.hessian[3*i:3*i+3, :] = (self.hessian[3*i:3*i+3, :]
                                          / math.sqrt(atmass))

        evals, evecs = numpy.linalg.eigh(self.hessian)

        # convert to cm^-1 [evals are in Hartree / (Bohr**2 * amu) ]
        evals = evals * Constants.Hartree_in_Joule / \
            (Constants.Bohr_in_Meter**2 * Constants.amu_in_kg)
        # in J/(m*m*kg) = 1/s**2

        for i in range(evals.size):
            if evals[i] > 0.0:
                evals[i] = math.sqrt(evals[i])
            else:
                evals[i] = -math.sqrt(-evals[i])

        evals = evals / (2.0*math.pi)  # frequency in 1/s
        evals = evals / Constants.cvel_ms * 1e-2  # wavenumber in 1/cm

        self.modes.set_freqs(evals)
        self.modes.set_modes_mw(evecs.transpose())


class TurbomoleResults(Results):

    def __init__(self, controlfile='control',
                 coordfile='coord', outname=None):
        """
        TurbomoleResults constructor.
        """
        self.mol = VibToolsMolecule()
        self.coordfile = coordfile

        if outname:
            print("WARNING: normal modes are read from aoforce output file")

            self.aoforceoutput = AOForceOutputFile(filename=outname)
        else:
            self.aoforceoutput = None
        self.controlfile = TMControlFile(filename=controlfile)

    def _get_modes(self):
        if self.aoforceoutput:
            return self.aoforceoutput.modes
        else:
            return self.controlfile.modes

    modes = property(_get_modes)

    def read(self):
        self.mol.read_from_coord(filename=self.coordfile)

        self.controlfile.read(self.mol)
        if self.aoforceoutput:
            self.aoforceoutput.read(self.mol)
        else:
            self.controlfile.read_hessian(self.mol)

    def get_tensor_deriv_c(self, tens, ncomp=None):
        if tens == 'dipole':
            return self.controlfile.dipgrad
        else:
            raise Exception('Not implemented for Turbomole')
