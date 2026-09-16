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
This module enables a comfortable connection between SNF and LocVib/VibTools.

SNF 4.0.0, Manual (2007):

https://ethz.ch/content/dam/ethz/special-interest/chab/physical-chemistry/
reiher-dam/documents/Software/SNF_4.0.0_manual.pdf
"""
import numpy
import math

# from .Constants import *
import VibTools.Constants as Constants

from .Molecule import VibToolsMolecule
from .Modes import VibModes
from .Results import Results


class SNFRestartFile(object):
    """
    reads relevant data from the restart (SNF) file for LocVib.

    Parameters
    ----------
    n_per_line : int or None
       reads the letters up to D in the format
       string of the SNF-restart file.\n
       Example: 4D20.10 (line 2) n_per_line =  4
    nvar : int or None
       -> SNF manual: number of nuclear coordinates = number of atoms x 3.
    natoms : int or None
       total number of atoms in the molecule: floor(nvar/3).
    iaus : int or None
       order of the used difference quotient.\n
       iaus = 2 (3-point-formula)\n
       iaus = 4 (5-point-formula)\n
       ! LocVib only supports iaus = 2
    displ : float or None
       Displacement.\n
       -> SNF manual: displ(LocVib) == cstep(SNF)\n
       step size for the numerical derivatives, i.e.,\n
       distortion of the Cartesian nuclear coordinates, in [bohr].
    intonly : Bool
       If true then only intensities should be calculated.
       If False then intensities and fequencies should be calculated.
    nmodes : int
        number of modes.
    dipol : narray (natoms, 6, 3)
      gradient of dipole moments.
    gradient : narray (natoms, 6, nvar)
      gradient of Hesse-Matrix ?
    pollen : narray (natoms, 6, 8?)
      gradient of polarizability tensor
    polvel : narray (natoms, 6, 8?)
      gradient of polarizability tensor (velocity representation).
    gtenlen : narray (natoms, 6, 9?)
      gradient of G/LAO tensor
    gtenvel : narray (natoms, 6, 9?)
      gradient of G/AO tensor
    aten :  narray (natoms, 6, 27?)
       gradient of A tensor
    """
    # Discrepancies between LocVib-doc and SNF-doc:
    # :iaus: int or None
    #    -> SNF manual: iaus(LocVib) == scfconf(SNF)
    #    convergence criterion for the energy (10^{-scfconf})
    # :nmodes: int
    #    -> SNF manual: nmodes(LocVib) == naus(SNF)
    #    number of steps along each coordinate
    #    (for a n-point central differences formula
    #    for the numerical derivatives,
    #    naus= (n − 1); n must be odd)

    def __init__(self):
        """
        SNFRestartFile constructor.
        For more details see the class description/docstring
        """
        self.n_per_line = None

        self.nvar = None
        self.natoms = None
        self.iaus = None
        self.displ = None

        self.intonly = False
        self.nmodes = 0

    def read(self, filename='restart'):
        """
        reads a SNF restart file.

        Parameters
        ----------
        filename : str
            filename (with path) of restart-file (SNF).
        """
        f = open(filename, 'r')

        f.readline()  # first line: some path

        format = f.readline()  # second line: format
        format = format.replace('d', 'D')
        self.n_per_line = int(format[1:format.find('D')])

        line = f.readline()  # third line
        # Convert third line in a python list
        line = [int(i) for i in line.split()]

        self.nvar = line[4]
        self.natoms = self.nvar // 3
        self.iaus = line[5]
        self.nmodes = line[6]
        if self.nmodes > 0:
            self.intonly = True

        # number of sets of calculations
        # normal run: one set are the 6 Cartesian displacements
        # intonly mode: one set are the two displacements per normal mode
        if self.intonly:
            self.ncalcsets = self.nmodes
        else:
            self.ncalcsets = self.natoms

        self.displ = float(f.readline().replace('D', 'E'))
        f.readline()  # fourth line: some other path
        f.readline()  # fifth line: and yet another path

        for i in range(self.ncalcsets):
            line = f.readline()
            line = [int(i) for i in line.split()]
            for i in line[1:]:
                if not (i == 1):
                    raise Exception('Not all points finished in SNF (step %i)'
                                    % line[0])
            f.readline()

        # FIXME: This only works with the ROA SNF version.
        #        It should be rewritten in a more general way that
        #        automatically detects which sections are present .

        self.dipole = self._read_section(f, 3)
        if not self.intonly:
            self.gradient = self._read_section(f, self.nvar)
        self.pollen = self._read_section(f, 8)
        self.polvel = self._read_section(f, 8)
        self.gtenlen = self._read_section(f, 9)
        self.gtenvel = self._read_section(f, 9)
        self.aten = self._read_section(f, 27)

        f.close()

    def _read_section(self, f, ncomp):
        """
        reads sections of restart-file.
        \n\n
        SNFRestartFile._read_section is
        a support/help-method for SNFRestartFile.read.

        Parameters
        ----------
        f : ?
           opened file from method SNFRestartFile.read.
        ncomp : int
           number of restart-file component.
        """

        if self.intonly:
            nread = ncomp * self.iaus
        else:
            nread = ncomp * self.iaus * 3

        result = numpy.zeros((self.ncalcsets, nread))

        nlines = nread // self.n_per_line
        if nlines*self.n_per_line < nread:
            nlines = nlines + 1

        for i in range(self.ncalcsets):
            f.readline()
            n = 0
            for j in range(nlines):
                line = f.readline().replace('D', 'E').split()
                for fl in line:
                    result[i, n] = float(fl)
                    n += 1

        result.shape = (self.ncalcsets, nread // ncomp, ncomp)
        return result

    def del_int_for_atoms(self, atoms):
        """
        deletes data entries for atoms.

        Parameters
        ----------
        atoms : integer or array
           Atom index indicating which atoms should be deleted.
           Example: \n
           atoms = 1 or atoms = [0,1,5,6]\n
           len(atoms) <= natoms
        """
        self.dipole[atoms, :, :] = 0.0
        self.pollen[atoms, :, :] = 0.0
        self.polvel[atoms, :, :] = 0.0
        self.gtenlen[atoms, :, :] = 0.0
        self.gtenvel[atoms, :, :] = 0.0
        self.aten[atoms, :, :] = 0.0


class SNFOutputFile(object):
    """
    reads the snf.out-files.

    Parameters
    ----------
    modes : class VibModes(3*natoms,mol)
       -> natoms: number of atoms of the molecule\n
       -> mol: class VibToolsMolecule (Modul: Molecule)\n
    lwl : float
       laser wave length
    filename : str or None
        filename (with path) of snf-output file.
    intonly : Bool
       If true then only intensities should be calculated.
       If False then intensities and fequencies should be calculated.
    """

    def __init__(self, filename='snf.out'):
        """
        SNFOutputFile constructor.
        For more details see the class description/docstring.
        """
        self.modes = None
        self.lwl = None
        self.filename = filename
        self.intonly = False

    def read(self, mol):
        """
        reads molecule data.

        Parameters
        ----------
        mol : VibTools.Molecule.
           VibTools class - Molecule.
        """
        f = open(self.filename, 'r', encoding='charmap')
        lines = f.readlines()  # lines = all line of snf.out
        f.close()

        # Check if intonly == True in lines
        for L in lines:  # line by line of lines
            if 'intensities-only-mode' in L:
                self.intonly = True

        first = 0
        last = 0
        # Preperation for further reading snf.out
        # assign start/last lines
        # i: current iteration, l: line by line of lines(snf.out)
        for i, l in enumerate(lines):
            if (first == 0) and ('root no.' in l):
                first = i
            if 'Generate fake outputs for' in l:
                last = i
                break
            if 'W A R N I N G' in l:
                last = i - 1
                break
            if 'Raman Optical Activity properties for freq.' in l:
                self.lwl = float(l[46:57])

        # read normal modes and corresponding frequencies from snf.out
        if not self.intonly:
            natoms = mol.natoms
            self.modes = VibModes(3*natoms-6, mol)

            lines = lines[first:last]
            lines = [L for L in lines if not len(L.strip()) == 0]
            lines = [L for L in lines if not L.startswith('1')]

            normalmodes = numpy.zeros((3*natoms-6, 3*natoms))
            freqs = numpy.zeros((3*natoms-6))

            ncol = len(lines[2][10:].split())

            icol = -ncol
            for i, l in enumerate(lines):
                irow = i % (3*natoms+2) - 2

                if irow == -2:
                    if not ('root' in l):
                        raise Exception('wrong number of lines \
                                        for normal modes')
                    icol += ncol
                else:
                    line = [float(fl) for fl in l[10:].split()]

                    if irow == -1:
                        freqs[icol:icol+ncol] = line
                    else:
                        normalmodes[icol:icol+ncol, irow] = line

            self.modes.set_modes_mw(normalmodes)
            self.modes.set_freqs(freqs)

    def read_mat(self, id_string):
        """Read in a matrix from the snf.out file, starting with the id_string.

        The matrix in the snf.out file should have the following form
        (only for the dipole gradients):

        ======== ======== ======== ======
        1        2        3
        ======== ======== ======== ======
        dmu_x/dx dmu_x/dy dmu_x/dz
        dmu_y/dx dmu_y/dy dmu_y/dz Atom 1
        dmu_z/dx dmu_z/dy dmu_z/dz
        ======== ======== ======== ======

        ======== ======== ======== ======
          4         5         6
        ======== ======== ======== ======
        dmu_x/dx dmu_x/dy dmu_x/dz
        dmu_y/dx dmu_y/dy dmu_y/dz Atom 2
        dmu_z/dx dmu_z/dy dmu_z/dz
        ======== ======== ======== ======


        etc.

        put one piece from the button to the right of
        the previous on and you get a 3 x 3*nr_atoms matrix
        but I want to work with a 3*nr_atoms x 3

        Parameters
        ----------
        Possible id_string strings : str
           "gradient of dipole moments",
           "gradient of polarizability tensor",
           "gradient of polarizability tensor (velocity representation)",
           "gradient of G/LAO tensor",
           "gradient of G/AO tensor",
           "gradient of A tensor".

        Returns
        -------
        matrix : numpy.ndarray
           matrix
        """

        f = open(self.filename, 'r', encoding='charmap')

        # Little hack to assure that you find the right polarizabilities
        # (length vs. velocity representation)
        vel = False
        if "velocity" in id_string:
            vel = True
        while (True):
            line = f.readline()
            if line == "":
                raise Exception("Could not find " + id_string + "!!!")
            if id_string in line:
                # Make sure that you don't get the velocity representation
                # instead of the length rep.
                if (not vel) and ("velocity" in line):
                    continue
                break

        # Get the header
        line = f.readline()
        columns = int(line.split()[5])
        num_rows = int(line.split()[9])
        i = 0
        matrix = numpy.zeros((num_rows, columns))

        # How is the matrix diplayed? Usually disp_col = 3
        line = f.readline()
        disp_col = len(line.split())

        row = 1
        while (i < num_rows):
            line = f.readline()
            beginning = str(row) + ":"
            if beginning in line:
                matrix[i: (i + disp_col),
                       row - 1] = [float(fl) for fl in line.split()[1:4]]
                if row == columns:
                    row = 1
                    i += disp_col
                else:
                    row += 1

        f.close()

        return matrix

    def read_derivatives(self, mol):
        """
        Based on VibTool's PySNF.SNFOutputFile.read_mat(id_strings).

        Parameters
        ----------
        mol : VibTools Molecule class
        """
        str_dip = "gradient of dipole moments"
        self.dipole = self.read_mat(str_dip).reshape(mol.natoms, 3, 3)
        str_pol = "gradient of polarizability tensor"
        self.pollen = self.read_mat(str_pol).reshape(mol.natoms, 3, 8)
        str_pol_vel1 = "gradient of polarizability tensor "
        str_pol_vel2 = "(velocity representation)"
        str_pol_vel = str_pol_vel1 + str_pol_vel2
        self.polvel = self.read_mat(str_pol_vel).reshape(mol.natoms, 3, 8)
        str_grad = "gradient of G/LAO tensor"
        self.gtenlen = self.read_mat(str_grad).reshape(mol.natoms, 3, 9)
        str_grad_GAO = "gradient of G/AO tensor"
        self.gtenvel = self.read_mat(str_grad_GAO).reshape(mol.natoms, 3, 9)
        str_grad_A = "gradient of A tensor"
        self.aten = self.read_mat(str_grad_A).reshape(mol.natoms, 3, 27)


class SNFControlFile:

    def __init__(self):
        """
        SNFControlFile constructor.
        For more details see the class description/docstring.
        """
        self.modes = None

    def read(self, mol):
        """"
        reads molecule data.

        Parameters
        ----------
        mol : VibTools.Molecule.
           VibTools class - Molecule.
        """
        f = open('snf_control', 'r', encoding='charmap')
        lines = f.readlines()
        f.close()

        for i, l in enumerate(lines):
            if l.startswith('$nummodes'):
                nmodes = int(l.split()[-1])
                start = i+1

        natoms = mol.natoms

        self.modes = VibModes(nmodes, mol)

        normalmodes = numpy.zeros((nmodes, 3*natoms))
        freqs = numpy.zeros((nmodes))

        for imode in range(nmodes):
            freqs[imode] = float(lines[start+imode*natoms].split()[0])
            for i in range(natoms):
                x, y, z = lines[start+imode*natoms+i].split()[-3:]
                normalmodes[imode, 3*i:3*i+3] = [x, y, z]

        self.modes.set_modes_c(normalmodes)
        self.modes.set_freqs(freqs)


class SNFResults(Results):
    """
    reads all relevant all SNF-data.

    (Attributes)
    Parameters
    ----------
    mol : class VibToolsMolecule
       VibTools.Molecule.
    snfoutput : class SNFOutputFile
       VibTools.PySNF.
    snfcontrol : class SNFControlFile
       VibTools.PySNF.
    restartfile :  SNFRestartFile
        VibTools.PySNF.

    Parameters
    ----------
    restartname : string
       filename for SNF restart file needed in class SNFRestartFile.
    coordfile : string
       filenname for coordfile.
    outname : string
       filename for SNF output file.
    intonly : Bool
       Default = False.
       If true then only intensities should be calculated.
       If False then intensities and fequencies should be calculated.
    """

    def __init__(self, outname='snf.out', restartname='restart',
                 coordfile='coord'):
        """
        SNFResults constructor.
        For more details see the class description/docstring.
        """
        self.mol = VibToolsMolecule()
        self.snfoutput = SNFOutputFile(filename=outname)
        self.snfcontrol = SNFControlFile()
        self.restartfile = SNFRestartFile()
        self.restartname = restartname
        self.coordfile = coordfile

        self.intonly = False

    def _get_modes(self):
        """internal SNF class get modes. """
        if self.snfoutput.intonly:
            return self.snfcontrol.modes
        else:
            return self.snfoutput.modes

    modes = property(_get_modes)
    lwl = property(lambda self: self.snfoutput.lwl)

    def read(self, deuterium=None):
        """
        reads the molecule, snfoutput, snfcontrol and restart files.
        """
        self.mol.read_from_coord(filename=self.coordfile, deuterium=deuterium)

        self.snfoutput.read(self.mol)
        self.intonly = self.snfoutput.intonly

        if self.snfoutput.intonly:
            self.snfcontrol.read(self.mol)

        if self.restartname:
            self.restartfile.read(filename=self.restartname)
        else:
            self.snfoutput.read_derivatives(self.mol)

            self.dipole_deriv_c = self.snfoutput.dipole
            self.pollen_deriv_c = (self.snfoutput.pollen[:, :, :6]
                                   / Constants.Bohr_in_Angstrom**2)
            self.polvel_deriv_c = (self.snfoutput.polvel[:, :, :6]
                                   / Constants.Bohr_in_Angstrom**2)
            self.gtenlen_deriv_c = self.snfoutput.gtenlen
            self.gtenvel_deriv_c = self.snfoutput.gtenvel
            self.aten_deriv_c = self.snfoutput.aten

    def get_tensor_mean(self, tens, ncomp=None):
        """
        gets tensor mean.

        Parameters
        ----------
        tens : numpy.ndarray
           tensor.
        ncomp : int
           index number for tensor.

        Returns
        -------
        mean : float
        tensor mean.
        """
        # ??? tens = restartfile.dipole_deriv_c ?????
        tensor = eval('self.restartfile' + '.' + tens)

        ncalcsets = tensor.shape[0]
        nval = tensor.shape[1] // 2
        if (ncomp is None):
            ncomp = tensor.shape[2]

        # displ = self.restartfile.displ

        mean = numpy.zeros((ncalcsets, nval, ncomp))

        for i in range(nval):
            mean[:, i, :] = (tensor[:, 2*i, :ncomp]
                             + tensor[:, 2*i+1, :ncomp]) / 2.0

        return mean

    def get_tensor_deriv_c(self, tens, ncomp=None):
        """
        gets tensor of detivative (cartesian).

        Parameters
        ----------
        tens : numpy.ndarray
           tensor.
        ncomp : int
           index number for tensor.

        Returns
        -------
        deriv : numpy.ndarray
        detivative.
        """
        if hasattr(self, tens+'_deriv_c'):
            return eval('self.'+tens+'_deriv_c')

        if not (self.restartfile.iaus == 2):
            raise Exception('Differentiation only implemented for iaus = 2')

        if self.intonly:
            raise Exception('Cartesian derivatives not available \
                            in intensities only mode')

        tensor = eval('self.restartfile' + '.' + tens)

        natoms = tensor.shape[0]
        if (ncomp is None):
            ncomp = tensor.shape[2]

        displ = self.restartfile.displ

        deriv = numpy.zeros((natoms, 3, ncomp))

        for i in range(3):
            deriv[:, i, :] = (tensor[:, 2*i, :ncomp]
                              - tensor[:, 2*i+1, :ncomp]) / (2.0 * displ)

        setattr(self, tens+'_deriv_c', deriv)
        return deriv

    def get_tensor_deriv_nm(self, tens, ncomp=None, modes=None):
        """
        gets normal modes derivative tensor.

        Parameters
        ----------
        tens : numpy.ndarray
           tensor.
        ncomp : int
           index number for tensor.
        modes : VibTools.Modes
           VibTools Modes class.

        Returns
        -------
        deriv : numpy.ndarray
        detivative nm.
        """
        if modes is None:
            if hasattr(self, tens+'_deriv_nm'):
                return eval('self.'+tens+'_deriv_nm')
        elif self.intonly:
            raise Exception('Argument modes of get_tensor_deriv_nm \
                            must be None in intensities-only mode')

        if self.intonly:
            tensor = eval('self.restartfile' + '.' + tens)

            nmodes = tensor.shape[0]
            if (ncomp is None):
                ncomp = tensor.shape[2]

            displ = self.restartfile.displ

            deriv_nm = numpy.zeros((nmodes, ncomp))
            deriv_nm[:, :] = (tensor[:, 0, :ncomp]
                              - tensor[:, 1, :ncomp]) / (2.0 * displ)

            modes = self.get_c_normalmodes()
            for i in range(nmodes):
                deriv_nm[i, :] = (deriv_nm[i, :]
                                  * math.sqrt(sum(modes[i, :]**2)))

        else:
            deriv_nm = Results.get_tensor_deriv_nm(self, tens, ncomp, modes)

        if modes is None:
            setattr(self, tens+'_deriv_nm', deriv_nm)
        return deriv_nm

    def check_consistency(self):
        """checks consistency of SNF restart file."""
        print()
        print("Checking consistency of SNF restart file ")
        print("  Large numbers mean that the change of the polarizabilities \
              deviates from linear behavior.")
        print()

        def check(ten):
            """internal check function for tensor ten (numpy.ndarray)."""
            if ten.startswith('pol'):
                mean_pol = self.get_tensor_mean(ten, 6)
            else:
                mean_pol = self.get_tensor_mean(ten)
            ncalcsets = mean_pol.shape[0]
            nval = mean_pol.shape[1]
            ncomp = mean_pol.shape[2]

            mean_pol = mean_pol.reshape((ncalcsets*nval, ncomp))
            for icomp in range(ncomp):
                print(" %14.8f  min: %3i  max:  %3i" %
                      (mean_pol[:, icomp].max()-mean_pol[:, icomp].min(),
                       mean_pol[:, icomp].argmin(),
                       mean_pol[:, icomp].argmax()))
            print()

            wrong = {}

            for i in range(ncalcsets):
                if (max(abs(mean_pol[3*i:3*i+3, 0] - mean_pol[0, 0])) > 0.001):
                    wrong[i+1] = [abs(mean_pol[3*i, 0]
                                  - mean_pol[0, 0]) > 0.001,
                                  abs(mean_pol[3*i+1, 0]
                                  - mean_pol[0, 0]) > 0.001,
                                  abs(mean_pol[3*i+2, 0]
                                  - mean_pol[0, 0]) > 0.001]
                    print(i+1, mean_pol[3*i:3*i+3, 0] - mean_pol[0, 0])

            return wrong

        print(" Dipole moment ")
        return check('dipole')

        return

        print(" Polarizability (length) ")
        check('pollen')

        print(" Polarizability (vel) ")
        check('polvel')

        print(" G-tensor (len) ")
        check('gtenlen')

        print(" G-tensor (vel) ")
        check('gtenvel')

        print(" A-tensor ")
        check('aten')
