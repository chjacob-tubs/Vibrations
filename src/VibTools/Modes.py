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

import math
import numpy
import pylab

from . import Constants


class VibModes(object):
    """
    Vibrational modes class

    :Attributes=Parameters

    Parameters
    ----------
    nmodes : int
        number of modes.
    mol : class
        -> VibToolsMolecule class.
    natoms: int or None
        total number of atoms in the molecule.
    freqs : array (nmodes)
        vibrational frequencies.
    modes_mw :  ndarray (nmodes x 3*natoms)
        mass-weighted normal modes.
    modes_c : ndarray(nmodes x 3*natoms)
        cartesian normal modes.

    Example
    -------
    *Modes(c/mw) format:* (3*natoms-6) times 3*natoms
        Atom 1: [[X Y Z  X Y Z ...  X Y Z]

        Atom 2: [X Y Z  X Y Z  ... X Y Z]

        ...

        Atom n: [X Y Z  X Y Z ...  X Y Z]]
    """

    def __init__(self, nmodes, mol):
        """
        VibModes constructor.
        For more details see the class description/docstring
        """
        self.nmodes = nmodes

        self.mol = mol
        self.natoms = mol.natoms

        self.freqs = numpy.arange(1.0, nmodes+0.1, 1.0)

        self.modes_mw = numpy.zeros((nmodes, 3*self.natoms))
        self.modes_c = numpy.zeros((nmodes, 3*self.natoms))

    def set_modes_mw(self, modes_mw):
        """set mass-weighted modes."""
        self.modes_mw[:, :] = modes_mw
        # Warum kein modes_mw_update wie bei modes_c????
        self.normalize_modes()

    def set_modes_c(self, modes_c):
        """set cartesian modes."""
        self.modes_c[:, :] = modes_c
        self.update_modes_mw()
        self.normalize_modes()

    def set_mode_mw(self, i, mode, freq=None):
        """set single mode in mass-weighted modes."""
        self.modes_mw[i, :] = mode
        if freq:
            self.freqs[i] = freq
        self.normalize_modes()

    def set_mode_c(self, i, mode, freq=None):
        """set single mode in cartesian modes (mass-weighted normalized)."""
        self.modes_c[i, :] = mode
        if freq:
            self.freqs[i] = freq
        self.update_modes_mw()
        self.normalize_modes()

    def set_freqs(self, freqs):
        """set frequencies."""
        self.freqs = freqs

    def get_modes_c_norm(self):
        """get cartesian normalized cartesian modes."""
        norm_modes = self.modes_c.copy()
        for imode in range(self.nmodes):
            nc = math.sqrt((self.modes_c[imode, :]**2).sum())
            norm_modes[imode, :] = (1.0/nc) * norm_modes[imode, :]
        return norm_modes

    def normalize_modes(self):
        """normalize cartesian and mass-weighted modes."""
        for imode in range(self.nmodes):
            nc = math.sqrt((self.modes_mw[imode, :]**2).sum())
            if abs(nc) > 0.0:
                self.modes_mw[imode, :] = (1.0/nc) * self.modes_mw[imode, :]
        self.update_modes_c()

    modes_c_norm = property(get_modes_c_norm)

    def update_modes_c(self):
        """update cartesian modes in respect to norm."""
        for idir in range(3):
            self.modes_c[:, idir::3] = (self.modes_mw[:, idir::3]
                                        / numpy.sqrt(self.mol.atmasses))

    def update_modes_mw(self):
        """update the mass-weighted modes in respect to the norm."""
        for idir in range(3):
            self.modes_mw[:, idir::3] = (self.modes_c[:, idir::3]
                                         * numpy.sqrt(self.mol.atmasses))

    def get_subset(self, modelist):
        """get subset of modes and frequencies.

        Parameters
        ----------
        modeslist : 1D-array
            indexes of internal atomic assignments
            (see example in detailed documentation).

        Returns
        -------
        subset : VibTools.Modes class
        returns the subset of the corresponding Modes class.
        """
        subset = VibModes(len(modelist), self.mol)
        subset.set_modes_mw(self.modes_mw[modelist].copy())
        subset.set_freqs(self.freqs[modelist].copy())
        return subset

    def get_range(self, minfreq, maxfreq):
        """get a certain freqency range(minimal frequency, maximal frequency).

        Parameters
        ----------
        minfreq : float
            lower bound for freqency range.
        maxfreq : float
            upper bound for freqency range.

        Returns
        -------
        subset : VibTools.Modes class
        returns the range(subset) of the corresponding Modes class.
        """
        modelist = (numpy.where((self.freqs > minfreq)
                    & (self.freqs < maxfreq))[0])
        return self.get_subset(modelist)

    def write_g98out(self, massweighted=False, normalize=True,
                     filename='g98.out'):
        """
        Creates Gaussian system outfile g98.out.

        Parameters
        ----------
        massweighted : Boolean
             massweighting on/off
        normalize : Boolean
             normalizing on/off
        filename : str
             filename with path
        """
        if massweighted:
            modes = self.modes_mw
        else:
            if normalize:
                modes = self.modes_c_norm
            else:
                modes = self.modes_c

        if self.nmodes % 3 == 0:
            temp_modes = modes
        else:
            temp_modes = numpy.zeros((3*(self.nmodes//3)+3, 3*self.natoms))
            temp_modes[:self.nmodes, :] = modes[:, :]

        temp_freqs = numpy.zeros((temp_modes.shape[0],))
        temp_freqs[:self.nmodes] = self.freqs

        f = open(filename, 'w')
        f.write(' Entering Gaussian System \n')
        f.write(' *********************************************\n')
        f.write(' Gaussian 98:\n')
        f.write(' frequency fake output\n')
        f.write(' *********************************************\n')
        f.write('                         Standard orientation:\n')
        f.write(' --------------------------------------------------------------------\n')
        f.write('  Center     Atomic     Atomic              Coordinates (Angstroms)  \n')
        f.write('  Number     Number      Type              X           Y           Z \n')
        f.write(' --------------------------------------------------------------------\n')

        atnums = self.mol.atnums
        coords = self.mol.coordinates

        for iatom in range(self.natoms):
            f.write(' %4i       %4i             0     %11.6f %11.6f %11.6f \n' %
                    (iatom+1, atnums[iatom], coords[iatom, 0], coords[iatom, 1], coords[iatom, 2]))

        f.write(' --------------------------------------------------------------------\n')
        f.write('     1 basis functions        1 primitive gaussians \n')
        f.write('     1 alpha electrons        1 beta electrons\n')
        f.write('\n')
        f.write(' Harmonic frequencies (cm**-1), IR intensities (KM/Mole), \n')
        f.write(' Raman scattering activities (A**4/amu), Raman depolarization ratios, \n')
        f.write(' reduced masses (AMU), force constants (mDyne/A) and normal coordinates: \n')

        for i in range(0, temp_modes.shape[0], 3):
            f.write('                   %4i                   %4i                  %4i \n' %
                    (i+1, i+2, i+3))
            f.write('                     a                      a                      a  \n')
            f.write(' Frequencies -- %10.4f             %10.4f             %10.4f \n' %
                    (temp_freqs[i], temp_freqs[i+1], temp_freqs[i+2]))
            f.write(' Red. masses -- %10.4f             %10.4f             %10.4f \n' % (0.0, 0.0, 0.0))
            f.write(' Frc consts  -- %10.4f             %10.4f             %10.4f \n' % (0.0, 0.0, 0.0))
            f.write(' IR Inten    -- %10.4f             %10.4f             %10.4f \n' % (0.0, 0.0, 0.0))
            f.write(' Raman Activ --     0.0000                 0.0000                 0.0000 \n')
            f.write(' Depolar     --     0.0000                 0.0000                 0.0000 \n')
            f.write(' Atom AN      X      Y      Z        X      Y      Z        X      Y      Z \n')

            for iatom in range(self.natoms):
                atnum = atnums[iatom]
                f.write('%4i %3i   %6.2f %6.2f %6.2f   %6.2f %6.2f %6.2f   %6.2f %6.2f %6.2f \n' %
                        (iatom+1, atnum,
                         temp_modes[i, 3*iatom],
                         temp_modes[i, 3*iatom+1],
                         temp_modes[i, 3*iatom+2],
                         temp_modes[i+1, 3*iatom],
                         temp_modes[i+1, 3*iatom+1],
                         temp_modes[i+1, 3*iatom+2],
                         temp_modes[i+2, 3*iatom],
                         temp_modes[i+2, 3*iatom+1],
                         temp_modes[i+2, 3*iatom+2],))

        f.write('\n')
        f.write('--------\n')
        f.close()

    def get_composition(self, groups):
        """
        gets composition

        Parameters
        ----------
        groups : list of lists
            You get the corresponding data from
            the molecule module (mol.attype_groups())

        Returns
        -------
        types : numpy.ndarray
            types array(:,3)
        """
        types = numpy.zeros((len(groups), self.nmodes))
        squared_modes = self.modes_mw**2
        for itype, attype in enumerate(groups):
            for at in attype:
                types[itype, :] += squared_modes[:, 3*at]
                types[itype, :] += squared_modes[:, 3*at+1]
                types[itype, :] += squared_modes[:, 3*at+2]
        return types

    def print_composition(self, groupnames, comp, labels=None):
        """
        prints composition.

        Parameters
        ----------
        groupnames : list of strings
             Molecule.residue_groues relies on Openbabel residue indices.
        comp : numpy.ndarray
           Returns of types = Modes.get_composition(groups).
        """
        if labels is None:
            lab = ["%4i %8.1f   " % (i, self.freqs[i])
                   for i in range(comp.shape[1])]
        else:
            lab = labels

        print((" "*len(lab[0])), end=' ')
        for t in groupnames:
            print("%5s  " % t, end=' ')
        print()

        for i in range(comp.shape[1]):
            print(lab[i], end=' ')
            for t in range(comp.shape[0]):
                print("%5.1f  " % (comp[t, i]*100.0), end=' ')
            print()

        print()
        print("min: " + " "*(len(lab[0])-5), end=' ')
        for t in range(comp.shape[0]):
            print("%5.1f  " % (comp[t, :].min()*100.0), end=' ')
        print()

        print("max: " + " "*(len(lab[0])-5), end=' ')
        for t in range(comp.shape[0]):
            print("%5.1f  " % (comp[t, :].max()*100.0), end=' ')
        print()
        print()

    def print_residue_composition(self, labels=None):
        """prints residue composition based on VibTools.Molecule"""
        groups, groupnames = self.mol.residue_groups()
        comp = self.get_composition(groups)
        self.print_composition(groupnames, comp, labels)

    def print_attype_composition(self, labels=None):
        """prints attype composition based on VibTools.Molecule"""
        groups, groupnames = self.mol.attype_groups()
        comp = self.get_composition(groups)
        self.print_composition(groupnames, comp, labels)

    def print_attype2_composition(self, labels=None):
        """prints attype2 composition based on VibTools.Molecule"""
        groups, groupnames = self.mol.attype_groups_2()
        comp = self.get_composition(groups)
        self.print_composition(groupnames, comp, labels)

    def print_attype3_composition(self, labels=None):
        """prints attype3 composition based on VibTools.Molecule"""
        groups, groupnames = self.mol.attype_groups_3()
        comp = self.get_composition(groups)
        self.print_composition(groupnames, comp, labels)

    def print_attype7B_composition(self, labels=None):
        """prints attype7B composition based on VibTools.Molecule"""
        groups, groupnames = self.mol.attype_groups_7B()
        comp = self.get_composition(groups)
        self.print_composition(groupnames, comp, labels)

    def print_attype_all_composition(self, labels=None):
        """prints all composition based on VibTools.Molecule"""
        groups, groupnames = self.mol.attype_all_groups()
        comp = self.get_composition(groups)
        self.print_composition(groupnames, comp, labels)

    def print_atom_composition(self, labels=None):
        """prints atom composition based on VibTools.Molecule"""
        groups, groupnames = self.mol.atom_groups()
        comp = self.get_composition(groups)
        self.print_composition(groupnames, comp, labels)

    def transform(self, tmat):
        """
        Unitary transformation function

        `H(q) = Q^t H(m) Q`

        `Q` := Tranformation matrix (`Q^t` = transposed)

        `H` := Hessian matrix;
        `m` = mass-weighted, `q` = diagonalized Hessian Matrix.

        Parameters
        ----------
        tmat : ndarray(3 x 3, dtype=float)
            Unitary transformation matrix.

        Returns
        -------
        tmodes : VibTools.Modes class
        transformed modes.
        """
        tmodes = VibModes(self.nmodes, self.mol)
        tmodes.set_modes_mw(numpy.dot(tmat, self.modes_mw))

        cmat = numpy.dot(numpy.dot(tmat, numpy.diag(self.freqs)),
                         tmat.transpose())

        tmodes.set_freqs(cmat.diagonal())

        return tmodes

    def sortmat_by_groups(self, groups):
        """
        sorts modes based on composition.

        Parameters
        ----------
        groups : list of lists
           Example H2O: [[], [], [], [], [], [], [], [], [], [], [0, 1, 2]].
           See VibTools.Molecule.attype_groups()

        Returns
        -------
        sortmat : ndarray(nmodes x nmodes)
        sorting matrix by groups
        """
        types = self.get_composition(groups)

        max_res = []
        for i in range(self.nmodes):
            maxind = -1
            for t in range(len(types)):
                if (types[t][i] > 0.4):
                    maxind = t
            max_res.append((i, maxind))

        max_res.sort(key=(lambda x: self.freqs[x[0]]), reverse=True)
        max_res.sort(key=(lambda x: x[1]))

        sortmat = numpy.zeros((self.nmodes, self.nmodes))
        for i in range(self.nmodes):
            sortmat[i, max_res[i][0]] = 1.0

        return sortmat

    def sortmat_by_residue(self, external_molecule=None):
        """sorts modes based on sortmat_by_groups (composition).

        Parameters
        ----------
        external_molecule : None
           None

        Returns
        -------
        sortmat_by_groups : ndarray(nmodes x nmodes)
        sorting matrix by groups.
        """
        if external_molecule:
            return (self.sortmat_by_groups(
                    external_molecule.residue_groups()[0]))
        else:
            return self.sortmat_by_groups(self.mol.residue_groups()[0])

    def sortmat_by_freqs(self):
        """*sorts modes based on freqs.*

        Returns
        -------
        sortmat : ndarray(nmodes x nmodes)
        sorting matrix by frequencies
        """
        inds = numpy.argsort(self.freqs)
        sortmat = numpy.zeros((self.nmodes, self.nmodes))
        for i in range(self.nmodes):
            sortmat[i, inds[i]] = 1.0

        return sortmat

    def get_fragment_modes(self, atomlist):
        """get a certain fragment of modes
        (based on VibTools.Molecule.get_fragment).

        Parameters
        ----------
        atomlist : list(dtype=int)
            List consisting of atom indices which should be
            present in the new mode class.
            Example: atomlist = [3,5,7]

        Returns
        -------
        subset : VibTools.Modes class
        returns the fragment of the corresponding Modes class
        """
        frag = self.mol.get_fragment(atomlist)

        indices = [0]*3*len(atomlist)
        for i, a in enumerate(atomlist):
            indices[3*i] = 3*a
            indices[3*i+1] = 3*a+1
            indices[3*i+2] = 3*a+2

        modes = VibModes(self.nmodes, frag)
        modes.set_freqs(self.freqs)

        modes.set_modes_mw(self.modes_mw[:, indices])

        return modes

    def overlap(self, other):
        """
        calculates the overlap between two mode matrices.

        Parameters
        ----------
        other : -> Format is equal to the attribute modes_mw
            A set of modes for which the overlap to the modes
            of the existing class is to be calculated.

        Returns
        -------
        ov : list(float)
        results of overlap calculation.
        """
        ov = numpy.zeros((self.nmodes,))

        for i in range(self.nmodes):
            ov[i] = numpy.dot(self.modes_mw[i], other.modes_mw[i])

        ov = ov**2

        return ov

    def center(self, imode):
        """
        Calculates the center of mass of a mode.

        Parameters
        ----------
        imode : int
            mode index in modes_mw or modes_c

        Returns
        -------
        cen : list(float)
        center of mass of the corresponding mode.
        """

        sq_mode = self.modes_mw[imode, :]**2

        c = numpy.zeros((self.natoms,))
        for iatom in range(self.natoms):
            c[iatom] = (sq_mode[3*iatom]
                        + sq_mode[3*iatom+1]
                        + sq_mode[3*iatom+2])

        cen = numpy.zeros((3,))
        coords = self.mol.coordinates

        for iatom in range(self.natoms):
            cen[0] += coords[iatom, 0] * c[iatom]
            cen[1] += coords[iatom, 1] * c[iatom]
            cen[2] += coords[iatom, 2] * c[iatom]

        return cen

    def distance(self, imode, jmode):
        """
        calculates distance between two centers of mass.

        Parameters
        ----------
        imode : int
            mode index in modes_mw (or modes_c?)
        jmode : int
            mode index in modes_mw (or modes_c?)


        Returns
        -------
        return : float
        Distance between two centers of mass.
        """
        dist = self.center(imode) - self.center(jmode)
        return math.sqrt(numpy.dot(dist, dist))

    def get_tdc_couplingmat(self, dipole):
        """
        getes tdc couplingmat.

        Parameters
        ----------
        dipole : numpy.ndarray
           dipolemoments
           - SNFResults
           - read()
           - modes.get_subset ... dipole=modes = res.modes.get_subset)a)
           dipole = res.get_tensor_deriv_nm('dipole',modes=modes).

        Returns
        -------
        tdcmat : numpy.ndarray
        tdc coupling matrix
        """
        tdcmat = numpy.diag(self.freqs)

        for i in range(self.nmodes):
            dipi_mag = math.sqrt(numpy.dot(dipole[i], dipole[i]))
            dipi_vec = dipole[i] / dipi_mag

            # dipole is in  a.u./(amu^-0.5 * A)
            print(i, dipi_mag * (Constants.au_in_Debye /
                  Constants.Bohr_in_Angstrom))

            for j in range(i+1, self.nmodes):
                dipj_mag = math.sqrt(numpy.dot(dipole[j], dipole[j]))
                dipj_vec = dipole[j] / dipj_mag

                dist_vec = self.center(i) - self.center(j)
                dist_mag = math.sqrt(numpy.dot(dist_vec, dist_vec))
                dist_vec = dist_vec / dist_mag
                dist_mag = dist_mag / Constants.Bohr_in_Angstrom
                # dist_mag is now in bohr

                x = (numpy.dot(dipi_vec, dipj_vec)
                     - 3.0*numpy.dot(dipi_vec, dist_vec)
                     * numpy.dot(dipj_vec, dist_vec))

                tdcmat[i, j] = (dipi_mag*dipj_mag / dist_mag**3) * x

        # convert to Omega matrix in cm-1
        fact = (Constants.Hartree_in_Joule /
                (Constants.amu_in_kg*Constants.Bohr_in_Meter**2))
        fact = fact / (Constants.cvel_ms*1e2)**2
        for i in range(self.nmodes):
            for j in range(i+1, self.nmodes):
                tdcmat[i, j] = tdcmat[i, j] * fact
                tdcmat[i, j] = tdcmat[i, j] / (4*math.pi**2 *
                                               (tdcmat[i, i]+tdcmat[j, j]))
                tdcmat[j, i] = tdcmat[i, j]
        return tdcmat

    def get_modes_asgn(self, groups, thresh=0.70, moddiff=10.0,
                       freqthresh=1200.0, graph=False):
        """
        Automatic assignment of normal modes to groups basing on cotributions
        of chosen groups/types of atoms to the normal modes.

        Parameters
        ----------
        groups : list
           assignment of atoms to the groups,
           see VibToolsMolecule.attype_groups() in Molecule module
        thresh : float
           threshold for modes similarity (0.0,1.0)
        modiff : float
           maximal distance between two neighboring
           normal modes in the same group
        freqthresh : float
           lower frequency limit of considered normal modes
        graph : bool
           plotting modes similarity matrix

        Returns
        -------
        groupmodes : list
        normal modes assigned to groups containing the most similar modes
        """

        nmodes = self.nmodes
        freqs = self.freqs

        M = numpy.zeros((nmodes, nmodes))
        tmp = []
        groupmodes = []

        compos = self.get_composition(groups)
        compos = compos.transpose()

        for i in range(nmodes):
            compos[i, :] /= math.sqrt(numpy.dot(compos[i, :], compos[i, :]))

        for i in range(nmodes):
            for j in range(nmodes):
                M[i, j] = numpy.dot(compos[i, :], compos[j, :])
                diff = abs(freqs[i]-freqs[j])
                if diff > moddiff:
                    M[i, j] /= diff / moddiff

        i = numpy.flatnonzero(freqs > freqthresh)[0]
        i = numpy.flatnonzero(M[i, :] > thresh)[0]
        tmp.append(i)
        while i < nmodes - 1:
            if M[i, i+1] < thresh:
                groupmodes.append(tmp)
                tmp = []
            i += 1
            tmp.append(i)
        if tmp != []:
            groupmodes.append(tmp)
        tmp = []

        for g in groupmodes:
            if len(g) > 1:
                tmp.append(g)
        groupmodes = tmp

        if graph:  # == True:
            pylab.matshow(M)
            pylab.colorbar()
            pylab.grid()
            for i in groupmodes:
                p = pylab.Rectangle((i[0]-0.5, i[0]-0.5),
                                    i[-1]-i[0]+1,
                                    i[-1]-i[0]+1,
                                    facecolor='none',
                                    edgecolor='white', lw='2.0')
                pylab.gca()
                pylab.gca().add_patch(p)
            pylab.show()

        return groupmodes
