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
#
# import openbabel # works on Version 3.1.1.1.

from openbabel import openbabel
import numpy


class AbstractMolecule(object):
    """
    A class to represent an abstract Molecule:

    The module Molecule.py makes molecular structures
    (e.g. from Turbomole or PySNF) usable for LocVib.

    (Attributes := Parameters)

    Parameters
    ----------
    natoms : int or None
        total number of atoms in the molecule
    atmasses : array (1 x natoms) or None
        All atomic masses in the molecule
    atnums : array (1 x  natoms) or None
        All atomic numbers in the molecule
    coordinates : ndarray (natoms x (x y z)) or None
        coordinates of atoms in molecule natoms times (x y z)
    """

    def __init__(self):
        """
        Constructs all the necessary attributes
        for the abstract-Molecule object.
        """
        self.natoms = None
        self.atmasses = None
        self.atnums = None
        self.coordinates = None

    def get_fragment(self, atomlist):
        """
        It takes a fragment from the Molecule.

        Parameter
        ---------
        atomlist: list of integers
            Index list of current molecule. Example (H2O):
            atomlist = [0,1,2] -> Complete H2O-Molecule.
            atomlist = [1] -> single atom is left.

        Returns
        -------
        frag: VibToolsMolecule class
        Fragment from the Molecule.
        """
        frag = AbstractMolecule()

        frag.natoms = len(atomlist)
        frag.atmasses = self.atmasses[atomlist]
        frag.atnums = self.atnums[atomlist]
        frag.coordinates = self.coordinates[atomlist, :]

        return frag


class VibToolsMolecule(AbstractMolecule):
    """
    *VibToolisMolecule() uses the properies of the class AbstractMolecule() and
    makes molecular structures (e.g. from Turbomole or PySNF)
    usable for LocVib and relies on **Openbabel**.*

    *Initial:
    Creates the object: openbabel.OBmol()
    and resets the molecular-cache
    (atomic masses, atomic numbers and coordinates).*

    *The OBMol class is designed to store all
    the basic information associated with a molecule,
    to make manipulations on the connection table of
    a molecule facile (see Openbabel).*

    :Attributes: =Parameters (from AbstractMolecule)

    Parameters
    ----------
    natoms : int or None
        total number of atoms in the molecule
    atmasses : array (1 x natoms) or None
        All atomic masses in the molecule
    atnums : array (1 x  natoms) or None
        All atomic numbers in the molecule
    coordinates : ndarray (natoms x (x y z)) or None
        coordinates of atoms in molecule natoms times (x y z)
    """

    def __init__(self):
        """
        Constructs all the necessary attributes
        for the abstract-Molecule object (openbabel).

        Reset molcache (openbabel).
        """
        self.obmol = openbabel.OBMol()
        self.reset_molcache()

    def reset_molcache(self):
        """
        reset_molcache() resets the entries for
        atomic masses,
        atomic numbers and
        coordinates.
        See class AbstractMolecule (object).
        """
        self._atmasses = None
        self._atnums = None
        self._coords = None

    def read(self, filename, filetype="xyz", deuterium=None):
        """
        Reads the file into the openbabel OBMolecule object.

        Parameters
        ----------
        filename : str
            filename with path
        filetype : str
            coordinate format (Default: xyz)
            (further informations: see openbabel)
        """
        self.reset_molcache()

        obconv = openbabel.OBConversion()
        # sets format: it requires both arguments
        obconv.SetInAndOutFormats(filetype, filetype)
        # Reads the file into the OBMolecule object
        if not obconv.ReadFile(self.obmol, filename):
            raise Exception('Error reading file')

        if deuterium is not None:
            for at in deuterium:
                self.obmol.GetAtom(at).SetIsotope(2)
            print('Set atoms ', deuterium, 'to deuterium')
            print('Atomic masses: ')
            print(self.atmasses)

    def read_from_coord(self, filename='coord', deuterium=None):
        """
        reads a coord-file with 'tmol' format
        and delivers the corresponding AbstractMolcule attributes from file.

        Parameters
        ----------
        filename : str
            filename with path (tmol-file)
        """
        self.read(filename, "tmol", deuterium=deuterium)

    def write(self, filename, filetype="xyz"):
        """
        reads a file into the openbabel OBMolecule object.

        Parameters
        ----------
        filename : str
            filename with path
        filetype : str
            coordinate format (Default: xyz)
            (further informations: see openbabel)
        """
        obconv = openbabel.OBConversion()
        obconv.SetInAndOutFormats(filetype, filetype)

        if not obconv.WriteFile(self.obmol, filename):
            raise Exception('Error writing file')

    def get_natoms(self):
        """
        Returns
        -------
        natoms : int or None
        total number of atoms in the molecule.
        """
        return self.obmol.NumAtoms()

    natoms = property(get_natoms)

    def get_atmasses(self):
        """
        Returns
        -------
        atmasses : array (1 x natoms) or None
        All atomic masses in the molecule.
        """
        if self._atmasses is None:
            self._atmasses = numpy.zeros((self.natoms,))
            for at in openbabel.OBMolAtomIter(self.obmol):
                i = at.GetIdx()-1
                self._atmasses[i] = at.GetExactMass()
        return self._atmasses
    atmasses = property(get_atmasses)

    def get_atnums(self):
        """
        Returns
        -------
        atnums : array (1 x  natoms) or None
        All atomic numbers in the molecule.
        """
        if self._atnums is None:
            self._atnums = numpy.zeros((self.natoms,))
            for at in openbabel.OBMolAtomIter(self.obmol):
                i = at.GetIdx()-1
                self._atnums[i] = at.GetAtomicNum()
        return self._atnums
    atnums = property(get_atnums)

    def get_coordinates(self):
        """
        Returns
        -------
        coordinates : ndarray (natoms x (x y z)) or None
        coordinates of atoms in molecule natoms times (x y z).
        """
        if self._coords is None:
            self._coords = numpy.zeros((self.natoms, 3))
            for at in openbabel.OBMolAtomIter(self.obmol):
                i = at.GetIdx()-1
                self._coords[i, 0] = at.GetX()
                self._coords[i, 1] = at.GetY()
                self._coords[i, 2] = at.GetZ()
        return self._coords

    coordinates = property(get_coordinates)

    def add_atoms(self, atnums, coords):
        """
        Parameters
        ----------
        atnums : array (1 x  natoms) or None
           All atomic numbers in the molecule.
        coordinates : ndarray (natoms x (x y z)) or None
           coordinates of atoms in molecule natoms times (x y z).
        """
        self.obmol.BeginModify()

        for i in range(len(atnums)):
            a = self.obmol.NewAtom()
            a.SetAtomicNum(int(atnums[i]))
            a.SetVector(coords[i][0], coords[i][1], coords[i][2])
        self.obmol.EndModify()

        self.reset_molcache()

    ###########################################
    # the following methods rely on Openbabel #
    ###########################################

    def residue_groups(self):
        """
        Iterate over all atoms in an OBResidue.

        To facilitate iteration through all atoms in a residue,
        without resorting to atom indexes (which may change in the future)
        a variety of iterator classes and methods are provided.

        Returns
        -------
        groups : list(lists), ints
            Openbal residue indices.
        gorupnames : list of strings
            Openbabel residue indices.
        """
        groupnames = []
        groups = []

        # force residue perception
        self.obmol.GetAtom(1).GetResidue()

        for r in openbabel.OBResidueIter(self.obmol):
            resgroup = []
            for at in openbabel.OBResidueAtomIter(r):
                resgroup.append(at.GetIdx()-1)
            groupnames.append(str(r.GetNum()))
            groups.append(resgroup)

        z = list(zip(groupnames, groups))
        z.sort(key=(lambda x: int(x[0])))
        groupnames = [x[0] for x in z]
        groups = [x[1] for x in z]

        return groups, groupnames

    def attype_groups(self):
        """
        see residue_groups in respect to attype groups.
        """
        groupnames = ['N', 'H', 'C', 'O',  'CA', 'HA',
                      'CB', 'HB', 'OXT', 'HXT', 'H2O']
        groups = []

        print(groupnames)

        for k in groupnames:
            groups.append([])
        for at in openbabel.OBMolAtomIter(self.obmol):
            attype = at.GetResidue().GetAtomID(at)
            attype = attype.strip()

            if attype in ['1HB', '2HB', '3HB', 'HB1', 'HB2', 'HB3']:
                attype = 'HB'
            if attype in ['1H', '2H', 'H 1', 'H 2', 'H1', 'H2']:
                attype = 'HXT'

            if at.GetResidue().GetName() == 'HOH':
                attype = 'H2O'

            ind = groupnames.index(attype)
            groups[ind].append(at.GetIdx()-1)

        return groups, groupnames

    def attype_groups_2(self):
        """
        see residue_groups in respect to attype 2groups.
        """
        groupnames = ['NH', 'CO', 'CA', 'HA', 'CHB']
        groups = []

        for k in groupnames:
            groups.append([])
        for at in openbabel.OBMolAtomIter(self.obmol):
            attype = at.GetResidue().GetAtomID(at)
            attype = attype.strip()

            if attype in ['1HB', '2HB', '3HB', 'HB1', 'HB2', 'HB3']:
                attype = 'HB'
            if attype in ['1H', '2H', 'H 1', 'H 2', 'H1', 'H2']:
                attype = 'HXT'

            if attype in ['N', 'H', 'HXT']:
                attype = 'NH'
            elif attype in ['C', 'O', 'OXT']:
                attype = 'CO'
            elif attype in ['CB', 'HB']:
                attype = 'CHB'

            ind = groupnames.index(attype)
            groups[ind].append(at.GetIdx()-1)

        return groups, groupnames

    def attype_groups_3(self):
        """
        see residue_groups in respect to attype3 groups.
        """
        # groupnames = ['NH', 'CO', 'CA', 'CHB1', 'CHB2', 'CHG1', 'CHD1']
        # Attention: This list has been modified to make the test runnable
        # added Strings: "CB", "HB3", "HA"
        groupnames = ['NH', 'CO', 'CA', 'CB', 'HA', 'HB3',
                      'CHB1', 'CHB2', 'CHG1', 'CHD1']

        groups = []

        for k in groupnames:
            groups.append([])
        for at in openbabel.OBMolAtomIter(self.obmol):
            attype = at.GetResidue().GetAtomID(at)
            attype = attype.strip()

            if attype.startswith('HB1'):
                attype = 'HB1'
            elif attype.startswith('HB2'):
                attype = 'HB2'
            elif attype.startswith('HG1'):
                attype = 'HG1'
            elif attype.startswith('HD1'):
                attype = 'HD1'
            if attype in ['1H', '2H', 'H 1', 'H 2', 'H1', 'H2']:
                attype = 'HXT'

            if attype in ['N', 'H', 'HXT']:
                attype = 'NH'
            elif attype in ['C', 'O', 'OXT']:
                attype = 'CO'
            elif attype in ['CB1', 'HB1']:
                attype = 'CHB1'
            elif attype in ['CB2', 'HB2']:
                attype = 'CHB2'
            elif attype in ['CG1', 'HG1']:
                attype = 'CHG1'
            elif attype in ['CD1', 'HD1']:
                attype = 'CHD1'

            ind = groupnames.index(attype)
            groups[ind].append(at.GetIdx()-1)

        return groups, groupnames

    def attype_groups_7B(self):
        """
        see residue_groups in respect to attype 7B groups.
        """
        # Attention: This list has been modified to make the test runnable
        # added Strings: "CB", "HB3", "HA"
        groupnames = ['NH', 'CO', 'CHA', 'CHB1',
                      'CHB2', 'CHG1', 'CHD1', 'Ring', 'Term',
                      'HB3']
        groups = []

        for k in groupnames:
            groups.append([])
        for at in openbabel.OBMolAtomIter(self.obmol):
            attype = at.GetResidue().GetAtomID(at)
            attype = attype.strip()
            resname = at.GetResidue().GetName()

            if attype.startswith('HA'):
                attype = 'CHA'
            elif attype.startswith('HB1'):
                attype = 'HB1'
            elif attype.startswith('HB2'):
                attype = 'HB2'
            elif attype.startswith('HG1'):
                attype = 'HG1'
            elif attype.startswith('HD1'):
                attype = 'HD1'
            if attype in ['1H', '2H', 'H 1', 'H 2', 'H1', 'H2']:
                attype = 'HXT'

            if attype in ['N', 'H', 'HXT']:
                attype = 'NH'
            elif attype in ['C', 'O', 'OXT']:
                attype = 'CO'
            elif attype in ['CA']:
                attype = 'CHA'
            elif attype in ['CB', 'CB1', 'HB1']:
                attype = 'CHB1'
            elif attype in ['CB2', 'HB2']:
                attype = 'CHB2'
            elif attype in ['CG1', 'HG1']:
                attype = 'CHG1'
            elif attype in ['CD1', 'HD1']:
                attype = 'CHD1'

            if resname == 'BLA':
                # if not attype in ['CHA', 'CHB1', 'CHB2', 'NH', 'CO']:
                if attype not in ['CHA', 'CHB1', 'CHB2', 'NH', 'CO']:
                    attype = 'Ring'
            if resname == 'UNK':
                attype = 'Term'

            ind = groupnames.index(attype)
            groups[ind].append(at.GetIdx()-1)

        return groups, groupnames

    def attype_all_groups(self):
        """
        see residue_groups in respect to all attype groups.
        """
        # groupnames = ['N', 'H', 'C', 'O', 'CA', 'HA', 'CB',
        #               'HB', 'OXT', 'HXT', 'H2O', 'CH3', 'HH3']
        groupnames = []
        groups = []

        # print groupnames

        # for k in groupnames:
        #    groups.append([])
        for at in openbabel.OBMolAtomIter(self.obmol):
            attype = at.GetResidue().GetAtomID(at)

            attype = attype.strip()

            if attype in ['1HB', '2HB', '3HB', 'HB1', 'HB2', 'HB3']:
                attype = 'HB'
            if attype in ['1H', '2H', 'H 1', 'H 2', 'H1', 'H2']:
                attype = 'HXT'
            if attype in ['1HH3', '2HH3', '3HH3']:
                attype = 'HH3'

            if at.GetResidue().GetName() == 'HOH':
                attype = 'H2O'

            if attype not in groupnames:
                groupnames.append(attype)
                groups.append([])

            ind = groupnames.index(attype)
            groups[ind].append(at.GetIdx() - 1)

        return groups, groupnames

    def atom_groups(self):
        """
        see residue_groups in respect to atom groups.
        """
        group_dict = {}

        # Openbabel 2.x Version
        #        pse = openbabel.OBElementTable()
        #
        #        for at in openbabel.OBMolAtomIter(self.obmol):
        #            atname = pse.GetSymbol(at.GetAtomicNum())

        # Openbabel 3.x Version (Start)
        for at in openbabel.OBMolAtomIter(self.obmol):
            atname = openbabel.GetSymbol(at.GetAtomicNum())
        # Openbabel 3.x Version (End)

            if atname in group_dict:
                group_dict[atname].append(at.GetIdx()-1)
            else:
                group_dict[atname] = [at.GetIdx()-1]

        groupnames = []
        groups = []

        for g in group_dict:
            groupnames.append(g)
            groups.append(group_dict[g])

        return groups, groupnames

    def amide1_groups(self, res):
        # Attention: The term "ref" is also used differently in VibTools:
        # see results/residue etc.
        """
        see residue_groups in respect to index of groups.

        Parameters
        ----------
        res : int
           index
        """
        groupnames = ['CO', 'NH', 'NH2', 'NH3', 'CAH',
                      'CA0', 'CA2', 'COo', 'CBa', 'R']
        groups = []

        for k in groupnames:
            groups.append([])
        for at in openbabel.OBMolAtomIter(self.obmol):
            attype = at.GetResidue().GetAtomID(at)
            attype = attype.strip()

            rnum = at.GetResidue().GetNum()

            if (rnum == res) and (attype in ['C', 'O']):
                typ = 'CO'
            elif (rnum == res + 1) and (attype in ['N', 'H']):
                typ = 'NH'
            elif (rnum == res + 1) and (attype in ['CA', 'HA']):
                typ = 'CAH'
            elif (rnum == res + 2) and attype in ['N', 'H']:
                typ = 'NH2'
            elif (rnum == res + 3) and attype in ['N', 'H']:
                typ = 'NH3'
            elif (rnum == res) and attype in ['CA', 'HA']:
                typ = 'CA0'
            elif (rnum == res + 2) and attype in ['CA', 'HA']:
                typ = 'CA2'
            elif attype in ['C', 'O']:
                typ = 'COo'
            elif attype in ['CB', '1HB', '2HB', '3HB']:
                typ = 'CBa'
            elif attype in ['CB', 'HB']:
                typ = 'CBo'
            else:
                typ = 'R'

            ind = groupnames.index(typ)
            groups[ind].append(at.GetIdx()-1)

        return groups, groupnames

    def amide2_groups(self, res):
        """
        see residue_groups in respect to amide2 (index) groups.

        Parameters
        ----------
        res : int
           index
        """
        groupnames = ['CO-1', 'NH', 'CO', 'NH+1', 'CO+1', 'NH+2', 'CH01', 'R']
        groups = []

        for k in groupnames:
            groups.append([])
        for at in openbabel.OBMolAtomIter(self.obmol):
            attype = at.GetResidue().GetAtomID(at)
            attype = attype.strip()

            rnum = at.GetResidue().GetNum()

            if attype in ['1HB', '2HB', '3HB', 'HB1', 'HB2', 'HB3']:
                attype = 'HB'

            if rnum == res:
                if attype in ['N', 'H']:
                    attype = 'NH'
                elif attype in ['C', 'O']:
                    attype = 'CO'
                elif attype in ['CA', 'HA', 'CB', 'HB']:
                    attype = 'CH01'
            elif rnum == res + 1:
                if attype in ['N', 'H']:
                    attype = 'NH+1'
                elif attype in ['C', 'O']:
                    attype = 'CO+1'
                elif attype in ['CA', 'HA', 'CB', 'HB']:
                    attype = 'CH01'
                else:
                    attype = 'R'
            elif rnum == res + 2:
                if attype in ['N', 'H']:
                    attype = 'NH+2'
                else:
                    attype = 'R'
            elif rnum == res - 1:
                if attype in ['C', 'O']:
                    attype = 'CO-1'
                else:
                    attype = 'R'
            else:
                attype = 'R'

            ind = groupnames.index(attype)
            groups[ind].append(at.GetIdx()-1)

        return groups, groupnames

    def amide3_groups(self, res):
        """
        see residue_groups in respect to amide3 (index) groups.

        Parameters
        ----------
        res : int
           index
        """
        groupnames = ['Am-1/0', 'CA0', 'Am0/1', 'CA+1',
                      'Am1/2', 'CA+2', 'CB', 'R']
        groups = []

        for k in groupnames:
            groups.append([])
        for at in openbabel.OBMolAtomIter(self.obmol):
            attype = at.GetResidue().GetAtomID(at)
            attype = attype.strip()

            rnum = at.GetResidue().GetNum()

            if attype in ['1HB', '2HB', '3HB', 'HB1', 'HB2', 'HB3']:
                attype = 'HB'

            if rnum == res:
                if attype in ['N', 'H']:
                    attype = 'Am-1/0'
                elif attype in ['C', 'O']:
                    attype = 'Am0/1'
                elif attype in ['CA', 'HA']:
                    attype = 'CA0'
                elif attype in ['CB', 'HB']:
                    attype = 'CB'
            elif rnum == res + 1:
                if attype in ['N', 'H']:
                    attype = 'Am0/1'
                elif attype in ['C', 'O']:
                    attype = 'Am1/2'
                elif attype in ['CA', 'HA']:
                    attype = 'CA+1'
                elif attype in ['CB', 'HB']:
                    attype = 'CB'
            elif rnum == res + 2:
                if attype in ['N', 'H']:
                    attype = 'Am1/2'
                elif attype in ['CA', 'HA']:
                    attype = 'CA+2'
                elif attype in ['CB', 'HB']:
                    attype = 'CB'
                else:
                    attype = 'R'
            elif rnum == res - 1:
                if attype in ['C', 'O']:
                    attype = 'Am-1/0'
                else:
                    attype = 'R'
            else:
                attype = 'R'

            try:
                ind = groupnames.index(attype)
            except ValueError:
                print(attype)
                raise
            groups[ind].append(at.GetIdx()-1)

        return groups, groupnames

    def skelCN_groups(self, res):
        """
        see residue_groups in respect to skelCN (index) groups.

        Parameters
        ----------
        res : int
           index
        """
        groupnames = ['NCA0', 'HA0', 'CO0',
                      'H+1', 'NCA+1', 'HA+1', 'CHB+1', 'CO+1',
                      'H+2', 'NCA+2',
                      'R']
        groups = []

        for k in groupnames:
            groups.append([])
        for at in openbabel.OBMolAtomIter(self.obmol):
            attype = at.GetResidue().GetAtomID(at)
            attype = attype.strip()

            rnum = at.GetResidue().GetNum()

            if attype in ['1HB', '2HB', '3HB', 'HB1', 'HB2', 'HB3']:
                attype = 'HB'

            if rnum == res:
                if attype in ['C', 'O']:
                    attype = 'CO0'
                elif attype in ['N', 'CA']:
                    attype = 'NCA0'
                elif attype in ['HA']:
                    attype = 'HA0'
                else:
                    attype = 'R'
            elif rnum == res + 1:
                if attype in ['H']:
                    attype = 'H+1'
                elif attype in ['C', 'O']:
                    attype = 'CO+1'
                elif attype in ['N', 'CA']:
                    attype = 'NCA+1'
                elif attype in ['HA']:
                    attype = 'HA+1'
                elif attype in ['CB', 'HB']:
                    attype = 'CHB+1'
            elif rnum == res + 2:
                if attype in ['H']:
                    attype = 'H+2'
                elif attype in ['N', 'CA']:
                    attype = 'NCA+2'
                else:
                    attype = 'R'
            else:
                attype = 'R'

            try:
                ind = groupnames.index(attype)
            except ValueError:
                print(attype)
                raise
            groups[ind].append(at.GetIdx()-1)

        return groups, groupnames

    def ch3_groups(self, res):
        """
        see residue_groups in respect to ch3 (index) groups.

        Parameters
        ----------
        res : int
           index
        """
        groupnames = ['CO0', 'NH+1', 'CA+1', 'CB+1', 'CO+1', 'NH+2', 'R']
        groups = []

        for k in groupnames:
            groups.append([])
        for at in openbabel.OBMolAtomIter(self.obmol):
            attype = at.GetResidue().GetAtomID(at)
            attype = attype.strip()

            rnum = at.GetResidue().GetNum()

            if attype in ['1HB', '2HB', '3HB', 'HB1', 'HB2', 'HB3']:
                attype = 'HB'

            if rnum == res:
                if attype in ['C', 'O']:
                    attype = 'CO0'
                else:
                    attype = 'R'
            elif rnum == res + 1:
                if attype in ['N', 'H']:
                    attype = 'NH+1'
                elif attype in ['C', 'O']:
                    attype = 'CO+1'
                elif attype in ['CA', 'HA']:
                    attype = 'CA+1'
                elif attype in ['CB', 'HB']:
                    attype = 'CB+1'
            elif rnum == res + 2:
                if attype in ['N', 'H']:
                    attype = 'NH+2'
                else:
                    attype = 'R'
            else:
                attype = 'R'

            try:
                ind = groupnames.index(attype)
            except ValueError:
                print(attype)
                raise
            groups[ind].append(at.GetIdx()-1)

        return groups, groupnames
