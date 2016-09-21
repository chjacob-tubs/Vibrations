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

import vibrations as vib
import VibTools
import numpy as np
import time
import sys
import os

class InputData:

    def __init__(self, dirname):

        f = open(dirname+'/grid_params')
        lines = f.readlines()
        f.close()

        whichmodes = None
        localized = None

        if os.path.isfile(dirname+'/whichmodes'):
            whichmodes = []
            f = open(dirname+'/whichmodes')
            lines2 = f.readlines()
            f.close
            for l in lines2:
                for el in l.split():
                    whichmodes.append(int(el))

        

        grid_params = lines[0].split()

        self.ngrid = int(grid_params[0])
        self.amp = int(grid_params[1])


        res = VibTools.SNFResults(outname=dirname+'/snf.out',restartname=dirname+'/restart',
                                  coordfile=dirname+'/coord')
        res.read()


        self.freqs = res.modes.freqs

        self.ints = res.get_ir_intensity()
        modes = res.modes

        if whichmodes is not None:
            self.freqs = res.modes.freqs[whichmodes]
            self.ints = res.get_ir_intensity(modes=res.modes.get_subset(whichmodes))
            modes = res.modes.get_subset(whichmodes)
        

        self.grid = vib.Grid(res.mol,modes)
        self.grid.generate_grids(self.ngrid,self.amp)

        self.v1 = vib.Potential(self.grid,1)
        self.v1.read_np(dirname+'/v1.npy')

        self.v1_harm = vib.Potential(self.grid,1)
        self.v1_harm.generate_harmonic()

        self.dm1_harm = vib.Dipole(self.grid,1)
        self.dm1_harm.generate_harmonic(res)

        self.v2 = vib.Potential(self.grid,2)
        self.v2.read_np(dirname+'/v2.npy')

        self.dm1 = vib.Dipole(self.grid,1)
        self.dm1.read_np(dirname+'/dm1.npy')

        self.dm2 = vib.Dipole(self.grid,2)
        self.dm2.read_np(dirname+'/dm2.npy')
        
        self.diagonal_eigv = np.load(dirname+'/diagonal_eigv.npy')
        self.vscf_singles_energies = np.load(dirname+'/vscf_singles.npy')
        
        self.vci1 = np.load(dirname+'/vci1.npy')
        self.vci2 = np.load(dirname+'/vci2.npy')
        self.vci3 = np.load(dirname+'/vci3.npy')
        self.vci4 = np.load(dirname+'/vci4.npy')

        self.vci1_int = np.load(dirname+'/vci1_int.npy')
        #self.vci2_int = np.load(dirname+'/vci2_int.npy')
        #self.vci3_int = np.load(dirname+'/vci3_int.npy')
        #self.vci4_int = np.load(dirname+'/vci4_int.npy')


class LogFile(object):
    
    """
    Manages a log file. 
    
    This class is an utilized  class written by Moritz Klammer
    for pyadf/Turbomole interface testing purposes
    """
    
    
    def __init__(self, filename):
        """Initializes `filename' as new logfile."""
        self.logfilename = filename + '_summary' + os.extsep + 'log'
        self.linemarkup = []
        open(self.logfilename, 'w').close()
        self.write_sep()
        self.write_line(test='TEST', setname='TESTSET', seconds='TIME', outcome='OUTCOME', markup='strong')
        self.write_sep()

    def write_sep(self, markup=None):
        self.linemarkup.append(markup)
        with open(self.logfilename, 'a') as logfile:
            logfile.write('-'*100 + '\n')
    
    def finish(self):
        self.write_sep()
    
    def write_line(self, test=None, setname=None, seconds=None, outcome=None, markup=None):
        """Adds another line (for another test) to the log file."""
        self.linemarkup.append(markup)
        _time = None
        try:
            seconds = int(seconds)
            _time = '{m:02}:{s:02}'.format(m=seconds/60, s=seconds%60)
        except Exception:
            _time = seconds
        
        with open(self.logfilename, 'a') as logfile:
            logfile.write('{0:50}'.format(test))
            logfile.write('{0:30}'.format(setname))
            logfile.write('{0:10}'.format(_time))
            logfile.write('{0:10}'.format(outcome))
            logfile.write('\n')
        
        
    def present(self):
        with open(self.logfilename, 'r') as logfile:
            i = 0
            for line in logfile:
                markup=self.linemarkup[i]
                if markup is None:
                    words = line.split()
                    if words:
                        if words[-1] == 'passed':
                            markup = 'fine'
                        elif words[-1] == 'failed':
                            markup = 'alert'
                        elif words[-1] == 'skipped':
                            markup = 'weak'
                sys.stdout.write(mark_up(line, markup))
                i += 1

def mark_up(text, markup):
    if markup == 'strong':
        return '\033[1m' + text + '\033[m'
    elif markup == 'alert':
        return '\033[1;31m' + text + '\033[m'
    elif markup == 'fine':
        return '\033[32m' + text + '\033[m'
    elif markup == 'weak':
        return '\033[33m' + text + '\033[m'
    else:
        return text




def test_diagonal_VSCF(inpdata):

    if inpdata is None:
        return 'Diagonal VSCF'

    dVSCF = vib.VSCFDiag(inpdata.v1)
    dVSCF.solve()
    dVSCF.print_results()
    
    # difference between the reference and current eigenvalues

    d = (np.abs(dVSCF.eigv - inpdata.diagonal_eigv)).mean()
    
    # test passed if the mean absolute error is below 1e-6 a.u.

    return d < 1e-6

def test_diagonal_VSCF_harmonic(inpdata):

    if inpdata is None:
        return 'Diagonal VSCF with harmonic potential'

    dVSCF = vib.VSCFDiag(inpdata.v1_harm)
    dVSCF.solve()
    dVSCF.print_results()

    results = np.zeros(dVSCF.nmodes)

    for i in range(dVSCF.nmodes):
        results[i] = (dVSCF.eigv[i][1] - dVSCF.eigv[i][0]) / vib.Misc.cm_in_au

    d = np.abs(results - inpdata.freqs).mean()

    return d < 1 

def test_VSCF_singles(inpdata):

    if inpdata is None:
        return 'VSCF gs + singles'

    VSCF = vib.VSCF2D(inpdata.v1,inpdata.v2)
    VSCF.solve_singles()

    # diffrerence between the reference VSCF energies (ground state and singles)
    # and currect energies

    d = np.abs(np.array(VSCF.energies) - inpdata.vscf_singles_energies).mean()
    
    # test passed if the mean absolute error is below 1cm-1

    return d < 1



def test_VCI1(inpdata):

    if inpdata is None:
        return 'VCI singles'

    VSCF = vib.VSCF2D(inpdata.v1,inpdata.v2)
    VSCF.solve_singles()

    VCI = vib.VCI(VSCF.get_groundstate_wfn(), inpdata.v1, inpdata.v2)
    VCI.generate_states(1)
    VCI.solve()

    d = np.abs(VCI.energies - inpdata.vci1).mean()

    return d < 1

def test_VCI2(inpdata):

    if inpdata is None:
        return 'VCI doubles'

    VSCF = vib.VSCF2D(inpdata.v1,inpdata.v2)
    VSCF.solve_singles()

    VCI = vib.VCI(VSCF.get_groundstate_wfn(), inpdata.v1, inpdata.v2)
    VCI.generate_states(2)
    VCI.solve()

    d = np.abs(VCI.energies - inpdata.vci2).mean()

    return d < 1

def test_VCI3(inpdata):

    if inpdata is None:
        return 'VCI triples'

    VSCF = vib.VSCF2D(inpdata.v1,inpdata.v2)
    VSCF.solve_singles()

    VCI = vib.VCI(VSCF.get_groundstate_wfn(), inpdata.v1, inpdata.v2)
    VCI.generate_states(3)
    VCI.solve()

    d = np.abs(VCI.energies - inpdata.vci3).mean()

    return d < 1

def test_VCI4(inpdata):

    if inpdata is None:
        return 'VCI quadruples'

    VSCF = vib.VSCF2D(inpdata.v1,inpdata.v2)
    VSCF.solve_singles()

    VCI = vib.VCI(VSCF.get_groundstate_wfn(), inpdata.v1, inpdata.v2)
    VCI.generate_states(4)
    VCI.solve()

    d = np.abs(VCI.energies - inpdata.vci4).mean()

    return d < 1

def test_VCI1_int(inpdata):

    if inpdata is None:
        return 'VCI singles intensities'

    VSCF = vib.VSCF2D(inpdata.v1,inpdata.v2)
    VSCF.solve_singles()

    VCI = vib.VCI(VSCF.get_groundstate_wfn(), inpdata.v1, inpdata.v2)
    VCI.generate_states(1)
    VCI.solve()
    VCI.calculate_intensities(inpdata.dm1, inpdata.dm2)

    d = np.abs(VCI.intensities - inpdata.vci1_int).mean()

    return d < 1

def test_VCI2_int(inpdata):

    if inpdata is None:
        return 'VCI doubles intensities'

    VSCF = vib.VSCF2D(inpdata.v1,inpdata.v2)
    VSCF.solve_singles()

    VCI = vib.VCI(VSCF.get_groundstate_wfn(), inpdata.v1, inpdata.v2)
    VCI.generate_states(2)
    VCI.solve()
    VCI.calculate_intensities(inpdata.dm1, inpdata.dm2)

    d = np.abs(VCI.intensities - inpdata.vci2_int).mean()

    return d < 1

def test_VCI3_int(inpdata):

    if inpdata is None:
        return 'VCI triples intensities'

    VSCF = vib.VSCF2D(inpdata.v1,inpdata.v2)
    VSCF.solve_singles()

    VCI = vib.VCI(VSCF.get_groundstate_wfn(), inpdata.v1, inpdata.v2)
    VCI.generate_states(3)
    VCI.solve()
    VCI.calculate_intensities(inpdata.dm1, inpdata.dm2)

    d = np.abs(VCI.intensities - inpdata.vci3_int).mean()
    return d < 1

def test_VCI4_int(inpdata):

    if inpdata is None:
        return 'VCI quadruples intensities'

    VSCF = vib.VSCF2D(inpdata.v1,inpdata.v2)
    VSCF.solve_singles()

    VCI = vib.VCI(VSCF.get_groundstate_wfn(), inpdata.v1, inpdata.v2)
    VCI.generate_states(4)
    VCI.solve()
    VCI.calculate_intensities(inpdata.dm1, inpdata.dm2)

    d = np.abs(VCI.intensities - inpdata.vci4_int).mean()
    return d < 1

tests = [
         test_diagonal_VSCF, 
         test_diagonal_VSCF_harmonic,
         test_VSCF_singles, 
         test_VCI1, 
         test_VCI2,
         #test_VCI3, 
         #test_VCI4,
         test_VCI1_int
         #test_VCI2_int
         #test_VCI3_int,
         #test_VCI4_int
         ]

testdata = ['data','ala6_amideI']

logfile = LogFile('tests_of_tests')
total_tests = 0
failed_tests = 0
skipped_tests = 0
passed_tests = 0


for test in tests:
    for testset in testdata:

        total_tests += 1
        _test = test(None)
        _setname = testset
        _outcome = None
        _seconds = None

        try:
            inpdata = InputData(testset)

        except Exception as e:
            sys.stderr.write(str(e) + '\n')
            skipped_tests += 1
            inpdata = None
            _setname += ' (not found)'
            _outcome = 'skipped'

        if inpdata is not None:

            starttime = time.time()
            testres = test(inpdata)
            endtime = time.time()
            _seconds = endtime - starttime
            
            if testres:
                _outcome = 'passed'
                passed_tests += 1
            else:
                _outcome = 'failed'
                failed_tests += 1

        logfile.write_line(test=_test, setname=_setname, seconds=_seconds, outcome=_outcome)

logfile.finish()

print
print '-'*100
print
print "TESTSUITE PROCESSED"
print
print "Total tests:   {0:4}".format(total_tests)
print "Passed tests:  {0:4} ({1:7.2%})".format(passed_tests,  float(passed_tests)  / float(total_tests))
print "Failed tests:  {0:4} ({1:7.2%})".format(failed_tests,  float(failed_tests)  / float(total_tests))
print "Skipped tests: {0:4} ({1:7.2%})".format(skipped_tests, float(skipped_tests) / float(total_tests))
print
logfile.present()
