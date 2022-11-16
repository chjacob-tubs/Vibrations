# This file is a part of 
# Vibrations - a Python Code for Anharmonic Theoretical Vibrational Spectroscopy
# Copyright (C) 2014-2022 by Pawel T. Panek, Adrian A. Hoeske, Julia Brüggemann,
# and Christoph R. Jacob.
#
#    Vibrations is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Vibrations is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Vibrations.  If not, see <http://www.gnu.org/licenses/>.
#
# In scientific publications using Vibrations please cite:
#   P. T. Panek, Ch. R. Jacob, ChemPhysChem 15 (2014) 3365.
#   P. T. Panek, Ch. R. Jacob, J. Chem. Phys. 144 (2016) 164111.
# 
# The most recent version of Vibrations is available at
#   http://www.christophjacob.eu/software

"""
Vibrations
==========

Vibrations is a Python code for vibrational calculations with
user-defined vibrational modes. For an overview of the capabilities
of the code, see the following articles U{ChemPhysChem 15 (2014) 3365 <http://dx.doi.org/10.1002/cphc.201402251>},
U{J. Chem. Phys. 144 (2016) 164111 <http://dx.doi.org/10.1063/1.4947213>}, and
U{J. Phys. Chem. Lett. 7 (2016) 3084 <http://dx.doi.org/10.1021/acs.jpclett.6b01451>}.

B{Usage}

See C{examples} directory for some examples of typical runs.

Vibrations can be run using Python's interpreter, or interactively with
some of the interactive Python consoles.

B{Requirements}

Vibrations is an independent code that for running needs only Python standard packages, extended with NumPy and SciPy.
However for typical usage, where normal modes from previous calculations are read in, the potential energy surfaces are
calculated with QM programs, another packages are needed
 - U{LocVib package<http://www.christophjacob.eu/>} for reading in normal modes, localizing them, etc.
 - U{PyADF suite<http://pyadf.org>} as an interface to QM codes, for calculating potential energy and property surfaces

Any suggestions and improvements are welcome.

@author: Pawel Panek
@organization: TU Braunschweig
@contact: c.jacob@tu-braunschweig.de
"""

from .VibrationalSCF import *
from .Grids import *
from .Surfaces import *
from .Wavefunctions import *
from .Misc import *
from .VibrationalCI import *

print()
print(' '+75*'*')
print(' *')
print(' *  Vibrations v0.9')
print(' *')                        
print(' *  Vibrations - a Python Code for Anharmonic Theoretical Vibrational Spectroscopy') 
print(' *  Copyright (C) 2014-2018 by Pawel T. Panek, and Christoph R. Jacob.')
print(' *')
print(' *     Vibrations is free software: you can redistribute it and/or modify')
print(' *     it under the terms of the GNU General Public License as published by')
print(' *     the Free Software Foundation, either version 3 of the License, or')
print(' *     (at your option) any later version.')
print(' *') 
print(' *     Vibrations is distributed in the hope that it will be useful,')
print(' *     but WITHOUT ANY WARRANTY; without even the implied warranty of')
print(' *     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the')  
print(' *     GNU General Public License for more details.')
print(' *')
print(' *     You should have received a copy of the GNU General Public License')
print(' *     along with Vibrations.  If not, see <http://www.gnu.org/licenses/>.') 
print(' *')
print(' *  In scientific publications using Vibrations please cite:')
print(' *    P. T. Panek, Ch. R. Jacob, ChemPhysChem 15 (2014) 3365.')
print(' *    P. T. Panek, Ch. R. Jacob, J. Chem. Phys. 144 (2016) 164111.')
print(' *')
print(' *  The most recent version of Vibrations is available at')
print(' *    http://www.christophjacob.eu/software')
print(' *')
print(' '+75*'*')
print(' *')
print(' *')
