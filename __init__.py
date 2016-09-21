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

"""
Vibrations
==========

Vibrations is Python code for vibrational calculations with
user-defined vibrational modes. For an overview of the capabilities
of the code, see the following articles U{ChemPhysChem 15 (2014) 3365 <http://dx.doi.org/10.1002/cphc.201402251>},
U{J. Chem. Phys. 144 (2016) 164111 <http://dx.doi.org/10.1063/1.4947213>}, and
U{J. Phys. Chem. Lett. 7 (2016) 3084 <http://dx.doi.org/10.1021/acs.jpclett.6b01451>}.

For an overview of possible usage see the C{examples} directory.

In general an input file, containing Python code, is needed,
and run with Python interpreter, e.g.,

C{python test.py}

B{Requirements}

Any suggestions and improvements are welcome.

@author: Pawel Panek
@organization: TU-Braunschweig
@contact: c.jacob@tu-bs.de
"""

from VibrationalSCF import *
from Grids import *
from Surfaces import *
from Wavefunctions import *
from Misc import *
from VibrationalCI import *

print
print ' '+75*'*'
print ' *'
print ' *  Vibrations v0.9'
print ' *'                        
print ' *  Vibrations - a Python Code for Anharmonic Vibrational Calculations' 
print ' *  wiht User-defined Vibrational Coordinates'
print ' *  Copyright (C) 2014-2016 by Pawel T. Panek, and Christoph R. Jacob.'
print ' *'
print ' *  Vibrations is free software: you can redistribute it and/or modify'
print ' *  it under the terms of the GNU General Public License as published by'
print ' *  the Free Software Foundation, either version 3 of the License, or'
print ' *  (at your option) any later version.'
print ' *' 
print ' *  Vibrations is distributed in the hope that it will be useful,'
print ' *  but WITHOUT ANY WARRANTY; without even the implied warranty of'
print ' *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the'  
print ' *  GNU General Public License for more details.'
print ' *'
print ' *  You should have received a copy of the GNU General Public License'
print ' *  along with Vibrations.  If not, see <http://www.gnu.org/licenses/>.' 
print ' *'
print ' '+75*'*'
print
print
