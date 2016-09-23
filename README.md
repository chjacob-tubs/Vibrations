# Vibrations

Vibrations is a Python code for vibrational calculations with
user-defined vibrational modes. For an overview of the capabilities
of the code, see the following articles [ChemPhysChem 15 (2014) 3365](http://dx.doi.org/10.1002/cphc.201402251),
[J. Chem. Phys. 144 (2016) 164111](http://dx.doi.org/10.1063/1.4947213), and
[J. Phys. Chem. Lett. 7 (2016) 3084](http://dx.doi.org/10.1021/acs.jpclett.6b01451).

## Requirements

Vibrations is an independent code that for running needs only Python standard packages, extended with NumPy and SciPy.
However, for typical usage, when normal modes from previous calculations are read in, and/or the potential energy surfaces are
calculated with QM programs, additional packages are needed
 * [LocVib package](http://www.christophjacob.eu/) for reading in normal modes, localizing them, etc.
 * [PyADF suite](http://pyadf.org) as an interface to QM codes, for calculating potential energy and property surfaces

## Installation

Just clone this repository and run the tests in `tests` directory.

    cd tests
    pyton tests.py

## Documentation

In order to compile the documentation [epydoc](http://epydoc.sourceforge.net) is needed.
In the main directory run:

    epydoc *.py

This generates the documentation in HTML format. For other formats see epydoc's
documentation.

## Usage

See `examples` directory for some examples of typical runs.

Vibrations can be run using Python's interpreter, or interactively with
some of the interactive Python consoles.

### Any suggestions and improvements are welcome.
