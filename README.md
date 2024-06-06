[![pipeline status](https://gitlab.pyadf.org/vib/vibrations/badges/update_doc/pipeline.svg)](https://gitlab.pyadf.org/vib/vibrations/-/commits/update_doc)
[![Custom Badge](https://gitlab.pyadf.org/vib/vibrations/badges/update_doc/coverage.svg?min_medium=30&min_acceptable=50&min_good=75&key_text=Unittest+Coverage&key_width=130)](https://gitlab.pyadf.org/vib/vibrations/-/commits/update_doc)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![Numpy 1.23.4](https://img.shields.io/badge/numpy-1.23.4-blue.svg)](https://numpy.org/devdocs/release/1.23.4-notes.html)
[![Matplotlib 3.6.1](https://img.shields.io/badge/matplotlib-3.6.1-blue.svg)](https://matplotlib.org/stable/users/installing/index.html)
[![Scipy 1.9.3](https://img.shields.io/badge/scipy-1.9.3-blue.svg)](https://scipy.org/stable/users/installing/index.html)
[![Openbabel 3.1.1](https://img.shields.io/badge/openbabel-3.1.1-cornflowerblue.svg)](https://bioweb.pasteur.fr/packages/pack@openbabel@3.1.1)
[![Openbabel 3.1.1.16](https://img.shields.io/badge/openbabel--wheel-3.1.1.16-blueviolet.svg)](https://pypi.org/project/openbabel-wheel/3.1.1.16/)

# Vibrations

Vibrations is a Python code for vibrational calculations with
user-defined vibrational modes. For an overview of the capabilities
of the code, see the following articles

[ChemPhysChem 15 (2014) 3365] (http://dx.doi.org/10.1002/cphc.201402251),

[J. Chem. Phys. 144 (2016) 164111] (http://dx.doi.org/10.1063/1.4947213),

[J. Phys. Chem. Lett. 7 (2016) 3084] (http://dx.doi.org/10.1021/acs.jpclett.6b01451).

We can find the documentation for LocVib/VibTools on the website

[Vibrations Documentation](https://vib.gitlab-pages.pyadf.org/vibrations)

Download-Link:

[Vibrations Github](https://github.com/chjacob-tubs/Vibrations)


## Requirements

Vibrations is an independent code that for running needs only Python standard 
packages, extended with NumPy, SciPy and Openbabel(for LocVib/PyADF). However, for typical usage, when normal 
modes from previous calculations are read in, and/or the potential energy surfaces 
are calculated with QM programs, additional packages are needed

 * [LocVib package](https://github.com/chjacob-tubs/LocVib) for reading normal modes, 
   localizing them,

 * [PyADF suite](http://pyadf.org) as an interface to QM codes, for calculating 
   potential energy and property surfaces.

## Installation

Just clone this repository, update $PYTHONPATH environment variable accordinly.
Or install it using pip by using

    vibrations/% pip install .

in the folder where `setup.py` is located.

Verify the installation with running the tests in the `tests/` and `unittests/` directory.

    vibrations/tests/% pyton tests.py
    vibrations/unittests/% pytest -v .

Other installation options can be found in the further documentation.

## Documentation

We can find the documentation, as mentioned above, on the Homepage (https://vib.gitlab-pages.pyadf.org/vibrations) or we can generate the HTML documentation ourselves with Sphinx.

Build documentation via Sphinx and extension packages (must be installed before).

    vibrations/doc/ % sphinx-build . Build

Opening the index.html file takes you to the home directory of the code documentation:

    vibrations/doc/Build/ % open index.html

## Usage

See `examples` directory for some examples of typical runs.

Vibrations can be run using Python's interpreter, or interactively with
some of the interactive Python consoles.

### Any suggestions and improvements are welcome.
