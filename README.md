# Vibrations

Vibrations is a Python code for vibrational calculations with
user-defined vibrational modes. For an overview of the capabilities
of the code, see the following articles

[ChemPhysChem 15 (2014) 3365] (http://dx.doi.org/10.1002/cphc.201402251),

[J. Chem. Phys. 144 (2016) 164111] (http://dx.doi.org/10.1063/1.4947213),

[J. Phys. Chem. Lett. 7 (2016) 3084] (http://dx.doi.org/10.1021/acs.jpclett.6b01451).

We can find the documentation for LocVib/VibTools on the website

[Vibrations Documentation](https://vibrations.readthedocs.io/en/latest/)

Download-Link:

[Vibrations Github](https://github.com/chjacob-tubs/Vibrations)


## Requirements

Vibrations is an independent code that for running needs only Python standard 
packages, extended with NumPy, SciPy and Openbabel(for LocVib/PyADF). However, for typical usage, when normal 
modes from previous calculations are read in, and/or the potential energy surfaces 
are calculated with QM programs, additional packages are needed

 * Already included as git subtree in /src/LocVib: [LocVib package](https://github.com/chjacob-tubs/LocVib) for reading normal modes and localizing them.

 * To install manually: [PyADF suite](http://pyadf.org) as an interface to QM codes, for calculating 
   potential energy and property surfaces.

## Installation

Just clone this repository, update $PYTHONPATH environment variable accordingly.

Or install it using pip install in the folder where pyproject.toml is located.

There are different predefined setup variants depending on your use case and prefered openbabel package:

If you want to use openbabel package install with
```bash
pip install ".[openbabel]"
```

If you want to use openbabel-wheel package install with
```bash
pip install ".[openbabel-wheel]"
```

If you want to run the tests and/or build the documentation it is recommended to do a full installation
```bash
pip install ".[full-openbabel]"
```
or
```bash
pip install ".[full-openbabel-wheel]"
```

You can build the Fortran libraries automatically with

```bash
pip install ".[fints]"
```

More details are in the documentation.

Verify the installation with running the tests:

```bash
    src/Vibrations/tests/% python tests.py
    src/Vibrations/unittests/% pytest -v .
```

Other installation options can be found in the further documentation.

## Documentation

We can find the documentation, as mentioned above, on the Homepage (https://vibrations.readthedocs.io/en/latest/) or we can generate the HTML documentation ourselves with Sphinx.

Build documentation via Sphinx and extension packages (works in docs installations only):

```bash
doc/ % sphinx-build . build
```

Opening the index.html file takes you to the home directory of the code documentation:

```bash
doc/build/ % open index.html
```

## Usage

See `examples` directory for some examples of typical runs.

Vibrations can be run using Python's interpreter, or interactively with
some of the interactive Python consoles.

### Any suggestions and improvements are welcome.
