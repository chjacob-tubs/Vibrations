# Vibrations

Vibrations is a Python code for vibrational calculations with
user-defined vibrational modes. For an overview of the capabilities
of the code, see the following articles

[ChemPhysChem 15 (2014) 3365](http://dx.doi.org/10.1002/cphc.201402251),

[J. Chem. Phys. 144 (2016) 164111](http://dx.doi.org/10.1063/1.4947213),

[J. Phys. Chem. Lett. 7 (2016) 3084](http://dx.doi.org/10.1021/acs.jpclett.6b01451).

You can find the documentation for Vibrations on the website:

[Vibrations Documentation](https://vibrations.readthedocs.io/en/latest/)

Download link:

[Vibrations on Github](https://github.com/chjacob-tubs/Vibrations)


## Requirements

Vibrations is an independent code that requires only Python standard packages, 
extended with NumPy, SciPy and Open Babel (for LocVib/PyADF). However, for typical usage, 
when normal modes from previous calculations are read in, and/or the potential energy surfaces 
are calculated with QM programs, additional packages are needed:

 * Already included as a git subtree in ``src/LocVib``: [LocVib package](https://github.com/chjacob-tubs/LocVib) for reading normal modes and localizing them.

 * To install manually: [PyADF suite](http://pyadf.org) as an interface to QM codes for calculating 
   potential energy and property surfaces.

## Installation

Just clone this repository and update the ``$PYTHONPATH`` environment variable accordingly.

Or install it using ``pip install`` in the folder containing the ``pyproject.toml`` file.

There are different predefined setup variants depending on your use case and prefered Open Babel package:

If you want to use the ``openbabel`` package, install with
```bash
pip install ".[openbabel]"
```

If you want to use the ``openbabel-wheel`` package, install with
```bash
pip install ".[openbabel-wheel]"
```

If you want to run the tests and/or build the documentation, it is recommended to do a full installation
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

For more details, see :ref:`Installation <Installation>`.

Verify the installation by running the tests:

```bash
    src/Vibrations/tests % python tests.py
    src/Vibrations/unittests % pytest -v .
```

Other installation options can be found in the further :ref:`documentation <Install in a Conda Environment (Recommended)>`.

## Building the Documentation

You can find the documentation, as mentioned above, on the homepage (https://vibrations.readthedocs.io/en/latest/), or generate the HTML documentation yourself with Sphinx.

Build documentation via Sphinx and extension packages (only works in installations that include documentation dependencies):

```bash
doc % sphinx-build . build
```

Open the ``index.html`` file to view the documentation homepage:

```bash
doc/build % open index.html
```

## Usage

See the `examples/` directory for some examples of typical runs.

Vibrations can be run using Python's interpreter, or interactively with
some of the interactive Python consoles.

### Any suggestions and improvements are welcome.
