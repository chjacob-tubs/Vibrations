# LocVib/VibTools

Python tools for localizing normal modes.

Copyright (C) 2009-2026 by Christoph R. Jacob and others.

In scientific publications using the LocVib tools, please cite:
  Ch. R. Jacob, J. Chem. Phys 130 (2009), 084106 (https://doi.org/10.1063/1.3077690).

We can find the documentation for LocVib/VibTools on the website

https://vib.gitlab-pages.pyadf.org/LocVib/ 

Download-Link:

https://github.com/chjacob-tubs/LocVib

## Requirements

LocVib is an independent code that for running needs only Python standard
packages, extended with NumPy, Matplotlib and Openbabel.
More details are in the documentation.

## Installation

Just clone this repository and update `$PYTHONPATH` environment variable accordingly.
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

More details are in the documentation.

Verify the installation with running pytest (works in testing and full installations only):

```bash
src/VibTools/tests % pytest - v
```

Other installation options can be found in the further documentation.

## Documentation

You can find the documentation, as mentioned above,
on the Homepage (https://vib.gitlab-pages.pyadf.org/LocVib/) 
or you can generate the HTML documentation yourself with Sphinx.

Build documentation via Sphinx and extension packages (works in docs installations only):

```bash
doc/ % sphinx-build . build
```

Opening the index.html file takes you to the home directory of the code documentation:

```bash
doc/build/ % open index.html
```

## Usage

See `src/VibTools/example/` directory for some examples of typical runs.

### Any suggestions and improvements are welcome.
