************
Installation
************

The *Vibrations* *Python* package releases are available as a Github repository
https://github.com/chjacob-tubs/Vibrations.

Vibrations relies on additional *Python* packages, that have to be installed on our system:

* **Python 3** (https://www.python.org/).

* **Numpy** (https://numpy.org/).

* **Matplotlib** (https://matplotlib.org/).

* **Openbabel 3** (https://open-babel.readthedocs.io/en/latest/UseTheLibrary/Python.html).

Non-standard packages:

* **LocVib** (https://github.com/chjacob-tubs/LocVib)

* **PyADF** (https://github.com/chjacob-tubs/pyadf-releases)

Additional (optinal) extensions:

* **Pytest** (https://pytest.org) (recommended).

* **Sphinx** (https://www.sphinx-doc.org)

For the installation we just need to download the file and follow the instructions below.

.. note::
   We highly recommend using a **Conda environment**.

Download the Code
=================

Download the Github repository as a zip file from the link below:

https://github.com/chjacob-tubs/Vibrations

Unzip the Vibrations-Package-Zip-File:

   >>> unzip file.zip

We should now find the following folder structure:

.. code-block:: bash

   vibrations/
   ├── doc/
   ├── Dockerfile
   ├── example/
   ├── LICENCE
   ├── README.md
   ├── requirements/
   ├── setup.py
   ├── src/
   ├── tests/
   └── unittests/

Install with pip (easiest method)
=================================

Execute the following command in the main code directory
(Python-3 and a current pip version must already be installed):


    >>> vibrations/ % pip install .

.. _automatic pip installation:

Install with Conda
==================

We highly recommend using the *Conda* package manager(https://conda.io/) for the environment
and the use of *Anaconda* (https://www.anaconda.com/) for using *Python*.
The above *Python* dependencies (*Numpy*, *Matplotlib*, *Openbabel*, *scipy*,; *LocVib*, *PyADF*) must be installed for *Vibrations* to work.

With *Pytest* we can determine the correct executability of the program.

If we are interested in the further development of *Vibrations* ourselfs,
it makes sense to install *Sphinx* for the documentation.

Installation Conda Environment
------------------------------

First we install *Conda* on our system.
The best way to do this is to follow the instructions
on the *Conda* homepage (https://conda.io/).

Here we show in short form which *Conda* commands are necessary
to install the necessary *Python* packages,
provided that our conda installation worked.

**Conda initialization:**

   >>> % conda init zsh

Or usage without initialization only with `source activate`:

   >>> % source activate
   >>> (base)%

Pip Installation in Conda Environment (recommended)
---------------------------------------------------

Create and activate the Vibrations Conda environment:

   >>> (base)% conda create --name VibENV python=3.11.4
   >>> (base)% conda activate VibENV
   >>> (VibENV)%

As a prerequisite we still need the pip package:

   >>> (VibENV)% conda install pip

Select the *vibrations* folder and run the *pip* installation:

   >>> (VibENV)% cd vibrations
   >>> (VibENV)/vibrations% pip install .

.. code-block:: console

   Processing /home/user/vibrations
     Preparing metadata (setup.py) ... done
   Requirement already satisfied: numpy~=1.23.4 in /home/name/.conda/envs/vibrations/lib/python3.11/site-packages (from Vibrations==1.0) (1.23.5)
   Requirement already satisfied: matplotlib~=3.6.1 in /home/name/.conda/envs/vibrations/lib/python3.11/site-packages (from Vibrations==1.0) (3.6.3)
   Requirement already satisfied: scipy~=1.9.3 in /home/name/.conda/envs/vibrations/lib/python3.11/site-packages (from Vibrations==1.0) (1.9.3)
   Requirement already satisfied: openbabel-wheel~=3.1.1.16 in /home/name/.conda/envs/vibrations/lib/python3.11/site-packages (from Vibrations==1.0) (3.1.1.19)
   .
   .
   .
   Requirement already satisfied: six>=1.5 in /home/name/.conda/envs/vibrations/lib/python3.11/site-packages (from python-dateutil>=2.7->matplotlib~=3.6.1->Vibrations==1.0) (1.16.0)
   Building wheels for collected packages: Vibrations
     Building wheel for Vibrations (setup.py) ... done
     Created wheel for Vibrations: filename=Vibrations-1.0-py3-none-any.whl size=156222 sha256=e837d20377535e76c88f27cc93659196139c795dc14aca78cbbd899d758ea6f3
     Stored in directory: /tmp/pip-ephem-wheel-cache-yl1ollq0/wheels/26/17/f4/afdbd4d31e095c018331cc1889f0c67cd402100bc8edf359e5
   Successfully built Vibrations
   Installing collected packages: Vibrations
     Attempting uninstall: Vibrations
       Found existing installation: Vibrations 1.0
       Uninstalling Vibrations-1.0:
         Successfully uninstalled Vibrations-1.0
   Successfully installed Vibrations-1.0


.. note:: 
   We can also install with pip in developer mode (editable). 
      >>> (VibENV)/vibrations% pip install -e .

Here everything is done regarding the installation.

**Optional:** For verification (see :doc:`Verify Installation with Pytest <verify_installation>`) of successful installation install **pytest** additionally:

   >>> (VibENV)/vibrations% conda install -c conda-forge pytest=7.2.0

.. note::
   The automatic Pip installation also works without Conda environment but Pip must be installed anyway.

Semi-Automatic Installation with Conda
--------------------------------------

Go to the Vibrations folder and perform the creation of the appropriate environment:

   >>> (base)% cd vibrations/
   >>> (base)vibrations/ % conda env create -f requirements/environment.yml
   Collecting package metadata (repodata.json): done
   Downloading and Extracting Packages
   kiwisolver-1.4.4     | 70 KB     | ################################################## | 100%
   libllvm14-14.0.6     | 33.4 MB   | ################################################## | 100%
   openssl-1.1.1v       | 3.7 MB    | ################################################## | 100%
   .
   .
   .
   certifi-2023.7.22    | 154 KB    | ################################################## | 100%
   Solving environment: done
   Preparing transaction: done
   Verifying transaction: done
   Executing transaction: done
   #
   # To activate this environment, use
   #
   #     $ conda activate VibENV
   #
   # To deactivate an active environment, use
   #
   #     $ conda deactivate

Now you only need to activate the environment and add Vibrations:

   >>> (base)vibrations/ % conda activate VibENV
   >>> (VibENV) vibrations/ % conda develop src/
   added /home/User/vibrations/src
   completed operation for: /home/User/vibrations/src

Manual Installation
-------------------

**Creating suitable Vibrations environment:**

   >>> conda create -n VibENV

   >>> conda activate VibENV

**Installation of the necessary packages:**

   >>> (VibENV)% conda install anaconda
   >>> (VibENV)% conda install -c conda-forge python~=3.11.4
   >>> (VibENV)% conda install -c conda-forge numpy~=1.23.4
   >>> (VibENV)% conda install -c conda-forge matplotlib`=3.6.1
   >>> (VibENV)% conda install -c conda-forge openbabel~=3.1.1
   >>> (VibENV)% conda install -c conda-forge pytest~=7.2.0

**Installation subpackages**

   >>> (VibENV) LocVib/src/% conda develop . 
   >>> (VibENV) pyadf/src/% conda develop .

optional for developing the Documentation:

   >>> (VibENV)% conda install -c conda-forge sphinx~=5.3.0
   >>> (VibENV)% pip install sphinx_rtd_theme~=0.4.3
   >>> (VibENV)% pip install sphinx_mdinclude~=0.5.3

Installation of *vibrations* itself:

   >>> (VibENV)LocVib% conda develop src/
   added /home/User/vibrations/src
   completed operation for: /home/User/vibrations/src


