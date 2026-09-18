# load: basic Docker image
FROM condaforge/miniforge3

# Create a "work folder"
WORKDIR /app

# Copy project in this work folder
COPY . /app

# Because of pyadf we unfortunately have to use conda here 12.04.2024
# PyADF Version 1.4
# Make RUN commands use the new environment:

# RUN
# Installation of local package
# Installation of dependencies for unittests
# Installation of local package
# Git Clone of PyADF
# Install pyadf requierements
RUN pip install . \
    && python -m pip install  ".[full-openbabel]" \
    && git clone https://github.com/chjacob-tubs/pyadf-releases.git \
    && conda install -c conda-forge numpy scipy xcfun pyscf openbabel rdkit  
# Install pyadf
RUN pip install /app/pyadf-releases/
