# load: basic Docker image
FROM python

# Create a "work folder"
WORKDIR /app

# Copy project in this work folder
COPY . /app

# RUN: Installation of dependencies and package itself
RUN python -m pip install  ".[full-openbabel]" \
    && pip install /app/

# Create example output files
WORKDIR /app/src/VibTools/example/

RUN python 1_composition.py > /app/doc/examples_output/1_output.txt \
    && python 2_locmodes.py > /app/doc/examples_output/2_output.txt \
    && python 3_couplings.py > /app/doc/examples_output/3_output.txt

WORKDIR /app/doc/

