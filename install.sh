#!/usr/bin/env bash

conda create --prefix ./condaenv
conda activate ./condaenv
conda install -c deepchem -c rdkit -c conda-forge -c omnia deepchem=2.3.0