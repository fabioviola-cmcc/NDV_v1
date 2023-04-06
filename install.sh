#!/bin/bash
conda create -n ndv_v1 matplotlib numpy munch xarray netCDF4 scipy basemap scikit-learn seaborn dask

mkdir files
mkdir output

source activate ndv_v1

pip install cmocean

conda deactivate