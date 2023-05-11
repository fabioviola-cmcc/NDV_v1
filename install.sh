#!/bin/bash
module load anaconda
conda create -n ndv_v1 matplotlib numpy munch xarray netCDF4 scipy basemap scikit-learn seaborn dask basemap-data-hires

mkdir files
mkdir output

source activate ndv_v1

conda install -c conda-forge cmocean

conda deactivate