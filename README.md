# NDV_v1
NEMO Validation and Diagnostics. Developed as part of AdriaClim (OPA, CMCC).

Afonso Gonçalves Neto - CMCC, OPA division
afonso.goncalves@cmcc.it

Lecce, IT, Apr 5, 2023
##################################################################
##################################################################

This file explains how to install and use the package NDV_v1. NDV_v1 is a Python package designed primarily to help with the diagnostics and validation of the model outputs related to the AdriaClim project. It can be applied to other products, however.

## 1. Installation

Here you will clone the script to Zeus and create an environment called ndv_v1 with the necessary packages to run the script

- cd folder_where_you_are_installing_the_package 
- git clone https://github.com/afonsogneto/NDV.git
- cd NDV_v1
- source install.sh (to create environment)

## 2. Use

Here you will activate the environment, load the necessary modules, modify the file ndv_input.py and run the script. Once on /NDV_v1:

- source setup.sh
- vi ndv_input.py
- Modify nvd_input.py according to your needs following the instructions in the file and save it
- bsub -Is -q p_long -P 0419 python ndv_run.py 

## 3. Output and files

- The output figures will be saved on /NDV_v1/output/folder_name_of_your_choice
- Inside this folder, the current version of ndv_input.py will be saved so you have a record of the input commands used to create those figures
- The intermediate .nc files are saved on NDV_v1/files. These files are saved only during the first time you run a specific combination of product/variables/period, so running the script becomes much faster once these files are already saved.

## 4. Produts

- AdriaClim Historical
- AdriaClim Projection
- CMEMS Reanalysis

## 5. Types of plots

- Daily time series of spatial mean
- Monthly time series of spatial mean
- Seasonal time series of spatial mean
- 2D map of mean field
- 2D map of monthly mean field
- 2D map of seasonal mean field
- Mean year

