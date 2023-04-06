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

a) cd folder_where_you_are_installing_the_package 
b) git clone https://github.com/afonsogneto/NDV.git
c) cd NDV_v1
e) install.sh (to create environment)

## 2. Use

Here you will activate the environment, load the necessary modules, modify the file ndv_input.py and run the script. Once on /NDV_v1:

a) setup.sh
b) vi ndv_input.py
c) Modify nvd_input.py according to your needs following the instructions in the file and save it
d) bsub -Is -q p_long -P 0419 python ndv_run.py 

## 3. Output and files

a) The output figures will be saved on /NDV_v1/output/folder_name_of_your_choice
b) Inside this folder, the current version of ndv_input.py will be saved so you have a record of the input commands used to create those figures
c) The intermediate .nc files are saved on NDV_v1/files. These files are saved only during the first time you run a specific combination of product/variables/period, so running the script becomes much faster once these files are already saved.

## 4. Produts

a) AdriaClim Historical
b) AdriaClim Projection
c) CMEMS Reanalysis

## 5. Types of plots

a) Daily time series of spatial mean
b) Monthly time series of spatial mean
c) Seasonal time series of spatial mean
d) 2D map of mean field
e) 2D map of monthly mean field
f) 2D map of seasonal mean field
g) Mean year

