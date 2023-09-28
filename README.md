# NDV_v1
NEMO Validation and Diagnostics. Developed as part of AdriaClim (OPA, CMCC).

NDV_v1 is a Python package designed primarily to help with the diagnostics and validation of the model outputs related to the AdriaClim project. It can be applied to other products as well. Follow the instructions below to install and use the tool.

## 1. Installation

Here you will clone the script to Zeus and create an environment called ndv_v1 with the necessary packages to run the script

```
$ cd folder_where_you_are_installing_the_package                    # to move inside the destination folder
$ git clone https://github.com/fabioviola-cmcc/NDV_v1.git           # to clone the repository
$ cd NDV_v1/                                                        # to move inside the cloned repository - we will refer to this folder as NDVROOT
$ source install.sh                                                 # to create environment
```

## 2. Use

The following steps will activate the environment and load the necessary modules. Then you **must** modify the file `ndv_input.py` to set the paths according to your needs, then run the script.

```
$ source setup.sh
$ cd scripts/
$ vi ndv_input.py
$ time bsub -n 144 -Is -q p_long -P 0419 python ndv_run.py          # example of execution
```

## 3. Output and files

- The output figures will be saved on `<NDVROOT>/output/<folder_name_of_your_choice>`
- Inside this folder, the current version of `ndv_input.py` will be saved so you have a record of the input commands used to create those figures
- The intermediate `.nc` files are saved on `<NDVROOT>/files`. These files are saved only during the first time you run a specific combination of product/variables/period, so running the script becomes much faster once these files are already saved.

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

## Development team

- Afonso Gonçalves Neto (CMCC, OPA division) - <afonso.goncalves@cmcc.it>
- Vladimir Santos da Costa (CMCC, OPA division) - <vladimir.santosdacosta@cmcc.it>
