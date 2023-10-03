##################################################################
##################################################################
# Nemo Diagnostics & Validation (NDV) package
# ndv_jobs.py
#
# Afonso Gonçalves Neto - CMCC, OPA division
# afonso.goncalves@cmcc.it
#
# Lecce, IT, December 29, 2022
##################################################################
##################################################################
## Import packages
from ndv_tools import *
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import xarray as xr
import netCDF4 as nc
import scipy
import os
import shutil
from mpl_toolkits.basemap import Basemap
import datetime
from datetime import date, timedelta
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from cmocean import cm
from munch import Munch
import math
from sklearn.metrics import mean_squared_error
import seaborn as sns
sns.set_theme(style="whitegrid")
##################################################################

# Function to load NetCDF files
def load_data_historical(path_input, file_input, path_mask, file_mask, run_input, freq_input, timei, timef, area_input, var_input, path_output, subpath_output, path_file, nemo_version):

    # If run refers to NEMO historical output, load data. Otherwise, create empty datasets 
    if 'historical' in run_input:
        ## Create list of timesteps to be loaded
        timei_this = timei[run_input.index("historical")]
        timef_this = timef[run_input.index("historical")]
        time_list = create_time_list(timei_this, timef_this)

        # Define current time used to define the name of the output directory
        current_time = define_current_time()

        # Create new output folder for current request
        if subpath_output == "":
            os.makedirs(os.path.join(path_output, f'{current_time}/'))
            path_output_this = os.path.join(path_output, f'{current_time}/')
            # Create subfolders in output folder
            os.makedirs(os.path.join(path_output_this, 'maps/'))
            os.makedirs(os.path.join(path_output_this, 'timeseries/'))
        else:
            if os.path.exists(os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')):
                print(f"Directory already exists: {os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')}")
                writeoption = input("Append figures to existing directory (a), overwrite directory (o) or quit (q)? ")
                if writeoption == 'A' or writeoption == 'a':
                    path_output_this = os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')
                elif writeoption == 'O' or writeoption == 'o':    
                    shutil.rmtree(os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/'))
                    os.makedirs(os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/'))
                    path_output_this = os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')
                    # Create subfolders in output folder
                    os.makedirs(os.path.join(path_output_this, 'maps/'))
                    os.makedirs(os.path.join(path_output_this, 'timeseries/'))
                elif writeoption == 'Q' or writeoption == 'q':
                    raise ValueError('Change the output subdirectory path before running the script again.')
                else:
                    raise ValueError('Invalid input.')
            else:
                os.makedirs(os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/'))
                path_output_this = os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')
                # Create subfolders in output folder
                os.makedirs(os.path.join(path_output_this, 'maps/'))
                os.makedirs(os.path.join(path_output_this, 'timeseries/'))

        # Attribute the NEMO var names to the plotted variables
        var_list2 = create_var_list2(var_input)

        # Associate grid to each input variable
        grid_input = create_grid_input(var_input)

        # Define var long names for each input variable
        var_long = create_var_long(var_input)

        # Define colorbar titles for each input variable
        cbar_title = create_cbar_title(var_input)

        # Define colormaps for each input variable
        cmap_list = create_cmap_list(var_input)

        # Create new var input variable
        var_input_new = var_input.copy()
        # Get file path
        path_file_this = (f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        # Check if intermediate files exist and load files
        if os.path.exists(path_file_this):
            ds_T_historical_this = nc.Dataset(f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            ds_U_historical_this = nc.Dataset(f'{path_file}historical_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            ds_V_historical_this = nc.Dataset(f'{path_file}historical_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            ds_W_historical_this = nc.Dataset(f'{path_file}historical_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            for i in range(0,len(var_input)):
                if 'historical' in run_input:
                    # Remove variable from var_input_new if it already exists in intermediate file
                    if grid_input[i] == 'T':
                        if f'{var_input[i]}_dailytimeseries' in ds_T_historical_this.variables.keys():
                            var_input_new.remove(var_input[i])
                    if grid_input[i] == 'U':
                        if f'{var_input[i]}_dailytimeseries' in ds_U_historical_this.variables.keys():
                            var_input_new.remove(var_input[i])
                    if grid_input[i] == 'V':
                        if f'{var_input[i]}_dailytimeseries' in ds_V_historical_this.variables.keys():
                            var_input_new.remove(var_input[i])
                    if grid_input[i] == 'W':
                        if f'{var_input[i]}_dailytimeseries' in ds_W_historical_this.variables.keys():
                            var_input_new.remove(var_input[i])

        if var_input_new != []:

            # Define original and calculated variables
            var_original = ['T', 'S', 'SST', 'SSS', 'SSH', 'WaterFlux', 'SaltFlux', 'HeatFlux', 'MLD', 'TurboclineDepth', 
                            'U', 'EIVU', 'TauX', 'V', 'EIVV', 'TauY', 'W', 'EIVW', 'VertEddyVisc', 'LatEddyDiff']
            var_calculated = {
              "SurfVel": ["U","V"],
              "T_VertInt": ["T"],
            }

            # Create list of variables to be loaded
            var_load = []
            for var in var_input_new:
                if var in var_original:
                    var_load+=[var]
                else:
                    var_load+=var_calculated[var]

            # Attribute the NEMO var names to the loaded variables
            var_list = create_var_list(var_load)

            # Associate grid to each loaded variable
            grid_list = create_grid_list(var_load)

            # Create filename for the first file based on the input parameters
            # Define frequency
            if freq_input == 'day':
                freq = '1d'
            elif freq_input == '3h':
                freq = '3h'
            # Define grids (T, U, V or W)
            var_aux_T = ['T', 'S', 'SST', 'SSS', 'SSH', 'WaterFlux', 'SaltFlux', 'HeatFlux', 'MLD', 'TurboclineDepth']
            var_aux_U = ['U', 'EIVU', 'TauX']
            var_aux_V = ['V', 'EIVV', 'TauY']
            var_aux_W = ['W', 'EIVW', 'VertEddyDiff', 'VertEddyVisc', 'LatEddyDiff', 'VelWPoint']
            for var in var_aux_T:
                if var in var_load:
                    grid = 'T'
                    file_input_this_T = eval(file_input[run_input.index("historical")])
                    file_input_this_T = file_input_this_T.replace('YYYYMMDD','*')
                    break
            for var in var_aux_U:
                if var in var_load:
                    grid = 'U'
                    file_input_this_U = eval(file_input[run_input.index("historical")])
                    file_input_this_U = file_input_this_U.replace('YYYYMMDD','*')
                    break
            for var in var_aux_V:
                if var in var_load:
                    grid = 'V'
                    file_input_this_V = eval(file_input[run_input.index("historical")])
                    file_input_this_V = file_input_this_V.replace('YYYYMMDD','*')
                    break
            for var in var_aux_W:
                if var in var_load:
                    grid = 'W'
                    file_input_this_W = eval(file_input[run_input.index("historical")])
                    file_input_this_W = file_input_this_W.replace('YYYYMMDD','*')
                    break
            del var_aux_T, var_aux_U, var_aux_V, var_aux_W, grid

            # Create directory path for the first month of data
            # Change path if current month is during spinup
            if timei_this == '199201' or timei_this == '199202' or timei_this == '199203':
                path_input_this = path_input[run_input.index("historical")]+run_input[run_input.index("historical")]+'/'+freq_input+'/'+timei_this[0:4]+'/'+timei_this[4:6]+'/'
            else:
                path_input_this = path_input[run_input.index("historical")]+run_input[run_input.index("historical")]+'/'+freq_input+'/'+timei_this[0:4]+'/'+timei_this[4:6]+'/'

            # Create dataset with data from the first month
            # Grid T, if grid T variables were selected
            if 'file_input_this_T' in locals():
                print('    '+path_input_this+file_input_this_T)
                # Open datasets
                ds_T_historical = xr.open_mfdataset(path_input_this+file_input_this_T, parallel = True)
                # Drop one dimension from spatial coordinates
                if nemo_version=='3.6' or nemo_version=='4.2':
                    df_msk = xr.open_dataset(f'{path_mask[run_input.index("historical")]}{file_mask[run_input.index("historical")]}')
                    ds_T_historical['nav_lat'] = df_msk['nav_lat'].sel(x=1, drop=True)
                    ds_T_historical['nav_lon'] = df_msk['nav_lon'].sel(y=1, drop=True)
                    ds_T_historical = ds_T_historical.rename({'deptht': 'depth'})
                if nemo_version=='ERDDAP':
                    ds_T_historical = ds_T_historical.rename({'lat': 'nav_lat'})
                    ds_T_historical = ds_T_historical.rename({'lon': 'nav_lon'})
                    ds_T_historical = ds_T_historical.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})
                if nemo_version=='4.2':
                    ds_T_historical = ds_T_historical.rename({'tos': 'sosstsst'})
                    ds_T_historical = ds_T_historical.rename({'sos': 'sosaline'})
                    ds_T_historical = ds_T_historical.rename({'zos': 'sossheig'})
                    ds_T_historical = ds_T_historical.rename({'thetao': 'votemper'})
                    ds_T_historical = ds_T_historical.rename({'so': 'vosaline'})

                # Set index for spatial coordinates
                ds_T_historical = ds_T_historical.set_xindex("nav_lat")
                ds_T_historical = ds_T_historical.set_xindex("nav_lon")
                # Slice dataset according to area_input
                ds_T_historical = ds_T_historical.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                ind_T = list(np.where(np.array(grid_list)=='T')[0])
                var_list_T = []
                for ind in ind_T:
                    var_list_T.append(var_list[ind])
                ds_T_historical = ds_T_historical[var_list_T]
            # Grid U, if grid U variables were selected
            if 'file_input_this_U' in locals():
                print('    '+path_input_this+file_input_this_U)
                # Open datasets
                ds_U_historical = xr.open_mfdataset(path_input_this+file_input_this_U, parallel = True)
                # Drop one dimension from spatial coordinates
                if nemo_version=='3.6' or nemo_version=='4.2':
                    ds_U_historical['nav_lat'] = df_msk['nav_lat'].sel(x=1, drop=True)
                    ds_U_historical['nav_lon'] = df_msk['nav_lon'].sel(y=1, drop=True)
                    ds_U_historical = ds_U_historical.rename({'depthu': 'depth'})
                if nemo_version=='ERDDAP':
                    ds_U_historical = ds_U_historical.rename({'lat': 'nav_lat'})
                    ds_U_historical = ds_U_historical.rename({'lon': 'nav_lon'})
                    ds_U_historical = ds_U_historical.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})
                # Set index for spatial coordinates
                ds_U_historical = ds_U_historical.set_xindex("nav_lat")
                ds_U_historical = ds_U_historical.set_xindex("nav_lon")
                # Slice dataset according to area_input
                ds_U_historical = ds_U_historical.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                ind_U = list(np.where(np.array(grid_list)=='U')[0])
                var_list_U = []
                for ind in ind_U:
                    var_list_U.append(var_list[ind])
                ds_U_historical = ds_U_historical[var_list_U]
            # Grid V, if grid V variables were selected
            if 'file_input_this_V' in locals():
                print('    '+path_input_this+file_input_this_V)
                # Open datasets
                ds_V_historical = xr.open_mfdataset(path_input_this+file_input_this_V, parallel = True)
                # Drop one dimension from spatial coordinates
                if nemo_version=='3.6' or nemo_version=='4.2':
                    ds_V_historical['nav_lat'] = ds_msk['nav_lat'].sel(x=1, drop=True)
                    ds_V_historical['nav_lon'] = ds_msk['nav_lon'].sel(y=1, drop=True)
                    ds_V_historical = ds_V_historical.rename({'depthv': 'depth'})
                if nemo_version=='ERDDAP':
                    ds_V_historical = ds_V_historical.rename({'lat': 'nav_lat'})
                    ds_V_historical = ds_V_historical.rename({'lon': 'nav_lon'})
                    ds_V_historical = ds_V_historical.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})
                # Set index for spatial coordinates
                ds_V_historical = ds_V_historical.set_xindex("nav_lat")
                ds_V_historical = ds_V_historical.set_xindex("nav_lon")
                # Slice dataset according to area_input
                ds_V_historical = ds_V_historical.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                ind_V = list(np.where(np.array(grid_list)=='V')[0])
                var_list_V = []
                for ind in ind_V:
                    var_list_V.append(var_list[ind])
                ds_V_historical = ds_V_historical[var_list_V]
            # Grid W, if grid W variables were selected
            if 'file_input_this_W' in locals():
                print('    '+path_input_this+file_input_this_W)
                # Open datasets
                ds_W_historical = xr.open_mfdataset(path_input_this+file_input_this_W, parallel = True)
                # Drop one dimension from spatial coordinates
                if nemo_version=='3.6' or nemo_version=='4.2':
                    ds_W_historical['nav_lat'] = ds_msk['nav_lat'].sel(x=1, drop=True)
                    ds_W_historical['nav_lon'] = ds_msk['nav_lon'].sel(y=1, drop=True)
                    ds_W_historical = ds_W_historical.rename({'depthw': 'depth'})
                if nemo_version=='ERDDAP':
                    ds_W_historical = ds_W_historical.rename({'lat': 'nav_lat'})
                    ds_W_historical = ds_W_historical.rename({'lon': 'nav_lon'})
                    ds_W_historical = ds_W_historical.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})
                # Set index for spatial coordinates
                ds_W_historical = ds_W_historical.set_xindex("nav_lat")
                ds_W_historical = ds_W_historical.set_xindex("nav_lon")
                # Slice dataset according to area_input
                ds_W_historical = ds_W_historical.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                ind_W = list(np.where(np.array(grid_list)=='W')[0])
                var_list_W = []
                for ind in ind_W:
                    var_list_W.append(var_list[ind])
                ds_W_historical = ds_W_historical[var_list_W]

            del path_input_this

            # Loop over other months to update the directory path
            for t in time_list[1:]:
                # Change path if current month is during spinup
                if t == '199201' or t == '199202' or t == '199203':
                    path_input_this = path_input[run_input.index("historical")]+run_input[run_input.index("historical")]+'/'+freq_input+'/'+t[0:4]+'/'+t[4:6]+'/'
                else:
                    path_input_this = path_input[run_input.index("historical")]+run_input[run_input.index("historical")]+'/'+freq_input+'/'+t[0:4]+'/'+t[4:6]+'/'

                # Open dataset in grid T, if grid T variables were selected
                if 'file_input_this_T' in locals():
                    print('    '+path_input_this+file_input_this_T)
                    # Open datasets
                    ds_this = xr.open_mfdataset(path_input_this+file_input_this_T, parallel = True)
                    # Drop one dimension from spatial coordinates
                    if nemo_version=='3.6' or nemo_version=='4.2':
                        ds_this['nav_lat'] = df_msk['nav_lat'].sel(x=1, drop=True)
                        ds_this['nav_lon'] = df_msk['nav_lon'].sel(y=1, drop=True)
                        ds_this = ds_this.rename({'deptht': 'depth'})
                    if nemo_version=='ERDDAP':
                        ds_this = ds_this.rename({'lat': 'nav_lat'})
                        ds_this = ds_this.rename({'lon': 'nav_lon'})
                        ds_this = ds_this.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})
                    if nemo_version=='4.2':
                        ds_this = ds_this.rename({'tos': 'sosstsst'})
                        ds_this = ds_this.rename({'sos': 'sosaline'})
                        ds_this = ds_this.rename({'zos': 'sossheig'})
                        ds_this = ds_this.rename({'thetao': 'votemper'})
                        ds_this = ds_this.rename({'so': 'vosaline'})

                    # Set index for spatial coordinates
                    ds_this = ds_this.set_xindex("nav_lat")
                    ds_this = ds_this.set_xindex("nav_lon")
                    # Slice dataset according to area_input
                    ds_this = ds_this.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                    ds_this = ds_this[var_list_T]
                    # Concatenate data to create a single dataset
                    ds_T_historical = xr.concat([ds_T_historical, ds_this], dim = "time_counter")
                    del ds_this

                # Open dataset in grid U, if grid U variables were selected
                if 'file_input_this_U' in locals():
                    print('    '+path_input_this+file_input_this_U)
                    # Open datasets
                    ds_this = xr.open_mfdataset(path_input_this+file_input_this_U, parallel = True)
                    # Drop one dimension from spatial coordinates
                    if nemo_version=='3.6' or nemo_version=='4.2':
                        ds_this['nav_lat'] = ds_msk['nav_lat'].sel(x=1, drop=True)
                        ds_this['nav_lon'] = ds_msk['nav_lon'].sel(y=1, drop=True)
                        ds_this = ds_this.rename({'depthu': 'depth'})
                    if nemo_version=='ERDDAP':
                        ds_this = ds_this.rename({'lat': 'nav_lat'})
                        ds_this = ds_this.rename({'lon': 'nav_lon'})
                        ds_this = ds_this.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})
                    # Set index for spatial coordinates
                    ds_this = ds_this.set_xindex("nav_lat")
                    ds_this = ds_this.set_xindex("nav_lon")
                    # Slice dataset according to area_input
                    ds_this = ds_this.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                    ds_this = ds_this[var_list_U]
                    # Concatenate data to create a single dataset
                    ds_U_historical = xr.concat([ds_U_historical, ds_this], dim = "time_counter")
                    del ds_this

                # Open dataset in grid V, if grid V variables were selected
                if 'file_input_this_V' in locals():
                    print('    '+path_input_this+file_input_this_V)
                    # Open datasets
                    ds_this = xr.open_mfdataset(path_input_this+file_input_this_V, parallel = True)
                    # Drop one dimension from spatial coordinates
                    if nemo_version=='3.6' or nemo_version=='4.2':
                        ds_this['nav_lat'] = ds_msk['nav_lat'].sel(x=1, drop=True)
                        ds_this['nav_lon'] = ds_msk['nav_lon'].sel(y=1, drop=True)
                        ds_this = ds_this.rename({'depthv': 'depth'})
                    if nemo_version=='ERDDAP':
                        ds_this = ds_this.rename({'lat': 'nav_lat'})
                        ds_this = ds_this.rename({'lon': 'nav_lon'})
                        ds_this = ds_this.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})
                    # Set index for spatial coordinates
                    ds_this = ds_this.set_xindex("nav_lat")
                    ds_this = ds_this.set_xindex("nav_lon")
                    # Slice dataset according to area_input
                    ds_this = ds_this.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                    ds_this = ds_this[var_list_V]
                    # Concatenate data to create a single dataset
                    ds_V_historical = xr.concat([ds_V_historical, ds_this], dim = "time_counter")
                    del ds_this

                # Open dataset in grid W, if grid W variables were selected
                if 'file_input_this_W' in locals():
                    print('    '+path_input_this+file_input_this_W)
                    # Open datasets
                    ds_this = xr.open_mfdataset(path_input_this+file_input_this_W, parallel = True)
                    # Drop one dimension from spatial coordinates
                    if nemo_version=='3.6' or nemo_version=='4.2':
                        ds_this['nav_lat'] = ds_msk['nav_lat'].sel(x=1, drop=True)
                        ds_this['nav_lon'] = ds_msk['nav_lon'].sel(y=1, drop=True)
                        ds_this = ds_this.rename({'depthw': 'depth'})
                    if nemo_version=='ERDDAP':
                        ds_this = ds_this.rename({'lat': 'nav_lat'})
                        ds_this = ds_this.rename({'lon': 'nav_lon'})
                        ds_this = ds_this.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})
                    # Set index for spatial coordinates
                    ds_this = ds_this.set_xindex("nav_lat")
                    ds_this = ds_this.set_xindex("nav_lon")
                    # Slice dataset according to area_input
                    ds_this = ds_this.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                    ds_this = ds_this[var_list_W]
                    # Concatenate data to create a single dataset
                    ds_W_historical = xr.concat([ds_W_historical, ds_this], dim = "time_counter")
                    del ds_this

                del path_input_this

            # Load mask file to apply to DataArrays
            ds_mask = xr.open_dataset(f'{path_mask[run_input.index("historical")]}{file_mask[run_input.index("historical")]}')
            # Open T mask if T variables were selected
            if 'file_input_this_T' in locals():
                mask = ds_mask['tmask'].isel(z = slice(0,1)).squeeze()
                bh =np.squeeze(np.nansum(np.squeeze(ds_mask['e3t_0'][0,:,:,:]*ds_mask['tmask']),axis=0))
                for var in var_list_T:
                    ds_T_historical[var] = ds_T_historical[var].where(mask==1)
                del mask
            # Open U mask if U variables were selected
            if 'file_input_this_U' in locals():
                mask = ds_mask['umask'].isel(z = slice(0,1)).squeeze()
                for var in var_list_U:
                    ds_U_historical[var] = ds_U_historical[var].where(mask==1)
                del mask
            # Open V mask if V variables were selected
            if 'file_input_this_V' in locals():
                mask = ds_mask['vmask'].isel(z = slice(0,1)).squeeze()
                for var in var_list_V:
                    ds_V_historical[var] = ds_V_historical[var].where(mask==1)
                del mask
            # Open W mask if W variables were selected
            if 'file_input_this_W' in locals():
                mask = ds_mask['fmask'].isel(z = slice(0,1)).squeeze()
                for var in var_list_W:
                    ds_W_historical[var] = ds_W_historical[var].where(mask==1)
                del mask

            del ds_mask

            # Compute calculated variables and delete unused DataArrays
            for var in var_input_new:
                if var == "SurfVel":
                    # Calculate Surface Velocity and add it to respective Dataset
                    ds_U_historical_interp, ds_V_historical_interp = xr.align(ds_U_historical, ds_V_historical, join="override")
                    ds_U_historical['SurfVel'] = np.sqrt(ds_U_historical_interp['vozocrtx'].isel(depth=0)**2 + ds_V_historical_interp['vomecrty'].isel(depth=0)**2)
                    # If U is not called, remove it from the Dataset and var_list
                    if 'U' not in var_input:
                        ds_U_historical = ds_U_historical.drop(var_list[var_load.index('U')])
                    # If V is not called, remove it from the Dataset and var_list
                    if 'V' not in var_input:
                        ds_V_historical = ds_V_historical.drop(var_list[var_load.index('V')])
                if var == "T_VertInt":
                    # Calculate Vertically-Integrated Temperature and add it to respective Dataset
                    ds_T_historical['T_VertInt'] = ds_T_historical['votemper'].integrate("depth")
                    bhf=[]
                    for i in range(0,len(ds_T_historical['T_VertInt'][:,0,0])):
                        bhf.append(bh)
                    ds_T_historical['T_VertInt'] = ds_T_historical['T_VertInt']/bhf
                    # If T is not called, remove it from the Dataset and var_list
                    if 'T' not in var_input:
                        ds_T_historical = ds_T_historical.drop(var_list[var_load.index('T')])

            # If any of the grids was not used, create an empty dataset for that grid
            if 'ds_T_historical' not in locals():
                ds_T_historical = xr.Dataset()
            if 'ds_U_historical' not in locals():
                ds_U_historical = xr.Dataset()
            if 'ds_V_historical' not in locals():
                ds_V_historical = xr.Dataset()
            if 'ds_W_historical' not in locals():
                ds_W_historical = xr.Dataset()  

            # Load datasets
            if 'ds_T_historical' in locals():
                ds_T_historical = ds_T_historical.load()
            if 'ds_U_historical' in locals():
                ds_U_historical = ds_U_historical.load()
            if 'ds_V_historical' in locals():
                ds_V_historical = ds_V_historical.load()
            if 'ds_W_historical' in locals():
                ds_W_historical = ds_W_historical.load()

        else:
            ds_T_historical = xr.Dataset()
            ds_U_historical = xr.Dataset()
            ds_V_historical = xr.Dataset()
            ds_W_historical = xr.Dataset()

            # Create empty output path
            #path_output_this = ''

    else:
        ds_T_historical = xr.Dataset()
        ds_U_historical = xr.Dataset()
        ds_V_historical = xr.Dataset()
        ds_W_historical = xr.Dataset()

        # Attribute the NEMO var names to the plotted variables
        var_list2 = create_var_list2(var_input)

        # Associate grid to each input variable
        grid_input = create_grid_input(var_input)

        # Define var long names for each input variable
        var_long = create_var_long(var_input)

        # Define colorbar titles for each input variable
        cbar_title = create_cbar_title(var_input)

        # Define colormaps for each input variable
        cmap_list = create_cmap_list(var_input)

        # Create empty output path
        path_output_this = ''

    return(ds_T_historical, ds_U_historical, ds_V_historical, ds_W_historical, var_list2, var_long, grid_input, cbar_title, cmap_list, path_output_this)

# Function to load NetCDF files
def load_data_projection(path_input, file_input, path_mask, file_mask, run_input, freq_input, timei, timef, area_input, var_input, path_output, subpath_output, path_output_this, path_file, nemo_version):
    # If run refers to NEMO, load data. Otherwise, create empty datasets 
    if 'projection' in run_input:
        ## Create list of timesteps to be loaded
        timei_this = timei[run_input.index("projection")]
        timef_this = timef[run_input.index("projection")]
        time_list = create_time_list(timei_this, timef_this)

        # Define current time used to define the name of the output directory
        current_time = define_current_time()

        # Create new output folder for current request (only if not running a comparison with historical run)
        if "historical" not in run_input:
            # Create new output folder for current request
            if subpath_output == "":
                os.makedirs(os.path.join(path_output, f'{current_time}/'))
                path_output_this = os.path.join(path_output, f'{current_time}/')
                # Create subfolders in output folder
                os.makedirs(os.path.join(path_output_this, 'maps/'))
                os.makedirs(os.path.join(path_output_this, 'timeseries/'))
            else:
                if os.path.exists(os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')):
                    print(f"Directory already exists: {os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')}")
                    writeoption = input("Append figures to existing directory (a), overwrite directory (o) or quit (q)? ")
                    if writeoption == 'A' or writeoption == 'a':
                        path_output_this = os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')
                    elif writeoption == 'O' or writeoption == 'o':    
                        shutil.rmtree(os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/'))
                        os.makedirs(os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/'))
                        path_output_this = os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')
                        # Create subfolders in output folder
                        os.makedirs(os.path.join(path_output_this, 'maps/'))
                        os.makedirs(os.path.join(path_output_this, 'timeseries/'))
                    elif writeoption == 'Q' or writeoption == 'q':
                        raise ValueError('Change the output subdirectory path before running the script again.')
                    else:
                        raise ValueError('Invalid input.')
                else:
                    os.makedirs(os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/'))
                    path_output_this = os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')
                    # Create subfolders in output folder
                    os.makedirs(os.path.join(path_output_this, 'maps/'))
                    os.makedirs(os.path.join(path_output_this, 'timeseries/'))

        # Attribute the NEMO var names to the plotted variables
        var_list2 = create_var_list2(var_input)

        # Associate grid to each input variable
        grid_input = create_grid_input(var_input)

        # Define var long names for each input variable
        var_long = create_var_long(var_input)

        # Define colorbar titles for each input variable
        cbar_title = create_cbar_title(var_input)

        # Define colormaps for each input variable
        cmap_list = create_cmap_list(var_input)

        # Create new var input variable
        var_input_new = var_input.copy()
        # Get file path
        path_file_this = (f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        # Check if intermediate files exist and load files
        if os.path.exists(path_file_this):
            ds_T_projection_this = nc.Dataset(f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            ds_U_projection_this = nc.Dataset(f'{path_file}projection_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            ds_V_projection_this = nc.Dataset(f'{path_file}projection_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            ds_W_projection_this = nc.Dataset(f'{path_file}projection_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            for i in range(0,len(var_input)):
                # Remove variable from var_input_new if it already exists in intermediate file
                if grid_input[i] == 'T':
                    if f'{var_input[i]}_dailytimeseries' in ds_T_projection_this.variables.keys():
                        var_input_new.remove(var_input[i])
                if grid_input[i] == 'U':
                    if f'{var_input[i]}_dailytimeseries' in ds_U_projection_this.variables.keys():
                        var_input_new.remove(var_input[i])
                if grid_input[i] == 'V':
                    if f'{var_input[i]}_dailytimeseries' in ds_V_projection_this.variables.keys():
                        var_input_new.remove(var_input[i])
                if grid_input[i] == 'W':
                    if f'{var_input[i]}_dailytimeseries' in ds_W_projection_this.variables.keys():
                        var_input_new.remove(var_input[i])

        if var_input_new != []:

            # Define original and calculated variables
            var_original = ['T', 'S', 'SST', 'SSS', 'SSH', 'WaterFlux', 'SaltFlux', 'HeatFlux', 'MLD', 'TurboclineDepth', 
                            'U', 'EIVU', 'TauX', 'V', 'EIVV', 'TauY', 'W', 'EIVW', 'VertEddyVisc', 'LatEddyDiff']
            var_calculated = {
              "SurfVel": ["U","V"],
              "T_VertInt": ["T"],
            }

            # Create list of variables to be loaded
            var_load = []
            for var in var_input_new:
                if var in var_original:
                    var_load+=[var]
                else:
                    var_load+=var_calculated[var]

            # Attribute the NEMO var names to the loaded variables
            var_list = create_var_list(var_load)

            # Associate grid to each loaded variable
            grid_list = create_grid_list(var_load)

            # Create filename for the first file based on the input parameters
            # Define frequency
            if freq_input == 'day':
                freq = '1d'
            elif freq_input == '3h':
                freq = '3h'
            # Define grids (T, U, V or W)
            var_aux_T = ['T', 'S', 'SST', 'SSS', 'SSH', 'WaterFlux', 'SaltFlux', 'HeatFlux', 'MLD', 'TurboclineDepth']
            var_aux_U = ['U', 'EIVU', 'TauX']
            var_aux_V = ['V', 'EIVV', 'TauY']
            var_aux_W = ['W', 'EIVW', 'VertEddyDiff', 'VertEddyVisc', 'LatEddyDiff', 'VelWPoint']
            for var in var_aux_T:
                if var in var_load:
                    grid = 'T'
                    file_input_this_T = eval(file_input[run_input.index("projection")])
                    file_input_this_T = file_input_this_T.replace('YYYYMMDD','*')
                    break
            for var in var_aux_U:
                if var in var_load:
                    grid = 'U'
                    file_input_this_U = eval(file_input[run_input.index("projection")])
                    file_input_this_U = file_input_this_U.replace('YYYYMMDD','*')
                    break
            for var in var_aux_V:
                if var in var_load:
                    grid = 'V'
                    file_input_this_V = eval(file_input[run_input.index("projection")])
                    file_input_this_V = file_input_this_V.replace('YYYYMMDD','*')
                    break
            for var in var_aux_W:
                if var in var_load:
                    grid = 'W'
                    file_input_this_W = eval(file_input[run_input.index("projection")])
                    file_input_this_W = file_input_this_W.replace('YYYYMMDD','*')
                    break
            del var_aux_T, var_aux_U, var_aux_V, var_aux_W, grid

            # Create directory path for the first month of data
            # Change path if current month is during spinup
            if timei_this == '202201' or timei_this == '202202' or timei_this == '202203':
                path_input_this = path_input[run_input.index("projection")]+run_input[run_input.index("projection")]+'/'+freq_input+'/'+timei_this[0:4]+'/'+timei_this[4:6]+'/'
            else:
                path_input_this = path_input[run_input.index("projection")]+run_input[run_input.index("projection")]+'/'+freq_input+'/'+timei_this[0:4]+'/'+timei_this[4:6]+'/'

            # Create dataset with data from the first month
            # Grid T, if grid T variables were selected
            if 'file_input_this_T' in locals():
                print('    '+path_input_this+file_input_this_T)
                # Open datasets
                ds_T_projection = xr.open_mfdataset(path_input_this+file_input_this_T, parallel = True)
                # Drop one dimension from spatial coordinates
                #ds_T_projection['nav_lat'] = ds_T_projection['nav_lat'].sel(x=1, drop=True)
                #ds_T_projection['nav_lon'] = ds_T_projection['nav_lon'].sel(y=1, drop=True)
                #gverri: replace two line above with 3 below cause of NEMO conversion intoo Erddap format
                ds_T_projection = ds_T_projection.rename({'lat': 'nav_lat'})
                ds_T_projection = ds_T_projection.rename({'lon': 'nav_lon'})
                ds_T_projection = ds_T_projection.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})

                # Set index for spatial coordinates
                ds_T_projection = ds_T_projection.set_xindex("nav_lat")
                ds_T_projection = ds_T_projection.set_xindex("nav_lon")
                # Slice dataset according to area_input
                ds_T_projection = ds_T_projection.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                ind_T = list(np.where(np.array(grid_list)=='T')[0])
                var_list_T = []
                for ind in ind_T:
                    var_list_T.append(var_list[ind])
                ds_T_projection = ds_T_projection[var_list_T]
            # Grid U, if grid U variables were selected
            if 'file_input_this_U' in locals():
                print('    '+path_input_this+file_input_this_U)
                # Open datasets
                ds_U_projection = xr.open_mfdataset(path_input_this+file_input_this_U, parallel = True)
                # Drop one dimension from spatial coordinates
                #ds_U_projection['nav_lat'] = ds_U_projection['nav_lat'].sel(x=1, drop=True)
                #ds_U_projection['nav_lon'] = ds_U_projection['nav_lon'].sel(y=1, drop=True)
                #gverri: replace two line above with 3 below cause of NEMO conversion intoo Erddap format
                ds_U_projection = ds_U_projection.rename({'lat': 'nav_lat'})
                ds_U_projection = ds_U_projection.rename({'lon': 'nav_lon'})
                ds_U_projection = ds_U_projection.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})

                # Set index for spatial coordinates
                ds_U_projection = ds_U_projection.set_xindex("nav_lat")
                ds_U_projection = ds_U_projection.set_xindex("nav_lon")
                # Slice dataset according to area_input
                ds_U_projection = ds_U_projection.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                ind_U = list(np.where(np.array(grid_list)=='U')[0])
                var_list_U = []
                for ind in ind_U:
                    var_list_U.append(var_list[ind])
                ds_U_projection = ds_U_projection[var_list_U]
            # Grid V, if grid V variables were selected
            if 'file_input_this_V' in locals():
                print('    '+path_input_this+file_input_this_V)
                # Open datasets
                ds_V_projection = xr.open_mfdataset(path_input_this+file_input_this_V, parallel = True)
                # Drop one dimension from spatial coordinates
                #ds_V_projection['nav_lat'] = ds_V_projection['nav_lat'].sel(x=1, drop=True)
                #ds_V_projection['nav_lon'] = ds_V_projection['nav_lon'].sel(y=1, drop=True)
                #gverri: replace two line above with 3 below cause of NEMO conversion intoo Erddap format
                ds_V_projection = ds_V_projection.rename({'lat': 'nav_lat'})
                ds_V_projection = ds_V_projection.rename({'lon': 'nav_lon'})
                ds_V_projection = ds_V_projection.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})

                # Set index for spatial coordinates
                ds_V_projection = ds_V_projection.set_xindex("nav_lat")
                ds_V_projection = ds_V_projection.set_xindex("nav_lon")
                # Slice dataset according to area_input
                ds_V_projection = ds_V_projection.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                ind_V = list(np.where(np.array(grid_list)=='V')[0])
                var_list_V = []
                for ind in ind_V:
                    var_list_V.append(var_list[ind])
                ds_V_projection = ds_V_projection[var_list_V]
            # Grid W, if grid W variables were selected
            if 'file_input_this_W' in locals():
                print('    '+path_input_this+file_input_this_W)
                # Open datasets
                ds_W_projection = xr.open_mfdataset(path_input_this+file_input_this_W, parallel = True)
                # Drop one dimension from spatial coordinates
                ds_W_projection['nav_lat'] = ds_W_projection['nav_lat'].sel(x=1, drop=True)
                ds_W_projection['nav_lon'] = ds_W_projection['nav_lon'].sel(y=1, drop=True)
                #gverri: replace two line above with 3 below cause of NEMO conversion intoo Erddap format
                ds_W_projection = ds_W_projection.rename({'lat': 'nav_lat'})
                ds_W_projection = ds_W_projection.rename({'lon': 'nav_lon'})
                ds_W_projection = ds_W_projection.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})

                # Set index for spatial coordinates
                ds_W_projection = ds_W_projection.set_xindex("nav_lat")
                ds_W_projection = ds_W_projection.set_xindex("nav_lon")
                # Slice dataset according to area_input
                ds_W_projection = ds_W_projection.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                ind_W = list(np.where(np.array(grid_list)=='W')[0])
                var_list_W = []
                for ind in ind_W:
                    var_list_W.append(var_list[ind])
                ds_W_projection = ds_W_projection[var_list_W]

            del path_input_this

            # Loop over other months to update the directory path
            for t in time_list[1:]:
                # Change path if current month is during spinup
                if t == '202201' or t == '202202' or t == '202203':
                    path_input_this = path_input[run_input.index("projection")]+run_input[run_input.index("projection")]+'/'+freq_input+'/'+t[0:4]+'/'+t[4:6]+'/'
                else:
                    path_input_this = path_input[run_input.index("projection")]+run_input[run_input.index("projection")]+'/'+freq_input+'/'+t[0:4]+'/'+t[4:6]+'/'

                # Open dataset in grid T, if grid T variables were selected
                if 'file_input_this_T' in locals():
                    print('    '+path_input_this+file_input_this_T)
                    # Open datasets
                    ds_this = xr.open_mfdataset(path_input_this+file_input_this_T, parallel = True)
                    # Drop one dimension from spatial coordinates
                    #ds_this['nav_lat'] = ds_this['nav_lat'].sel(x=1, drop=True)
                    #ds_this['nav_lon'] = ds_this['nav_lon'].sel(y=1, drop=True)
                    #gverri: replace two line above with 3 below cause of NEMO conversion intoo Erddap format
                    ds_this = ds_this.rename({'lat': 'nav_lat'})
                    ds_this = ds_this.rename({'lon': 'nav_lon'})
                    ds_this = ds_this.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})

                    # Set index for spatial coordinates
                    ds_this = ds_this.set_xindex("nav_lat")
                    ds_this = ds_this.set_xindex("nav_lon")
                    # Slice dataset according to area_input
                    ds_this = ds_this.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                    ds_this = ds_this[var_list_T]
                    # Concatenate data to create a single dataset
                    ds_T_projection = xr.concat([ds_T_projection, ds_this], dim = "time_counter")
                    del ds_this

                # Open dataset in grid U, if grid U variables were selected
                if 'file_input_this_U' in locals():
                    print('    '+path_input_this+file_input_this_U)
                    # Open datasets
                    ds_this = xr.open_mfdataset(path_input_this+file_input_this_U, parallel = True)
                    # Drop one dimension from spatial coordinates
                    #ds_this['nav_lat'] = ds_this['nav_lat'].sel(x=1, drop=True)
                    #ds_this['nav_lon'] = ds_this['nav_lon'].sel(y=1, drop=True)
                    #gverri: replace two line above with 3 below cause of NEMO conversion intoo Erddap format
                    ds_this = ds_this.rename({'lat': 'nav_lat'})
                    ds_this = ds_this.rename({'lon': 'nav_lon'})
                    ds_this = ds_this.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})

                    # Set index for spatial coordinates
                    ds_this = ds_this.set_xindex("nav_lat")
                    ds_this = ds_this.set_xindex("nav_lon")
                    # Slice dataset according to area_input
                    ds_this = ds_this.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                    ds_this = ds_this[var_list_U]
                    # Concatenate data to create a single dataset
                    ds_U_projection = xr.concat([ds_U_projection, ds_this], dim = "time_counter")
                    del ds_this

                # Open dataset in grid V, if grid V variables were selected
                if 'file_input_this_V' in locals():
                    print('    '+path_input_this+file_input_this_V)
                    # Open datasets
                    ds_this = xr.open_mfdataset(path_input_this+file_input_this_V, parallel = True)
                    # Drop one dimension from spatial coordinates
                    #ds_this['nav_lat'] = ds_this['nav_lat'].sel(x=1, drop=True)
                    #ds_this['nav_lon'] = ds_this['nav_lon'].sel(y=1, drop=True)
                    #gverri: replace two line above with 3 below cause of NEMO conversion intoo Erddap format
                    ds_this = ds_this.rename({'lat': 'nav_lat'})
                    ds_this = ds_this.rename({'lon': 'nav_lon'})
                    ds_this = ds_this.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})

                    # Set index for spatial coordinates
                    ds_this = ds_this.set_xindex("nav_lat")
                    ds_this = ds_this.set_xindex("nav_lon")
                    # Slice dataset according to area_input
                    ds_this = ds_this.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                    ds_this = ds_this[var_list_V]
                    # Concatenate data to create a single dataset
                    ds_V_projection = xr.concat([ds_V_projection, ds_this], dim = "time_counter")
                    del ds_this

                # Open dataset in grid W, if grid W variables were selected
                if 'file_input_this_W' in locals():
                    print('    '+path_input_this+file_input_this_W)
                    # Open datasets
                    ds_this = xr.open_mfdataset(path_input_this+file_input_this_W, parallel = True)
                    # Drop one dimension from spatial coordinates
                    #ds_this['nav_lat'] = ds_this['nav_lat'].sel(x=1, drop=True)
                    #ds_this['nav_lon'] = ds_this['nav_lon'].sel(y=1, drop=True)
                    #gverri: replace two line above with 3 below cause of NEMO conversion intoo Erddap format
                    ds_this = ds_this.rename({'lat': 'nav_lat'})
                    ds_this = ds_this.rename({'lon': 'nav_lon'})
                    ds_this = ds_this.swap_dims({'nav_lat': 'y', 'nav_lon': 'x'})

                    # Set index for spatial coordinates
                    ds_this = ds_this.set_xindex("nav_lat")
                    ds_this = ds_this.set_xindex("nav_lon")
                    # Slice dataset according to area_input
                    ds_this = ds_this.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
                    ds_this = ds_this[var_list_W]
                    # Concatenate data to create a single dataset
                    ds_W_projection = xr.concat([ds_W_projection, ds_this], dim = "time_counter")
                    del ds_this

                del path_input_this

            # Load mask file to apply to DataArrays
            ds_mask = xr.open_dataset(f'{path_mask[run_input.index("historical")]}{file_mask[run_input.index("historical")]}')
            # Open T mask if T variables were selected
            if 'file_input_this_T' in locals():
                mask = ds_mask['tmask'].isel(z = slice(0,1)).squeeze()
                bp =np.squeeze(np.nansum(np.squeeze(ds_mask['e3t_0'][0,:,:,:]*ds_mask['tmask']),axis=0))
                for var in var_list_T:
                    ds_T_projection[var] = ds_T_projection[var].where(mask==1)
                del mask
            # Open U mask if U variables were selected
            if 'file_input_this_U' in locals():
                mask = ds_mask['umask'].isel(z = slice(0,1)).squeeze()
                for var in var_list_U:
                    ds_U_projection[var] = ds_U_projection[var].where(mask==1)
                del mask
            # Open V mask if V variables were selected
            if 'file_input_this_V' in locals():
                mask = ds_mask['vmask'].isel(z = slice(0,1)).squeeze()
                for var in var_list_V:
                    ds_V_projection[var] = ds_V_projection[var].where(mask==1)
                del mask
            # Open W mask if W variables were selected
            if 'file_input_this_W' in locals():
                mask = ds_mask['fmask'].isel(z = slice(0,1)).squeeze()
                for var in var_list_W:
                    ds_W_projection[var] = ds_W_projection[var].where(mask==1)
                del mask

            del ds_mask

            # Compute calculated variables and delete unused DataArrays
            for var in var_input_new:
                if var == "SurfVel":
                    # Calculate Surface Velocity and add it to respective Dataset
                    ds_U_projection_interp, ds_V_projection_interp = xr.align(ds_U_projection, ds_V_projection, join="override")
                    ds_U_projection['SurfVel'] = np.sqrt(ds_U_projection_interp['vozocrtx'].isel(depth=0)**2 + ds_V_projection_interp['vomecrty'].isel(depth=0)**2)
                    # Add Surface Velocity to var_list
                    # If U is not called, remove it from the Dataset and var_list
                    if 'U' not in var_input:
                        ds_U_projection = ds_U_projection.drop(var_list[var_load.index('U')])
                    # If V is not called, remove it from the Dataset and var_list
                    if 'V' not in var_input:
                        ds_V_projection = ds_V_projection.drop(var_list[var_load.index('V')])
                if var == "T_VertInt":
                    # Calculate Vertically-Integrated Temperature and add it to respective Dataset
                    ds_T_projection['T_VertInt'] = ds_T_projection['votemper'].integrate("depth")
                    bpf=[]
                    for i in range(0,len(ds_T_projection['T_VertInt'][:,0,0])):
                        bpf.append(bp)
                    ds_T_projection['T_VertInt'] = ds_T_projection['T_VertInt']/bpf
                    # Add Vertically-Integrated Temperature to var_list
                    #var_list.insert(var_input.index('T_VertInt'),'T_VertInt')
                    # If T is not called, remove it from the Dataset and var_list
                    if 'T' not in var_input:
                        ds_T_projection = ds_T_projection.drop(var_list[var_load.index('T')])

            # If any of the grids was not used, create an empty dataset for that grid
            if 'ds_T_projection' not in locals():
                ds_T_projection = xr.Dataset()
            if 'ds_U_projection' not in locals():
                ds_U_projection = xr.Dataset()
            if 'ds_V_projection' not in locals():
                ds_V_projection = xr.Dataset()
            if 'ds_W_projection' not in locals():
                ds_W_projection = xr.Dataset()

            # Load datasets
            if 'ds_T_projection' in locals():
                ds_T_projection = ds_T_projection.load()
            if 'ds_U_projection' in locals():
                ds_U_projection = ds_U_projection.load()
            if 'ds_V_projection' in locals():
                ds_V_projection = ds_V_projection.load()
            if 'ds_W_projection' in locals():
                ds_W_projection = ds_W_projection.load()

        else:
            ds_T_projection = xr.Dataset()
            ds_U_projection = xr.Dataset()
            ds_V_projection = xr.Dataset()
            ds_W_projection = xr.Dataset()

    else:
        ds_T_projection = xr.Dataset()
        ds_U_projection = xr.Dataset()
        ds_V_projection = xr.Dataset()
        ds_W_projection = xr.Dataset()

        # Attribute the NEMO var names to the plotted variables
        var_list2 = create_var_list2(var_input)

        # Associate grid to each input variable
        grid_input = create_grid_input(var_input)

        # Define var long names for each input variable
        var_long = create_var_long(var_input)

        # Define colorbar titles for each input variable
        cbar_title = create_cbar_title(var_input)

        # Define colormaps for each input variable
        cmap_list = create_cmap_list(var_input)

    return(ds_T_projection, ds_U_projection, ds_V_projection, ds_W_projection, var_list2, var_long, grid_input, cbar_title, cmap_list, path_output_this)

# Function to load NetCDF files
def load_data_reanalysis(path_input, file_input, path_mask, file_mask, run_input, freq_input, timei, timef, area_input, var_input, path_output, subpath_output, path_output_this, path_file):
    # If run refers to CMEMS, load data. Otherwise, create empty dataset
    if 'reanalysis' in run_input:
        ## Create list of timesteps to be loaded
        timei_this = timei[run_input.index("reanalysis")]
        timef_this = timef[run_input.index("reanalysis")]
        time_list = create_time_list(timei_this, timef_this)

        # Define current time used to define the name of the output directory
        current_time = define_current_time()

        # Create new output folder for current request (only if not running a comparison with NEMO)
        if "historical" not in run_input and "projection" not in run_input:
            if subpath_output == "":
                os.makedirs(os.path.join(path_output, f'{current_time}/'))
                path_output_this = os.path.join(path_output, f'{current_time}/')
                # Create subfolders in output folder
                os.makedirs(os.path.join(path_output_this, 'maps/'))
                os.makedirs(os.path.join(path_output_this, 'timeseries/'))
            else:
                if os.path.exists(os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')):
                    print(f"Directory already exists: {os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')}")
                    writeoption = input("Append figures to existing directory (a), overwrite directory (o) or quit (q)? ")
                    if writeoption == 'A' or writeoption == 'a':
                        path_output_this = os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')
                    elif writeoption == 'O' or writeoption == 'o':    
                        shutil.rmtree(os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/'))
                        os.makedirs(os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/'))
                        path_output_this = os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')
                        # Create subfolders in output folder
                        os.makedirs(os.path.join(path_output_this, 'maps/'))
                        os.makedirs(os.path.join(path_output_this, 'timeseries/'))
                    elif writeoption == 'Q' or writeoption == 'q':
                        raise ValueError('Change the output subdirectory path before running the script again.')
                    else:
                        raise ValueError('Invalid input.')
                else:
                    os.makedirs(os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/'))
                    path_output_this = os.path.join(path_output, f'{current_time[0:9]}{subpath_output}/')
                    # Create subfolders in output folder
                    os.makedirs(os.path.join(path_output_this, 'maps/'))
                    os.makedirs(os.path.join(path_output_this, 'timeseries/'))

        # Attribute the NEMO var names to the plotted variables
        var_list2_cmems = create_var_list2_cmems(var_input)

        # Associate grid to each input variable
        grid_input_cmems = create_grid_input_cmems(var_input)

        # Define var long names for each input variable
        var_long = create_var_long(var_input)

        # Define colorbar titles for each input variable
        cbar_title = create_cbar_title(var_input)

        # Define colormaps for each input variable
        cmap_list = create_cmap_list(var_input)

        # Create new var input variable
        var_input_new = var_input.copy()
        # Get file path
        path_file_this = (f'{path_file}reanalysis_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        # Check if intermediate files exist and load files
        if os.path.exists(path_file_this):
            ds_cmems_this = nc.Dataset(f'{path_file}reanalysis_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            for i in range(0,len(var_input)):
                # Remove variable from var_input_new if it already exists in intermediate file
                if f'{var_input[i]}_dailytimeseries' in ds_cmems_this.variables.keys():
                    var_input_new.remove(var_input[i])

        if var_input_new != []:

            # Define original and calculated variables
            var_original = ['T', 'S', 'zos','U', 'V']
            var_calculated = {
                "SurfVel": ["U","V"],
                "SST": ["T"],
                "SSS": ["S"],
                "T_VertInt": ["T"],
                "SSH": ["zos"],
                }

            # Create list of variables to be loaded
            var_load = []
            for var in var_input_new:
                if var in var_original:
                    var_load+=[var]
                else:
                    var_load+=var_calculated[var]

            # Attribute the NEMO var names to the loaded variables
            var_list_cmems = create_var_list_cmems(var_load)

            # Associate grid to each loaded variable
            grid_list_cmems = create_grid_list_cmems(var_load)

            # Create filename for the first file based on the input parameters
            # Define frequency
            if freq_input == 'day':
                freq = 'd'
            elif freq_input == '3h':
                freq = 'h'
            # Define grids (T, U, V or W)
            var_aux_TEMP = ['T']
            var_aux_PSAL = ['S']
            var_aux_ASLV = ['zos']
            var_aux_RFVL = ['U', 'V']

            for var in var_aux_TEMP:
                if var in var_load:
                    grid = 'TEMP'
                    file_input_this_TEMP = eval(file_input[run_input.index("reanalysis")])
                    file_input_this_TEMP = file_input_this_TEMP.replace('YYYYMMDD','*')
                    break
            for var in var_aux_PSAL:
                if var in var_load:
                    grid = 'PSAL'
                    file_input_this_PSAL = eval(file_input[run_input.index("reanalysis")])
                    file_input_this_PSAL = file_input_this_PSAL.replace('YYYYMMDD','*')
                    break
            for var in var_aux_ASLV:
                if var in var_load:
                    grid = 'ASLV'
                    file_input_this_ASLV = eval(file_input[run_input.index("reanalysis")])
                    file_input_this_ASLV = file_input_this_ASLV.replace('YYYYMMDD','*')
                    break
            for var in var_aux_RFVL:
                if var in var_load:
                    grid = 'RFVL'
                    file_input_this_RFVL = eval(file_input[run_input.index("reanalysis")])
                    file_input_this_RFVL = file_input_this_RFVL.replace('YYYYMMDD','*')
                    break
            del var_aux_TEMP, var_aux_PSAL, var_aux_ASLV, var_aux_RFVL, grid

            # Create directory path for the first month of data
            # Change path if current month is during spinup
            path_input_this = path_input[run_input.index("reanalysis")]+run_input[run_input.index("reanalysis")]+'/'+freq_input+'/'+timei_this[0:4]+'/'+timei_this[4:6]+'/'

            # Create dataset with data from the first month
            # Grid TEMP, if grid TEMP variables were selected
            if 'file_input_this_TEMP' in locals():
                print('    '+path_input_this+file_input_this_TEMP)
                ds_TEMP = xr.open_mfdataset(path_input_this+file_input_this_TEMP, parallel = True).sel(lat = slice(area_input[0], area_input[1]), lon = slice(area_input[2], area_input[3]))
                ind_TEMP = list(np.where(np.array(grid_list_cmems)=='TEMP')[0])
                var_list_TEMP = []
                for ind in ind_TEMP:
                    var_list_TEMP.append(var_list_cmems[ind])
                ds_TEMP = ds_TEMP[var_list_TEMP]
            # Grid PSAL, if grid PSAL variables were selected
            if 'file_input_this_PSAL' in locals():
                print('    '+path_input_this+file_input_this_PSAL)
                ds_PSAL = xr.open_mfdataset(path_input_this+file_input_this_PSAL, parallel = True).sel(lat = slice(area_input[0], area_input[1]), lon = slice(area_input[2], area_input[3]))
                ind_PSAL = list(np.where(np.array(grid_list_cmems)=='PSAL')[0])
                var_list_PSAL = []
                for ind in ind_PSAL:
                    var_list_PSAL.append(var_list_cmems[ind])
                ds_PSAL = ds_PSAL[var_list_PSAL]
            # Grid ASLV, if grid ASLV variables were selected
            if 'file_input_this_ASLV' in locals():
                print('    '+path_input_this+file_input_this_ASLV)
                ds_ASLV = xr.open_mfdataset(path_input_this+file_input_this_ASLV, parallel = True).sel(lat = slice(area_input[0], area_input[1]), lon = slice(area_input[2], area_input[3]))
                ind_ASLV = list(np.where(np.array(grid_list_cmems)=='ASLV')[0])
                var_list_ASLV = []
                for ind in ind_ASLV:
                    var_list_ASLV.append(var_list_cmems[ind])
                ds_ASLV = ds_ASLV[var_list_ASLV]
            # Grid RFVL, if grid RFVL variables were selected
            if 'file_input_this_RFVL' in locals():
                print('    '+path_input_this+file_input_this_RFVL)
                ds_RFVL = xr.open_mfdataset(path_input_this+file_input_this_RFVL, parallel = True).sel(lat = slice(area_input[0], area_input[1]), lon = slice(area_input[2], area_input[3]))
                ind_RFVL = list(np.where(np.array(grid_list_cmems)=='RFVL')[0])
                var_list_RFVL = []
                for ind in ind_RFVL:
                    var_list_RFVL.append(var_list_cmems[ind])
                ds_RFVL = ds_RFVL[var_list_RFVL]

            del path_input_this

            # Loop over other months to update the directory path
            for t in time_list[1:]:

                path_input_this = path_input[run_input.index("reanalysis")]+run_input[run_input.index("reanalysis")]+'/'+freq_input+'/'+t[0:4]+'/'+t[4:6]+'/'

                # Open dataset in grid TEMP, if grid TEMP variables were selected
                if 'file_input_this_TEMP' in locals():
                    print('    '+path_input_this+file_input_this_TEMP)
                    ds_this = xr.open_mfdataset(path_input_this+file_input_this_TEMP, parallel = True).sel(lat = slice(area_input[0], area_input[1]), lon = slice(area_input[2], area_input[3]))
                    ds_this = ds_this[var_list_TEMP]
                    # Concatenate data to create a single dataset
                    ds_TEMP = xr.concat([ds_TEMP, ds_this], dim = "time")
                    del ds_this

                # Open dataset in grid PSAL, if grid PSAL variables were selected
                if 'file_input_this_PSAL' in locals():
                    print('    '+path_input_this+file_input_this_PSAL)
                    ds_this = xr.open_mfdataset(path_input_this+file_input_this_PSAL, parallel = True).sel(lat = slice(area_input[0], area_input[1]), lon = slice(area_input[2], area_input[3]))
                    ds_this = ds_this[var_list_PSAL]
                    # Concatenate data to create a single dataset
                    ds_PSAL = xr.concat([ds_PSAL, ds_this], dim = "time")
                    del ds_this

                # Open dataset in grid ASLV, if grid ASLV variables were selected
                if 'file_input_this_ASLV' in locals():
                    print('    '+path_input_this+file_input_this_ASLV)
                    ds_this = xr.open_mfdataset(path_input_this+file_input_this_ASLV, parallel = True).sel(lat = slice(area_input[0], area_input[1]), lon = slice(area_input[2], area_input[3]))
                    ds_this = ds_this[var_list_ASLV]
                    # Concatenate data to create a single dataset
                    ds_ASLV = xr.concat([ds_ASLV, ds_this], dim = "time")
                    del ds_this

                # Open dataset in grid RFVL, if grid RFVL variables were selected
                if 'file_input_this_RFVL' in locals():
                    print('    '+path_input_this+file_input_this_RFVL)
                    ds_this = xr.open_mfdataset(path_input_this+file_input_this_RFVL, parallel = True).sel(lat = slice(area_input[0], area_input[1]), lon = slice(area_input[2], area_input[3]))
                    ds_this = ds_this[var_list_RFVL]
                    # Concatenate data to create a single dataset
                    ds_RFVL = xr.concat([ds_RFVL, ds_this], dim = "time")
                    del ds_this

                del path_input_this

            # Load datasets
            #if 'ds_TEMP' in locals():
            #    ds_TEMP = ds_TEMP.load()
            #if 'ds_PSAL' in locals():
            #    ds_PSAL = ds_PSAL.load()
            #if 'ds_ASLV' in locals():
            #    ds_ASLV = ds_ASLV.load()
            #if 'ds_RFVL' in locals():
            #    ds_RFVL = ds_RFVL.load()

            # Load mask file to apply to DataArrays
            if path_mask != '':
                #ds_mask = xr.open_dataset(f'{path_mask[run_input.index("reanalysis")]}{file_mask[run_input.index("reanalysis")]}').sel(lat = slice(area_input[0], area_input[1]), lon = slice(area_input[2], area_input[3]))
                ds_mask = xr.open_dataset(f'{path_mask[run_input.index("reanalysis")]}{file_mask[run_input.index("reanalysis")]}')
                # Drop time dimension
                #ds_mask = ds_mask.drop_dims('t')
                # Rename lat and lon to avoid conflict later
                ds_mask = ds_mask.rename({'z': 'depth'})
                #ds_mask = ds_mask.rename({'t': 'time'})
                ds_mask = ds_mask.rename({'nav_lat': 'lat'})
                ds_mask = ds_mask.rename({'nav_lon': 'lon'})
                # Drop useless extra dimension
                ds_mask['lat'] = ds_mask['lat'].sel(x=0)
                ds_mask['lon'] = ds_mask['lon'].sel(y=0)
                # Swap dimensions
                ds_mask = ds_mask.swap_dims({'y': 'lat', 'x': 'lon'})
                # Apply area_input
                ds_mask = ds_mask.sel(lat = slice(area_input[0], area_input[1]), lon = slice(area_input[2], area_input[3]))
                mask = ds_mask['tmask']
                # Apply filter for Adriatic if applicable
                if area_input[0]==39 and area_input[1]==46:
                    mask = xr.where(((mask['lat']<41) & (mask['lon']<16.3)),np.nan,mask)
                    mask = xr.where(((mask['lat']<42) & (mask['lon']<14)),np.nan,mask)
                br =np.squeeze(np.nansum(np.squeeze(ds_mask['e3t_0'][0,:,:,:]*mask),axis=0))
                # Open mask if TEMP variables were selected
                if 'file_input_this_TEMP' in locals():
                    for var in var_list_TEMP:
                        ds_TEMP[var] = ds_TEMP[var].where(mask==1)
                # Open mask if PSAL variables were selected
                if 'file_input_this_PSAL' in locals():
                    for var in var_list_PSAL:
                        ds_PSAL[var] = ds_PSAL[var].where(mask==1)
                # Open mask if ASLV variables were selected
                if 'file_input_this_ASLV' in locals():
                    for var in var_list_ASLV:
                        ds_ASLV[var] = ds_ASLV[var].where(mask==1)
                # Open mask if RFVL variables were selected
                if 'file_input_this_RFVL' in locals():
                    for var in var_list_RFVL:
                        ds_RFVL[var] = ds_RFVL[var].where(mask==1)

                del ds_mask

            # Compute calculated variables and delete unused DataArrays
            for var in var_input_new:
                if var == "SST":
                    # Calculate Sea Surface Temperature and add it to respective Dataset
                    ds_TEMP['SST'] = ds_TEMP['thetao'].isel(depth=0)
                    # If T is not called, remove it from the Dataset and var_list
                    if 'T' not in var_input and 'T_VertInt' not in var_input:
                        ds_TEMP = ds_TEMP.drop(var_list_cmems[var_load.index('T')])

                if var == "SSS":
                    # Calculate Sea Surface Temperature and add it to respective Dataset
                    ds_PSAL['SSS'] = ds_PSAL['so'].isel(depth=0)
                    # If S is not called, remove it from the Dataset and var_list
                    if 'S' not in var_input:
                        ds_PSAL = ds_PSAL.drop(var_list_cmems[var_load.index('S')])

                if var == "SurfVel":
                    # Calculate Surface Velocity and add it to respective Dataset
                    ds_RFVL['SurfVel'] = np.sqrt(ds_RFVL['uo'].isel(depth=0)**2 + ds_RFVL['vo'].isel(depth=0)**2)
                    # If U is not called, remove it from the Dataset and var_list
                    if 'U' not in var_input:
                        ds_RFVL = ds_RFVL.drop(var_list_cmems[var_load.index('U')])
                    # If V is not called, remove it from the Dataset and var_list
                    if 'V' not in var_input:
                        ds_RFVL = ds_RFVL.drop(var_list_cmems[var_load.index('V')])

                if var == "T_VertInt":
                    # Calculate Vertically-Integrated Temperature and add it to respective Dataset
                    ds_TEMP['T_VertInt'] = np.squeeze(ds_TEMP['thetao']).fillna(0).integrate("depth")
                    brf=[]
                    for i in range(0,len(ds_TEMP['T_VertInt'][:,0,0])):
                        brf.append(br)
                    ds_TEMP['T_VertInt'] = ds_TEMP['T_VertInt']/brf

                    # Mask T_VertInt again due to use of fillna(0) above
                    #ds_TEMP['T_VertInt'] = ds_TEMP['T_VertInt'].where(mask==1)
                    # If T is not called, remove it from the Dataset and var_list
                    if 'T' not in var_input:
                        ds_TEMP = ds_TEMP.drop(var_list_cmems[var_load.index('T')])

                if var == "SSH":
                    # Calculate Sea Surface Height and add it to respective Dataset
                    ds_ASLV['SSH'] = ds_ASLV['zos'].isel(depth=0)
                    # If zos is not called, remove it from the Dataset and var_list
                    if 'zos' not in var_input:
                        ds_ASLV = ds_ASLV.drop(var_list_cmems[var_load.index('zos')])

            # Change coordinate names to match NEMO's or create empty dataset (TEMP)
            if 'ds_TEMP' in locals():
                ds_TEMP = ds_TEMP.rename({'time': 'time_counter'})
                ds_TEMP = ds_TEMP.rename({'lat': 'nav_lat'})
                ds_TEMP = ds_TEMP.rename({'lon': 'nav_lon'})
            else:
                ds_TEMP = xr.Dataset()

            # Change coordinate names to match NEMO's or create empty dataset (PSAL)
            if 'ds_PSAL' in locals():
                ds_PSAL = ds_PSAL.rename({'time': 'time_counter'})
                ds_PSAL = ds_PSAL.rename({'lat': 'nav_lat'})
                ds_PSAL = ds_PSAL.rename({'lon': 'nav_lon'})
            else:
                ds_PSAL = xr.Dataset()

            # Change coordinate names to match NEMO's or create empty dataset (ASLV)
            if 'ds_ASLV' in locals():
                ds_ASLV = ds_ASLV.rename({'time': 'time_counter'})
                ds_ASLV = ds_ASLV.rename({'lat': 'nav_lat'})
                ds_ASLV = ds_ASLV.rename({'lon': 'nav_lon'})  
            else:
                ds_ASLV = xr.Dataset()

            # Change coordinate names to match NEMO's or create empty dataset (RFVL)
            if 'ds_RFVL' in locals():
                ds_RFVL = ds_RFVL.rename({'time': 'time_counter'})
                ds_RFVL = ds_RFVL.rename({'lat': 'nav_lat'})
                ds_RFVL = ds_RFVL.rename({'lon': 'nav_lon'})  
            else:
                ds_RFVL = xr.Dataset()

            # Merge datasets
            ds_cmems = xr.merge([ds_TEMP, ds_PSAL, ds_ASLV, ds_RFVL])
            ds_cmems = ds_cmems.isel(t=0)

            ds_cmems = ds_cmems.load()

        else:
            ds_cmems = xr.Dataset()

    else:
        ds_cmems = xr.Dataset()

        # Attribute the CMEMS var names to the plotted variables
        var_list2_cmems = create_var_list2_cmems(var_input)

        # Define var long names for each input variable
        var_long = create_var_long(var_input)

        # Define colorbar titles for each input variable
        cbar_title = create_cbar_title(var_input)

        # Define colormaps for each input variable
        cmap_list = create_cmap_list(var_input)

    return(ds_cmems, var_list2_cmems, var_long, cbar_title, cmap_list, path_output_this)

# Function to save intermediate files
def save_intermediatefiles(ds_T_historical, ds_U_historical, ds_V_historical, ds_W_historical, ds_T_projection, ds_U_projection, ds_V_projection, ds_W_projection, ds_cmems, path_mask, file_mask, var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, path_file):
    # Load mask file(s)
    if 'historical' in run_input:
        # Get dx and dy
        ds_mask = xr.open_dataset(f'{path_mask[run_input.index("historical")]}{file_mask[run_input.index("historical")]}')
        # Drop useless extra dimension
        #ds_mask['nav_lat'] = ds_mask['nav_lat'].sel(x=0)
        #ds_mask['nav_lon'] = ds_mask['nav_lon'].sel(y=0)
        # Swap dimensions
        #ds_mask = ds_mask.swap_dims({'y': 'nav_lat', 'x': 'nav_lon'})
        # tmask
        mask = ds_mask['tmask'].isel(z = slice(0,1)).squeeze()
        da_area_T = ds_mask['e1t'].squeeze() * ds_mask['e2t'].squeeze().where(mask==1)
        # umask
        mask = ds_mask['umask'].isel(z = slice(0,1)).squeeze()
        da_area_U = ds_mask['e1u'].squeeze() * ds_mask['e2u'].squeeze().where(mask==1)
        # vmask
        mask = ds_mask['vmask'].isel(z = slice(0,1)).squeeze()
        da_area_V = ds_mask['e1v'].squeeze() * ds_mask['e2v'].squeeze().where(mask==1)
        #fmask
        mask = ds_mask['fmask'].isel(z = slice(0,1)).squeeze()
        da_area_W = ds_mask['e1f'].squeeze() * ds_mask['e2f'].squeeze().where(mask==1)
        del mask, ds_mask

    if 'projection' in run_input:
        # Get dx and dy
        ds_mask = xr.open_dataset(f'{path_mask[run_input.index("projection")]}{file_mask[run_input.index("projection")]}')
        # Drop useless extra dimension
        #ds_mask['nav_lat'] = ds_mask['nav_lat'].sel(x=0)
        #ds_mask['nav_lon'] = ds_mask['nav_lon'].sel(y=0)
        # Swap dimensions
        #ds_mask = ds_mask.swap_dims({'y': 'nav_lat', 'x': 'nav_lon'})
        # tmask
        mask = ds_mask['tmask'].isel(z = slice(0,1)).squeeze()
        da_area_T = ds_mask['e1t'].squeeze() * ds_mask['e2t'].squeeze().where(mask==1)
        # umask
        mask = ds_mask['umask'].isel(z = slice(0,1)).squeeze()
        da_area_U = ds_mask['e1u'].squeeze() * ds_mask['e2u'].squeeze().where(mask==1)
        # vmask
        mask = ds_mask['vmask'].isel(z = slice(0,1)).squeeze()
        da_area_V = ds_mask['e1v'].squeeze() * ds_mask['e2v'].squeeze().where(mask==1)
        #fmask
        mask = ds_mask['fmask'].isel(z = slice(0,1)).squeeze()
        da_area_W = ds_mask['e1f'].squeeze() * ds_mask['e2f'].squeeze().where(mask==1)
        del mask, ds_mask

    if 'reanalysis' in run_input:
        ds_mask = xr.open_dataset(f'{path_mask[run_input.index("reanalysis")]}{file_mask[run_input.index("reanalysis")]}')
        # Rename lat and lon to avoid conflict later
        ds_mask = ds_mask.rename({'z': 'depth'})
        # Drop useless extra dimension
        ds_mask['nav_lat'] = ds_mask['nav_lat'].sel(x=0)
        ds_mask['nav_lon'] = ds_mask['nav_lon'].sel(y=0)
        # Swap dimensions
        ds_mask = ds_mask.swap_dims({'y': 'nav_lat', 'x': 'nav_lon'})
        # Apply area_input
        ds_mask = ds_mask.sel(nav_lat = slice(area_input[0], area_input[1]), nav_lon = slice(area_input[2], area_input[3]))
        # Get mask
        mask = ds_mask['tmask'].isel(depth = slice(0,1)).squeeze()
        # Apply filter for Adriatic if applicable
        if area_input[0]==39 and area_input[1]==46:
            mask = xr.where(((mask['nav_lat']<41) & (mask['nav_lon']<16.3)),0,mask)
            mask = xr.where(((mask['nav_lat']<42) & (mask['nav_lon']<14)),0,mask)
        # Calculate area
        da_area_cmems = ds_mask['e1t'].squeeze() * ds_mask['e2t'].squeeze().where(mask==1)
        del mask, ds_mask

    ## Save intermediate files
    if 'historical' in run_input:

        # Identify time
        timei_this = timei[run_input.index("historical")]
        timef_this = timef[run_input.index("historical")]
        # Create new var input variable
        var_input_new = var_input.copy()
        var_list2_new = var_list2.copy()
        grid_input_new = grid_input.copy()
        # Get file path
        path_file_this = (f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        # Check if intermediate files exist and load files
        if os.path.exists(path_file_this):
            mode_historical = "a"
            ds_T_this_aux = xr.open_dataset(f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            ds_U_this_aux = xr.open_dataset(f'{path_file}historical_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            ds_V_this_aux = xr.open_dataset(f'{path_file}historical_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            ds_W_this_aux = xr.open_dataset(f'{path_file}historical_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            for i in range(0,len(var_input)):
                # Remove variable from var_input_new if it already exists in intermediate file
                if grid_input[i] == 'T':
                    if f'{var_input[i]}_dailytimeseries' in ds_T_this_aux.variables.keys():
                        grid_input_new.pop(var_input_new.index(var_input[i]))
                        var_input_new.remove(var_input[i])
                        var_list2_new.remove(var_list2[i])
                if grid_input[i] == 'U':
                    if f'{var_input[i]}_dailytimeseries' in ds_U_this_aux.variables.keys():
                        grid_input_new.pop(var_input_new.index(var_input[i]))
                        var_input_new.remove(var_input[i])
                        var_list2_new.remove(var_list2[i])
                if grid_input[i] == 'V':
                    if f'{var_input[i]}_dailytimeseries' in ds_V_this_aux.variables.keys():
                        grid_input_new.pop(var_input_new.index(var_input[i]))
                        var_input_new.remove(var_input[i])
                        var_list2_new.remove(var_list2[i])
                if grid_input[i] == 'W':
                    if f'{var_input[i]}_dailytimeseries' in ds_W_this_aux.variables.keys():
                        grid_input_new.pop(var_input_new.index(var_input[i]))
                        var_input_new.remove(var_input[i])
                        var_list2_new.remove(var_list2[i])
            del ds_T_this_aux, ds_U_this_aux, ds_V_this_aux, ds_W_this_aux
        else:
            mode_historical = "w"

        # If there are variables to load, calculate derived fields and save file
        if var_input_new != []:    
            ds_T_this = xr.Dataset()
            ds_U_this = xr.Dataset()
            ds_V_this = xr.Dataset()
            ds_W_this = xr.Dataset()
            for i in range(0,len(var_input_new)):
                if grid_input_new[i] == 'T':
                    ds_T_this[f'{var_input_new[i]}_dailytimeseries'] = (ds_T_historical[var_list2_new[i]] * da_area_T).sum(dim=["x", "y"], skipna = True) / da_area_T.sum(dim=["x", "y"], skipna = True)                
                    da_aux = ds_T_historical[var_list2_new[i]].resample(time_counter='1M', label='left', loffset=timedelta(days=15)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_monthly'})
                    ds_T_this[f'{var_input_new[i]}_monthlytimeseries'] = da_aux
                    del da_aux
                    da_aux = ds_T_historical[var_list2_new[i]].resample(time_counter='AS', label='left', loffset=timedelta(days=182)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_annual'})
                    ds_T_this[f'{var_input_new[i]}_annualtimeseries'] = da_aux
                    del da_aux
                    ds_T_this[f'{var_input_new[i]}_meanmap'] = ds_T_historical[var_list2_new[i]].mean(dim='time_counter', skipna=True)
                    ds_T_this[f'{var_input_new[i]}_monthlymap'] = ds_T_historical[var_list2_new[i]].groupby("time_counter.month").mean('time_counter', skipna=True)
                    ds_T_this[f'{var_input_new[i]}_seasonalmap'] = ds_T_historical[var_list2_new[i]].groupby("time_counter.season").mean('time_counter', skipna=True)
                    ds_T_this[f'{var_input_new[i]}_dailymeanyear'] = ds_T_historical[var_list2_new[i]].groupby("time_counter.dayofyear").mean('time_counter').mean(dim=["x", "y"], skipna = True)
                    da_aux = ds_T_historical[var_list2_new[i]].groupby("time_counter.month").mean('time_counter').mean(dim=["x", "y"], skipna = True).rename({'month': 'monthofyear'})
                    ds_T_this[f'{var_input_new[i]}_monthlymeanyear'] = da_aux
                    del da_aux
                    ds_T_this = ds_T_this.swap_dims({"x": "nav_lon", "y": "nav_lat"})
                if grid_input_new[i] == 'U':
                    ds_U_this[f'{var_input_new[i]}_dailytimeseries'] = (ds_U_historical[var_list2_new[i]] * da_area_U).sum(dim=["x", "y"], skipna = True) / da_area_U.sum(dim=["x", "y"], skipna = True)                
                    da_aux = ds_U_historical[var_list2_new[i]].resample(time_counter='1M', label='left', loffset=timedelta(days=15)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_monthly'})
                    ds_U_this[f'{var_input_new[i]}_monthlytimeseries'] = da_aux
                    del da_aux
                    da_aux = ds_U_historical[var_list2_new[i]].resample(time_counter='AS', label='left', loffset=timedelta(days=182)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_annual'})
                    ds_U_this[f'{var_input_new[i]}_annualtimeseries'] = da_aux
                    del da_aux
                    ds_U_this[f'{var_input_new[i]}_meanmap'] = ds_U_historical[var_list2_new[i]].mean(dim='time_counter', skipna=True)
                    ds_U_this[f'{var_input_new[i]}_monthlymap'] = ds_U_historical[var_list2_new[i]].groupby("time_counter.month").mean('time_counter', skipna=True)
                    ds_U_this[f'{var_input_new[i]}_seasonalmap'] = ds_U_historical[var_list2_new[i]].groupby("time_counter.season").mean('time_counter', skipna=True)
                    ds_U_this[f'{var_input_new[i]}_dailymeanyear'] = ds_U_historical[var_list2_new[i]].groupby("time_counter.dayofyear").mean('time_counter').mean(dim=["x", "y"], skipna = True)
                    da_aux = ds_U_historical[var_list2_new[i]].groupby("time_counter.month").mean('time_counter').mean(dim=["x", "y"], skipna = True).rename({'month': 'monthofyear'})
                    ds_U_this[f'{var_input_new[i]}_monthlymeanyear'] = da_aux
                    del da_aux
                    ds_U_this = ds_U_this.swap_dims({"x": "nav_lon", "y": "nav_lat"})
                if grid_input_new[i] == 'V':
                    ds_V_this[f'{var_input_new[i]}_dailytimeseries'] = (ds_V_historical[var_list2_new[i]] * da_area_V).sum(dim=["x", "y"], skipna = True) / da_area_V.sum(dim=["x", "y"], skipna = True)                
                    da_aux = ds_V_historical[var_list2_new[i]].resample(time_counter='1M', label='left', loffset=timedelta(days=15)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_monthly'})
                    ds_V_this[f'{var_input_new[i]}_monthlytimeseries'] = da_aux
                    del da_aux
                    da_aux = ds_V_historical[var_list2_new[i]].resample(time_counter='AS', label='left', loffset=timedelta(days=182)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_annual'})
                    ds_V_this[f'{var_input_new[i]}_annualtimeseries'] = da_aux
                    del da_aux
                    ds_V_this[f'{var_input_new[i]}_meanmap'] = ds_V_historical[var_list2_new[i]].mean(dim='time_counter', skipna=True)
                    ds_V_this[f'{var_input_new[i]}_monthlymap'] = ds_V_historical[var_list2_new[i]].groupby("time_counter.month").mean('time_counter', skipna=True)
                    ds_V_this[f'{var_input_new[i]}_seasonalmap'] = ds_V_historical[var_list2_new[i]].groupby("time_counter.season").mean('time_counter', skipna=True)
                    ds_V_this[f'{var_input_new[i]}_dailymeanyear'] = ds_V_historical[var_list2_new[i]].groupby("time_counter.dayofyear").mean('time_counter').mean(dim=["x", "y"], skipna = True)
                    da_aux = ds_V_historical[var_list2_new[i]].groupby("time_counter.month").mean('time_counter').mean(dim=["x", "y"], skipna = True).rename({'month': 'monthofyear'})
                    ds_V_this[f'{var_input_new[i]}_monthlymeanyear'] = da_aux
                    del da_aux
                    ds_V_this = ds_V_this.swap_dims({"x": "nav_lon", "y": "nav_lat"})
                if grid_input_new[i] == 'W':
                    ds_W_this[f'{var_input_new[i]}_dailytimeseries'] = (ds_W_historical[var_list2_new[i]] * da_area_W).sum(dim=["x", "y"], skipna = True) / da_area_W.sum(dim=["x", "y"], skipna = True)                
                    da_aux = ds_W_historical[var_list2_new[i]].resample(time_counter='1M', label='left', loffset=timedelta(days=15)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_monthly'})
                    ds_W_this[f'{var_input_new[i]}_monthlytimeseries'] = da_aux
                    del da_aux
                    da_aux = ds_W_historical[var_list2_new[i]].resample(time_counter='AS', label='left', loffset=timedelta(days=182)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_annual'})
                    ds_W_this[f'{var_input_new[i]}_annualtimeseries'] = da_aux
                    del da_aux
                    ds_W_this[f'{var_input_new[i]}_meanmap'] = ds_W_historical[var_list2_new[i]].mean(dim='time_counter', skipna=True)
                    ds_W_this[f'{var_input_new[i]}_monthlymap'] = ds_W_historical[var_list2_new[i]].groupby("time_counter.month").mean('time_counter', skipna=True)
                    ds_W_this[f'{var_input_new[i]}_seasonalmap'] = ds_W_historical[var_list2_new[i]].groupby("time_counter.season").mean('time_counter', skipna=True)
                    ds_W_this[f'{var_input_new[i]}_dailymeanyear'] = ds_W_historical[var_list2_new[i]].groupby("time_counter.dayofyear").mean('time_counter').mean(dim=["x", "y"], skipna = True)
                    da_aux = ds_W_historical[var_list2_new[i]].groupby("time_counter.month").mean('time_counter').mean(dim=["x", "y"], skipna = True).rename({'month': 'monthofyear'})
                    ds_W_this[f'{var_input_new[i]}_monthlymeanyear'] = da_aux
                    del da_aux
                    ds_W_this = ds_W_this.swap_dims({"x": "nav_lon", "y": "nav_lat"})
            print('    Historical')
            ds_T_this.to_netcdf(f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc', mode = mode_historical)        
            ds_U_this.to_netcdf(f'{path_file}historical_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc', mode = mode_historical)        
            ds_V_this.to_netcdf(f'{path_file}historical_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc', mode = mode_historical)        
            ds_W_this.to_netcdf(f'{path_file}historical_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc', mode = mode_historical)        
            del ds_T_this, ds_U_this, ds_V_this, ds_W_this

    if 'projection' in run_input:

        # Identify time
        timei_this = timei[run_input.index("projection")]
        timef_this = timef[run_input.index("projection")]
        # Create new var input variable
        var_input_new = var_input.copy()
        var_list2_new = var_list2.copy()
        grid_input_new = grid_input.copy()
        # Get file path
        path_file_this = (f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        # Check if intermediate files exist and load files
        if os.path.exists(path_file_this):
            mode_projection = "a"
            ds_T_this_aux = xr.open_dataset(f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            ds_U_this_aux = xr.open_dataset(f'{path_file}projection_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            ds_V_this_aux = xr.open_dataset(f'{path_file}projection_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            ds_W_this_aux = xr.open_dataset(f'{path_file}projection_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            for i in range(0,len(var_input)):
                # Remove variable from var_input_new if it already exists in intermediate file
                if grid_input[i] == 'T':
                    if f'{var_input[i]}_dailytimeseries' in ds_T_this_aux.variables.keys():
                        grid_input_new.pop(var_input_new.index(var_input[i]))
                        var_input_new.remove(var_input[i])
                        var_list2_new.remove(var_list2[i])
                if grid_input[i] == 'U':
                    if f'{var_input[i]}_dailytimeseries' in ds_U_this_aux.variables.keys():
                        grid_input_new.pop(var_input_new.index(var_input[i]))
                        var_input_new.remove(var_input[i])
                        var_list2_new.remove(var_list2[i])
                if grid_input[i] == 'V':
                    if f'{var_input[i]}_dailytimeseries' in ds_V_this_aux.variables.keys():
                        grid_input_new.pop(var_input_new.index(var_input[i]))
                        var_input_new.remove(var_input[i])
                        var_list2_new.remove(var_list2[i])
                if grid_input[i] == 'W':
                    if f'{var_input[i]}_dailytimeseries' in ds_W_this_aux.variables.keys():
                        grid_input_new.pop(var_input_new.index(var_input[i]))
                        var_input_new.remove(var_input[i])
                        var_list2_new.remove(var_list2[i])
            del ds_T_this_aux, ds_U_this_aux, ds_V_this_aux, ds_W_this_aux
        else:
            mode_projection = "w"

        # If there are variables to load, calculate derived fields and save file
        if var_input_new != []:
            ds_T_this = xr.Dataset()
            ds_U_this = xr.Dataset()
            ds_V_this = xr.Dataset()
            ds_W_this = xr.Dataset()
            for i in range(0,len(var_input_new)):
                if grid_input_new[i] == 'T':
                    ds_T_this[f'{var_input_new[i]}_dailytimeseries'] = (ds_T_projection[var_list2_new[i]] * da_area_T).sum(dim=["x", "y"], skipna = True) / da_area_T.sum(dim=["x", "y"], skipna = True)                
                    da_aux = ds_T_projection[var_list2_new[i]].resample(time_counter='1M', label='left', loffset=timedelta(days=15)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_monthly'})
                    ds_T_this[f'{var_input_new[i]}_monthlytimeseries'] = da_aux
                    del da_aux
                    da_aux = ds_T_projection[var_list2_new[i]].resample(time_counter='AS', label='left', loffset=timedelta(days=182)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_annual'})
                    ds_T_this[f'{var_input_new[i]}_annualtimeseries'] = da_aux
                    del da_aux
                    ds_T_this[f'{var_input_new[i]}_meanmap'] = ds_T_projection[var_list2_new[i]].mean(dim='time_counter', skipna=True)
                    ds_T_this[f'{var_input_new[i]}_monthlymap'] = ds_T_projection[var_list2_new[i]].groupby("time_counter.month").mean('time_counter', skipna=True)
                    ds_T_this[f'{var_input_new[i]}_seasonalmap'] = ds_T_projection[var_list2_new[i]].groupby("time_counter.season").mean('time_counter', skipna=True)
                    ds_T_this[f'{var_input_new[i]}_dailymeanyear'] = ds_T_projection[var_list2_new[i]].groupby("time_counter.dayofyear").mean('time_counter').mean(dim=["x", "y"], skipna = True)
                    da_aux = ds_T_projection[var_list2_new[i]].groupby("time_counter.month").mean('time_counter').mean(dim=["x", "y"], skipna = True).rename({'month': 'monthofyear'})
                    ds_T_this[f'{var_input_new[i]}_monthlymeanyear'] = da_aux
                    del da_aux
                    ds_T_this = ds_T_this.swap_dims({"x": "nav_lon", "y": "nav_lat"})
                if grid_input_new[i] == 'U':
                    ds_U_this[f'{var_input_new[i]}_dailytimeseries'] = (ds_U_projection[var_list2_new[i]] * da_area_T).sum(dim=["x", "y"], skipna = True) / da_area_T.sum(dim=["x", "y"], skipna = True)                
                    da_aux = ds_U_projection[var_list2_new[i]].resample(time_counter='1M', label='left', loffset=timedelta(days=15)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_monthly'})
                    ds_U_this[f'{var_input_new[i]}_monthlytimeseries'] = da_aux
                    del da_aux
                    da_aux = ds_U_projection[var_list2_new[i]].resample(time_counter='AS', label='left', loffset=timedelta(days=182)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_annual'})
                    ds_U_this[f'{var_input_new[i]}_annualtimeseries'] = da_aux
                    del da_aux
                    ds_U_this[f'{var_input_new[i]}_meanmap'] = ds_U_projection[var_list2_new[i]].mean(dim='time_counter', skipna=True)
                    ds_U_this[f'{var_input_new[i]}_monthlymap'] = ds_U_projection[var_list2_new[i]].groupby("time_counter.month").mean('time_counter', skipna=True)
                    ds_U_this[f'{var_input_new[i]}_seasonalmap'] = ds_U_projection[var_list2_new[i]].groupby("time_counter.season").mean('time_counter', skipna=True)
                    ds_U_this[f'{var_input_new[i]}_dailymeanyear'] = ds_U_projection[var_list2_new[i]].groupby("time_counter.dayofyear").mean('time_counter').mean(dim=["x", "y"], skipna = True)
                    da_aux = ds_U_projection[var_list2_new[i]].groupby("time_counter.month").mean('time_counter').mean(dim=["x", "y"], skipna = True).rename({'month': 'monthofyear'})
                    ds_U_this[f'{var_input_new[i]}_monthlymeanyear'] = da_aux
                    del da_aux
                    ds_U_this = ds_U_this.swap_dims({"x": "nav_lon", "y": "nav_lat"})
                if grid_input_new[i] == 'V':
                    ds_V_this[f'{var_input_new[i]}_dailytimeseries'] = (ds_V_projection[var_list2_new[i]] * da_area_T).sum(dim=["x", "y"], skipna = True) / da_area_T.sum(dim=["x", "y"], skipna = True)                
                    da_aux = ds_V_projection[var_list2_new[i]].resample(time_counter='1M', label='left', loffset=timedelta(days=15)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_monthly'})
                    ds_V_this[f'{var_input_new[i]}_monthlytimeseries'] = da_aux
                    del da_aux
                    da_aux = ds_V_projection[var_list2_new[i]].resample(time_counter='AS', label='left', loffset=timedelta(days=182)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_annual'})
                    ds_V_this[f'{var_input_new[i]}_annualtimeseries'] = da_aux
                    del da_aux            
                    ds_V_this[f'{var_input_new[i]}_meanmap'] = ds_V_projection[var_list2_new[i]].mean(dim='time_counter', skipna=True)
                    ds_V_this[f'{var_input_new[i]}_monthlymap'] = ds_V_projection[var_list2_new[i]].groupby("time_counter.month").mean('time_counter', skipna=True)
                    ds_V_this[f'{var_input_new[i]}_seasonalmap'] = ds_V_projection[var_list2_new[i]].groupby("time_counter.season").mean('time_counter', skipna=True)
                    ds_V_this[f'{var_input_new[i]}_dailymeanyear'] = ds_V_projection[var_list2_new[i]].groupby("time_counter.dayofyear").mean('time_counter').mean(dim=["x", "y"], skipna = True)
                    da_aux = ds_V_projection[var_list2_new[i]].groupby("time_counter.month").mean('time_counter').mean(dim=["x", "y"], skipna = True).rename({'month': 'monthofyear'})
                    ds_V_this[f'{var_input_new[i]}_monthlymeanyear'] = da_aux
                    del da_aux
                    ds_V_this = ds_V_this.swap_dims({"x": "nav_lon", "y": "nav_lat"})
                if grid_input_new[i] == 'W':
                    ds_W_this[f'{var_input_new[i]}_dailytimeseries'] = (ds_W_projection[var_list2_new[i]] * da_area_T).sum(dim=["x", "y"], skipna = True) / da_area_T.sum(dim=["x", "y"], skipna = True)                
                    da_aux = ds_W_projection[var_list2_new[i]].resample(time_counter='1M', label='left', loffset=timedelta(days=15)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_monthly'})
                    ds_W_this[f'{var_input_new[i]}_monthlytimeseries'] = da_aux
                    del da_aux
                    da_aux = ds_W_projection[var_list2_new[i]].resample(time_counter='AS', label='left', loffset=timedelta(days=182)).mean(skipna=True).mean(dim=["x", "y"], skipna = True).rename({'time_counter': 'time_counter_annual'})
                    ds_W_this[f'{var_input_new[i]}_annualtimeseries'] = da_aux
                    del da_aux
                    ds_W_this[f'{var_input_new[i]}_meanmap'] = ds_W_projection[var_list2_new[i]].mean(dim='time_counter', skipna=True)
                    ds_W_this[f'{var_input_new[i]}_monthlymap'] = ds_W_projection[var_list2_new[i]].groupby("time_counter.month").mean('time_counter', skipna=True)
                    ds_W_this[f'{var_input_new[i]}_seasonalmap'] = ds_W_projection[var_list2_new[i]].groupby("time_counter.season").mean('time_counter', skipna=True)
                    ds_W_this[f'{var_input_new[i]}_dailymeanyear'] = ds_W_projection[var_list2_new[i]].groupby("time_counter.dayofyear").mean('time_counter').mean(dim=["x", "y"], skipna = True)
                    da_aux = ds_W_projection[var_list2_new[i]].groupby("time_counter.month").mean('time_counter').mean(dim=["x", "y"], skipna = True).rename({'month': 'monthofyear'})
                    ds_W_this[f'{var_input_new[i]}_monthlymeanyear'] = da_aux
                    del da_aux
                    ds_W_this = ds_W_this.swap_dims({"x": "nav_lon", "y": "nav_lat"})
            print('    Projection')
            ds_T_this.to_netcdf(f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc', mode = mode_projection)        
            ds_U_this.to_netcdf(f'{path_file}projection_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc', mode = mode_projection)        
            ds_V_this.to_netcdf(f'{path_file}projection_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc', mode = mode_projection)        
            ds_W_this.to_netcdf(f'{path_file}projection_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc', mode = mode_projection)        
            del ds_T_this, ds_U_this, ds_V_this, ds_W_this

    if 'reanalysis' in run_input:

        # Identify time
        timei_this = timei[run_input.index("reanalysis")]
        timef_this = timef[run_input.index("reanalysis")]
        # Create new var input variable
        var_input_new = var_input.copy()
        var_list2_cmems_new = var_list2_cmems.copy()
        # Get file path
        path_file_this = (f'{path_file}reanalysis_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')       
        # Check if intermediate files exist and load files
        if os.path.exists(path_file_this) :
            ds_cmems_this_aux = xr.open_dataset(f'{path_file}reanalysis_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
            mode_cmems = "a"
            for i in range(0,len(var_input)):
                # Remove variable from var_input_new if it already exists in intermediate file
                if f'{var_input[i]}_dailytimeseries' in ds_cmems_this_aux.variables.keys():
                    var_input_new.remove(var_input[i])
                    var_list2_cmems_new.remove(var_list2_cmems[i])
            del ds_cmems_this_aux
        else:
            mode_cmems = "w"

        # If there are variables to load, calculate derived fields and save file
        if var_input_new != []:    
            ds_cmems_this = xr.Dataset()
            for i in range(0,len(var_input_new)):   
                ds_cmems_this[f'{var_input_new[i]}_dailytimeseries'] = (ds_cmems[var_list2_cmems_new[i]] * da_area_cmems).sum(dim=["nav_lat", "nav_lon"], skipna = True) / da_area_cmems.sum(dim=["nav_lat", "nav_lon"], skipna = True)
                da_aux = ds_cmems[var_list2_cmems_new[i]].resample(time_counter='1M', label='left', loffset=timedelta(days=15)).mean(skipna=True).mean(dim=["nav_lat", "nav_lon"], skipna = True).rename({'time_counter': 'time_counter_monthly'})
                ds_cmems_this[f'{var_input_new[i]}_monthlytimeseries'] = da_aux
                del da_aux
                da_aux = ds_cmems[var_list2_cmems_new[i]].resample(time_counter='AS', label='left', loffset=timedelta(days=182)).mean(skipna=True).mean(dim=["nav_lat", "nav_lon"], skipna = True).rename({'time_counter': 'time_counter_annual'})
                ds_cmems_this[f'{var_input_new[i]}_annualtimeseries'] = da_aux
                del da_aux
                ds_cmems_this[f'{var_input_new[i]}_meanmap'] = ds_cmems[var_list2_cmems_new[i]].mean(dim='time_counter', skipna=True)
                ds_cmems_this[f'{var_input_new[i]}_monthlymap'] = ds_cmems[var_list2_cmems_new[i]].groupby("time_counter.month").mean('time_counter', skipna=True)
                ds_cmems_this[f'{var_input_new[i]}_seasonalmap'] = ds_cmems[var_list2_cmems_new[i]].groupby("time_counter.season").mean('time_counter', skipna=True)
                ds_cmems_this[f'{var_input_new[i]}_dailymeanyear'] = ds_cmems[var_list2_cmems_new[i]].groupby("time_counter.dayofyear").mean('time_counter').mean(dim=["nav_lat", "nav_lon"], skipna = True)
                da_aux = ds_cmems[var_list2_cmems_new[i]].groupby("time_counter.month").mean('time_counter').mean(dim=["nav_lat", "nav_lon"], skipna = True).rename({'month': 'monthofyear'})
                ds_cmems_this[f'{var_input_new[i]}_monthlymeanyear'] = da_aux
                del da_aux
            # Interpolate Dataset to model's grid
            if 'historical' in run_input or 'projection' in run_input:
                if 'historical' in run_input:
                    # Identify time
                    timei_this = timei[run_input.index("historical")]
                    timef_this = timef[run_input.index("historical")]
                    path_file_this = (f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
                    ds_T_this = xr.open_dataset(f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
                    ds_U_this = xr.open_dataset(f'{path_file}historical_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
                    ds_V_this = xr.open_dataset(f'{path_file}historical_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
                    ds_W_this = xr.open_dataset(f'{path_file}historical_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
                else:
                    # Identify time
                    timei_this = timei[run_input.index("projection")]
                    timef_this = timef[run_input.index("projection")]
                    path_file_this = (f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
                    ds_T_this = xr.open_dataset(f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
                    ds_U_this = xr.open_dataset(f'{path_file}projection_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
                    ds_V_this = xr.open_dataset(f'{path_file}projection_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
                    ds_W_this = xr.open_dataset(f'{path_file}projection_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')

                # Identify time
                timei_this = timei[run_input.index("reanalysis")]
                timef_this = timef[run_input.index("reanalysis")]
                path_file_this = (f'{path_file}reanalysis_interp_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
                if os.path.exists(path_file_this):
                    mode_cmems_interp = "a"
                else:
                    mode_cmems_interp = "w"
                ds_cmems_this_interp = xr.Dataset()
                if 'T' in grid_input:
                    ds_cmems_this_interp = ds_cmems_this.interp_like(ds_T_this)
                elif 'U' in grid_input:
                    ds_cmems_this_interp = ds_cmems_this.interp_like(ds_U_this)
                elif 'V' in grid_input:
                    ds_cmems_this_interp = ds_cmems_this.interp_like(ds_V_this)
                elif 'W' in grid_input:
                    ds_cmems_this_interp = ds_cmems_this.interp_like(ds_W_this)
                for i in range(0,len(var_input_new)):   
                    ds_cmems_this_interp[f'{var_input_new[i]}_dailytimeseries'] = ds_cmems_this_interp[f'{var_input_new[i]}_dailytimeseries'].chunk(chunks={'time_counter': 1})
                    ds_cmems_this_interp[f'{var_input_new[i]}_monthlytimeseries'] = ds_cmems_this_interp[f'{var_input_new[i]}_monthlytimeseries'].chunk(chunks={'time_counter_monthly': 1})
                    ds_cmems_this_interp[f'{var_input_new[i]}_annualtimeseries'] = ds_cmems_this_interp[f'{var_input_new[i]}_annualtimeseries'].chunk(chunks={'time_counter_annual': 1})
                    ds_cmems_this_interp[f'{var_input_new[i]}_monthlymap'] = ds_cmems_this_interp[f'{var_input_new[i]}_monthlymap'].chunk(chunks={'month': 1})
                    ds_cmems_this_interp[f'{var_input_new[i]}_seasonalmap'] = ds_cmems_this_interp[f'{var_input_new[i]}_seasonalmap'].chunk(chunks={'season': 1})
                    ds_cmems_this_interp[f'{var_input_new[i]}_dailymeanyear'] = ds_cmems_this_interp[f'{var_input_new[i]}_dailymeanyear'].chunk(chunks={'dayofyear': 1})
                    ds_cmems_this_interp[f'{var_input_new[i]}_monthlymeanyear'] = ds_cmems_this_interp[f'{var_input_new[i]}_monthlymeanyear'].chunk(chunks={'monthofyear': 1})
                # Drop depth dimension
                #ds_cmems_this_interp = ds_cmems_this_interp.drop_dims('depth')
                # Save NetCDF file
                print('    Reanalysis (interp)')
                ds_cmems_this_interp.to_netcdf(f'{path_file}reanalysis_interp_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc', mode = mode_cmems_interp)
            # Drop depth dimension
            #ds_cmems_this = ds_cmems_this.drop_dims('depth')
            # Save NetCDF file
            print('    Reanalysis')
            ds_cmems_this.to_netcdf(f'{path_file}reanalysis_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc', mode = mode_cmems)         
            del ds_cmems_this

# Function to plot daily mean time series
def plot_dailytimeseries(path_mask, file_mask, var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, path_file):

    # Load datasets
    if 'historical' in run_input:
        # Identify time
        timei_this = timei[run_input.index("historical")]
        timef_this = timef[run_input.index("historical")]
        ds_T_historical = nc.Dataset(f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_historical = nc.Dataset(f'{path_file}historical_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_historical = nc.Dataset(f'{path_file}historical_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_historical = nc.Dataset(f'{path_file}historical_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'projection' in run_input:
        # Identify time
        timei_this = timei[run_input.index("projection")]
        timef_this = timef[run_input.index("projection")]
        ds_T_projection = nc.Dataset(f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_projection = nc.Dataset(f'{path_file}projection_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_projection = nc.Dataset(f'{path_file}projection_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_projection = nc.Dataset(f'{path_file}projection_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'reanalysis' in run_input:
        # Identify time
        timei_this = timei[run_input.index("reanalysis")]
        timef_this = timef[run_input.index("reanalysis")]
        ds_cmems = nc.Dataset(f'{path_file}reanalysis_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')

    for i in range(0,len(var_input)):

        # Single product
        if len(run_input)==1:
            if 'historical' in run_input:
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_historical[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_T_historical['time_counter'][:]:
                        time_this = np.append(time_this,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'U':
                    da = ds_U_historical[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_U_historical['time_counter'][:]:
                        time_this = np.append(time_this,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'V':
                    da = ds_V_historical[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_V_historical['time_counter'][:]:
                        time_this = np.append(time_this,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'W':
                    da = ds_W_historical[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_W_historical['time_counter'][:]:
                        time_this = np.append(time_this,date(1900,1,1) + timedelta(t/86400))
            if 'projection' in run_input:
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_projection[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_T_projection['time_counter'][:]:
                        time_this = np.append(time_this,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'U':
                    da = ds_U_projection[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_U_projection['time_counter'][:]:
                        time_this = np.append(time_this,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'V':
                    da = ds_V_projection[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_V_projection['time_counter'][:]:
                        time_this = np.append(time_this,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'W':
                    da = ds_W_projection[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_W_projection['time_counter'][:]:
                        time_this = np.append(time_this,date(1900,1,1) + timedelta(t/86400))
            if 'reanalysis' in run_input:
                # Get DataArray for current variable
                da = ds_cmems[f'{var_input[i]}_dailytimeseries']
                # Define time
                time_this = []
                for t in ds_cmems['time_counter'][:]:
                    time_this = np.append(time_this,date(1900,1,1) + timedelta(t/1440))

            # Plot figure
            fig = plt.figure()
            fig.set_size_inches(8, 4)
            ax = fig.add_axes([0.17,0.24,0.8,0.66])
            plt.plot(time_this, da[:], color='black', linewidth=1.5)
            ax.set_ylabel(cbar_title[i])
            ax.set_xlabel('Time')
            mean_this = str(np.round(np.nanmean(da),2))
            std_this = str(np.round(np.nanstd(da),2))
            var_this = str(np.round(np.nanvar(da),2))
            ax.set_title(f'Daily mean {var_long[i]} ({area_input[0]}°N-{area_input[1]}°N, {area_input[2]}°E-{area_input[3]}°E) \n '
                         f'Mean = {mean_this}, '
                         f'Std = {std_this}, '
                         f'Var = {var_this}, '
                         f'({run_input})')
            plt.savefig(f'{path_output_this}timeseries/{var_input[i]}_dailymean_{run_input[0]}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
            plt.close(fig)

            print(f'    {var_long[i]} OK.')

        # Comparison between two products
        if len(run_input)==2:

            if run_input[0] == 'historical':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_historical[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_T_historical['time_counter'][:]:
                        time_this_1 = np.append(time_this_1,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'U':
                    da_1 = ds_U_historical[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_U_historical['time_counter'][:]:
                        time_this_1 = np.append(time_this_1,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'V':
                    da_1 = ds_V_historical[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_V_historical['time_counter'][:]:
                        time_this_1 = np.append(time_this_1,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'W':
                    da_1 = ds_W_historical[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_W_historical['time_counter'][:]:
                        time_this_1 = np.append(time_this_1,date(1900,1,1) + timedelta(t/86400))
            elif run_input[1] == 'historical':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_historical[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_T_historical['time_counter'][:]:
                        time_this_2 = np.append(time_this_2,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'U':
                    da_2 = ds_U_historical[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_U_historical['time_counter'][:]:
                        time_this_2 = np.append(time_this_2,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'V':
                    da_2 = ds_V_historical[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_V_historical['time_counter'][:]:
                        time_this_2 = np.append(time_this_2,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'W':
                    da_2 = ds_W_historical[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_W_historical['time_counter'][:]:
                        time_this_2 = np.append(time_this_2,date(1900,1,1) + timedelta(t/86400))
            if run_input[0] == 'projection':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_projection[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_T_projection['time_counter'][:]:
                        time_this_1 = np.append(time_this_1,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'U':
                    da_1 = ds_U_projection[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_U_projection['time_counter'][:]:
                        time_this_1 = np.append(time_this_1,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'V':
                    da_1 = ds_V_projection[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_V_projection['time_counter'][:]:
                        time_this_1 = np.append(time_this_1,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'W':
                    da_1 = ds_W_projection[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_W_projection['time_counter'][:]:
                        time_this_1 = np.append(time_this_1,date(1900,1,1) + timedelta(t/86400))
            elif run_input[1] == 'projection':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_projection[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_T_projection['time_counter'][:]:
                        time_this_2 = np.append(time_this_2,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'U':
                    da_2 = ds_U_projection[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_U_projection['time_counter'][:]:
                        time_this_2 = np.append(time_this_2,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'V':
                    da_2 = ds_V_projection[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_V_projection['time_counter'][:]:
                        time_this_2 = np.append(time_this_2,date(1900,1,1) + timedelta(t/86400))
                elif grid_input[i] == 'W':
                    da_2 = ds_W_projection[f'{var_input[i]}_dailytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_W_projection['time_counter'][:]:
                        time_this_2 = np.append(time_this_2,date(1900,1,1) + timedelta(t/86400))
            if run_input[0] == 'reanalysis':
                # Get DataArray for current variable
                da_1 = ds_cmems[f'{var_input[i]}_dailytimeseries']
                # Define time
                time_this_1 = []
                for t in ds_cmems['time_counter'][:]:
                    time_this_1 = np.append(time_this_1,date(1900,1,1) + timedelta(t/1440))
            elif run_input[1] == 'reanalysis':
                da_2 = ds_cmems[f'{var_input[i]}_dailytimeseries']
                # Define time
                time_this_2 = []
                for t in ds_cmems['time_counter'][:]:
                    time_this_2 = np.append(time_this_2,date(1900,1,1) + timedelta(t/1440))

            len_diff = len(da_2) - len(da_1)
            if len_diff > 0:
                da_2 = da_2[0:-len_diff]
                time_this_2 = time_this_2[0:-len_diff]
            elif len_diff < 0:
                da_1 = da_1[0:len_diff]
                time_this_1 = time_this_1[0:len_diff]

            fig = plt.figure()
            fig.set_size_inches(8, 4)
            ax = fig.add_axes([0.1,0.24,0.87,0.54])
            plt.plot(time_this_1, da_1[:], color='steelblue', linewidth=1.5)
            plt.plot(time_this_2, da_2[:], color='darkred', linewidth=1.5)
            ax.set_ylabel(cbar_title[i])
            ax.set_xlabel('Time')
            mean_this = [str(np.round(np.nanmean(da_1),2)),str(np.round(np.nanmean(da_2),2))]
            std_this = [str(np.round(np.nanstd(da_1),2)),str(np.round(np.nanstd(da_2),2))]
            var_this = [str(np.round(np.nanvar(da_1),2)),str(np.round(np.nanvar(da_2),2))]
            corr_this = str(np.round(ma.corrcoef([ma.masked_invalid(da_1),ma.masked_invalid(da_2)]),2)[1][0])
            bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1,da_2))),2))
            rmse_this = str(np.round(mean_squared_error(da_1, da_2, squared=False),2))
            fig.text(0.55, 0.95, f'Daily mean {var_long[i]} ({area_input[0]}°N-{area_input[1]}°N, {area_input[2]}°E-{area_input[3]}°E)', ha="center", va="bottom", size="medium",color="black")
            fig.text(0.55, 0.90, f'Corr = {corr_this}, Bias = {bias_this}, RMSE = {rmse_this}', ha="center", va="bottom", size="medium", color="black") 
            fig.text(0.55, 0.85, f'Mean = {mean_this[0]}, Std = {std_this[0]}, ({run_input[0]})', ha="center", va="bottom", size="medium", color="steelblue")
            fig.text(0.55, 0.80, f'Mean = {mean_this[1]}, Std = {std_this[1]}, ({run_input[1]})', ha="center", va="bottom", size="medium",color="darkred")
            plt.savefig(f'{path_output_this}timeseries/{var_input[i]}_dailymean_{run_input[0]}_{run_input[1]}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
            plt.close(fig)

            print(f'    {var_long[i]} OK.')

    # Close datasets
    if 'historical' in run_input:
        ds_T_historical.close()
        ds_U_historical.close()
        ds_V_historical.close()
        ds_W_historical.close()
    if 'projection' in run_input:
        ds_T_projection.close()
        ds_U_projection.close()
        ds_V_projection.close()
        ds_W_projection.close()
    if 'reanalysis' in run_input:
        ds_cmems.close()

# Function to plot monthly mean time series
def plot_monthlytimeseries(path_mask, file_mask, var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, path_file):
    
    # Load datasets
    if 'historical' in run_input:
        # Identify time
        timei_this = timei[run_input.index("historical")]
        timef_this = timef[run_input.index("historical")]
        ds_T_historical = nc.Dataset(f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_historical = nc.Dataset(f'{path_file}historical_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_historical = nc.Dataset(f'{path_file}historical_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_historical = nc.Dataset(f'{path_file}historical_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'projection' in run_input:
        # Identify time
        timei_this = timei[run_input.index("projection")]
        timef_this = timef[run_input.index("projection")]
        ds_T_projection = nc.Dataset(f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_projection = nc.Dataset(f'{path_file}projection_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_projection = nc.Dataset(f'{path_file}projection_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_projection = nc.Dataset(f'{path_file}projection_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'reanalysis' in run_input:
        # Identify time
        timei_this = timei[run_input.index("reanalysis")]
        timef_this = timef[run_input.index("reanalysis")]
        ds_cmems = nc.Dataset(f'{path_file}reanalysis_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')

    for i in range(0,len(var_input)):

        # Single product
        if len(run_input)==1:
            if 'historical' in run_input:
                # Identify time
                timei_this = timei[run_input.index("historical")]
                timef_this = timef[run_input.index("historical")]
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_historical[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_T_historical['time_counter_monthly'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'U':
                    da = ds_U_historical[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_U_historical['time_counter_monthly'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'V':
                    da = ds_V_historical[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_V_historical['time_counter_monthly'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'W':
                    da = ds_W_historical[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_W_historical['time_counter_monthly'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
            if 'projection' in run_input:
                # Identify time
                timei_this = timei[run_input.index("projection")]
                timef_this = timef[run_input.index("projection")]
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_projection[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_T_projection['time_counter_monthly'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'U':
                    da = ds_U_projection[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_U_projection['time_counter_monthly'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'V':
                    da = ds_V_projection[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_V_projection['time_counter_monthly'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'W':
                    da = ds_W_projection[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this = []
                    for t in ds_W_projection['time_counter_monthly'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
            if 'reanalysis' in run_input:
                # Identify time
                timei_this = timei[run_input.index("reanalysis")]
                timef_this = timef[run_input.index("reanalysis")]
                # Get DataArray for current variable
                da = ds_cmems[f'{var_input[i]}_monthlytimeseries']
                # Define time
                time_this = []
                for t in ds_cmems['time_counter_monthly'][:]:
                    time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))

            # Plot figure
            fig = plt.figure()
            fig.set_size_inches(8, 4)
            ax = fig.add_axes([0.17,0.24,0.8,0.66])
            plt.plot(time_this, da[:], color='black', linewidth=1.5)
            ax.set_ylabel(cbar_title[i])
            ax.set_xlabel('Time')
            mean_this = str(np.round(np.nanmean(da),2))
            std_this = str(np.round(np.nanstd(da),2))
            var_this = str(np.round(np.nanvar(da),2))
            ax.set_title(f'Monthly mean {var_long[i]} ({area_input[0]}°N-{area_input[1]}°N, {area_input[2]}°E-{area_input[3]}°E) \n '
                         f'Mean = {mean_this}, '
                         f'Std = {std_this}, '
                         f'Var = {var_this}, '
                         f'({run_input})')
            plt.savefig(f'{path_output_this}timeseries/{var_input[i]}_monthlymean_{run_input[0]}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
            plt.close(fig)

            print(f'    {var_long[i]} OK.')

        # Comparison between two products
        if len(run_input)==2:

            if run_input[0] == 'historical':
                # Identify time
                timei_this = timei[run_input.index("historical")]
                timef_this = timef[run_input.index("historical")]
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_historical[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_T_historical['time_counter_monthly'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'U':
                    da_1 = ds_U_historical[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_U_historical['time_counter_monthly'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'V':
                    da_1 = ds_V_historical[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_V_historical['time_counter_monthly'][:]:
                        time_this_1 = np.append(time_this_,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'W':
                    da_1 = ds_W_historical[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_W_historical['time_counter_monthly'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
            elif run_input[1] == 'historical':
                # Identify time
                timei_this = timei[run_input.index("historical")]
                timef_this = timef[run_input.index("historical")]
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_historical[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_T_historical['time_counter_monthly'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'U':
                    da_2 = ds_U_historical[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_U_historical['time_counter_monthly'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'V':
                    da_2 = ds_V_historical[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_V_historical['time_counter_monthly'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'W':
                    da_2 = ds_W_historical[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_W_historical['time_counter_monthly'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
            if run_input[0] == 'projection':
                # Identify time
                timei_this = timei[run_input.index("projection")]
                timef_this = timef[run_input.index("projection")]
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_projection[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_T_projection['time_counter_monthly'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'U':
                    da_1 = ds_U_projection[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_U_projection['time_counter_monthly'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'V':
                    da_1 = ds_V_projection[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_V_projection['time_counter_monthly'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'W':
                    da_1 = ds_W_projection[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_W_projection['time_counter_monthly'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
            elif run_input[1] == 'projection':
                # Identify time
                timei_this = timei[run_input.index("projection")]
                timef_this = timef[run_input.index("projection")]
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_projection[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_T_projection['time_counter_monthly'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'U':
                    da_2 = ds_U_projection[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_U_projection['time_counter_monthly'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'V':
                    da_2 = ds_V_projection[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_V_projection['time_counter_monthly'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
                elif grid_input[i] == 'W':
                    da_2 = ds_W_projection[f'{var_input[i]}_monthlytimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_W_projection['time_counter_monthly'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
            if run_input[0] == 'reanalysis':
                # Identify time
                timei_this = timei[run_input.index("reanalysis")]
                timef_this = timef[run_input.index("reanalysis")]
                # Get DataArray for current variable
                da_1 = ds_cmems[f'{var_input[i]}_monthlytimeseries']
                # Define time
                time_this_1 = []
                for t in ds_cmems['time_counter_monthly'][:]:
                    time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))
            elif run_input[1] == 'reanalysis':
                # Identify time
                timei_this = timei[run_input.index("reanalysis")]
                timef_this = timef[run_input.index("reanalysis")]
                # Get DataArray for current variable
                da_2 = ds_cmems[f'{var_input[i]}_monthlytimeseries']
                # Define time
                time_this_2 = []
                for t in ds_cmems['time_counter_monthly'][:]:
                    time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+14))

            fig = plt.figure()
            fig.set_size_inches(8, 4)
            ax = fig.add_axes([0.1,0.24,0.87,0.54])
            plt.plot(time_this_1, da_1[:], color='steelblue', linewidth=1.5)
            plt.plot(time_this_2, da_2[:], color='darkred', linewidth=1.5)
            ax.set_ylabel(cbar_title[i])
            ax.set_xlabel('Time')
            mean_this = [str(np.round(np.nanmean(da_1),2)),str(np.round(np.nanmean(da_2),2))]
            std_this = [str(np.round(np.nanstd(da_1),2)),str(np.round(np.nanstd(da_2),2))]
            var_this = [str(np.round(np.nanvar(da_1),2)),str(np.round(np.nanvar(da_2),2))]
            corr_this = str(np.round(ma.corrcoef([ma.masked_invalid(da_1),ma.masked_invalid(da_2)]),2)[1][0])
            bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1,da_2))),2))
            rmse_this = str(np.round(mean_squared_error(da_1, da_2, squared=False),2))
            fig.text(0.55, 0.95, f'Monthly mean {var_long[i]} ({area_input[0]}°N-{area_input[1]}°N, {area_input[2]}°E-{area_input[3]}°E)', ha="center", va="bottom", size="medium",color="black")
            fig.text(0.55, 0.90, f'Corr = {corr_this}, Bias = {bias_this}, RMSE = {rmse_this}', ha="center", va="bottom", size="medium", color="black") 
            fig.text(0.55, 0.85, f'Mean = {mean_this[0]}, Std = {std_this[0]}, ({run_input[0]})', ha="center", va="bottom", size="medium", color="steelblue")
            fig.text(0.55, 0.80, f'Mean = {mean_this[1]}, Std = {std_this[1]}, ({run_input[1]})', ha="center", va="bottom", size="medium",color="darkred")
            plt.savefig(f'{path_output_this}timeseries/{var_input[i]}_monthlymean_{run_input[0]}_{run_input[1]}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
            plt.close(fig)

            print(f'    {var_long[i]} OK.')

    # Close datasets
    if 'historical' in run_input:
        ds_T_historical.close()
        ds_U_historical.close()
        ds_V_historical.close()
        ds_W_historical.close()
    if 'projection' in run_input:
        ds_T_projection.close()
        ds_U_projection.close()
        ds_V_projection.close()
        ds_W_projection.close()
    if 'reanalysis' in run_input:
        ds_cmems.close()

# Function to plot annual mean time series
def plot_annualtimeseries(path_mask, file_mask, var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, path_file):

    # Load datasets
    if 'historical' in run_input:
        # Identify time
        timei_this = timei[run_input.index("historical")]
        timef_this = timef[run_input.index("historical")]
        ds_T_historical = nc.Dataset(f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_historical = nc.Dataset(f'{path_file}historical_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_historical = nc.Dataset(f'{path_file}historical_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_historical = nc.Dataset(f'{path_file}historical_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'projection' in run_input:
        # Identify time
        timei_this = timei[run_input.index("projection")]
        timef_this = timef[run_input.index("projection")]
        ds_T_projection = nc.Dataset(f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_projection = nc.Dataset(f'{path_file}projection_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_projection = nc.Dataset(f'{path_file}projection_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_projection = nc.Dataset(f'{path_file}projection_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'reanalysis' in run_input:
        # Identify time
        timei_this = timei[run_input.index("reanalysis")]
        timef_this = timef[run_input.index("reanalysis")]
        ds_cmems = nc.Dataset(f'{path_file}reanalysis_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')

    for i in range(0,len(var_input)):

        # Single product
        if len(run_input)==1:
            if 'historical' in run_input:
                # Identify time
                timei_this = timei[run_input.index("historical")]
                timef_this = timef[run_input.index("historical")]
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_historical[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this = []
                    for t in ds_T_historical['time_counter_annual'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'U':
                    da = ds_U_historical[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this = []
                    for t in ds_U_historical['time_counter_annual'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'V':
                    da = ds_V_historical[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this = []
                    for t in ds_V_historical['time_counter_annual'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'W':
                    da = ds_W_historical[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this = []
                    for t in ds_W_historical['time_counter_annual'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
            if 'projection' in run_input:
                # Identify time
                timei_this = timei[run_input.index("projection")]
                timef_this = timef[run_input.index("projection")]
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_projection[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this = []
                    for t in ds_T_projection['time_counter_annual'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'U':
                    da = ds_U_projection[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this = []
                    for t in ds_U_projection['time_counter_annual'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'V':
                    da = ds_V_projection[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this = []
                    for t in ds_V_projection['time_counter_annual'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'W':
                    da = ds_W_projection[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this = []
                    for t in ds_W_projection['time_counter_annual'][:]:
                        time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
            if 'reanalysis' in run_input:
                # Identify time
                timei_this = timei[run_input.index("reanalysis")]
                timef_this = timef[run_input.index("reanalysis")]
                # Get DataArray for current variable
                da = ds_cmems[f'{var_input[i]}_annualtimeseries']
                # Define time
                time_this = []
                for t in ds_cmems['time_counter_annual'][:]:
                    time_this = np.append(time_this,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))

            # Plot figure
            fig = plt.figure()
            fig.set_size_inches(8, 4)
            ax = fig.add_axes([0.17,0.24,0.8,0.66])
            plt.plot(time_this, da[:], color='black', linewidth=1.5)
            ax.set_ylabel(cbar_title[i])
            ax.set_xlabel('Time')
            mean_this = str(np.round(np.nanmean(da),2))
            std_this = str(np.round(np.nanstd(da),2))
            var_this = str(np.round(np.nanvar(da),2))
            ax.set_title(f'Annual mean {var_long[i]} ({area_input[0]}°N-{area_input[1]}°N, {area_input[2]}°E-{area_input[3]}°E) \n '
                         f'Mean = {mean_this}, '
                         f'Std = {std_this}, '
                         f'Var = {var_this}, '
                         f'({run_input})')
            plt.savefig(f'{path_output_this}timeseries/{var_input[i]}_annualmean_{run_input[0]}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
            plt.close(fig)

            print(f'    {var_long[i]} OK.')

        # Comparison between two products
        if len(run_input)==2:

            if run_input[0] == 'historical':
                # Identify time
                timei_this = timei[run_input.index("historical")]
                timef_this = timef[run_input.index("historical")]
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_historical[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_T_historical['time_counter_annual'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'U':
                    da_1 = ds_U_historical[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_U_historical['time_counter_annual'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'V':
                    da_1 = ds_V_historical[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_V_historical['time_counter_annual'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'W':
                    da_1 = ds_W_historical[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_W_historical['time_counter_annual'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
            elif run_input[1] == 'historical':
                # Identify time
                timei_this = timei[run_input.index("historical")]
                timef_this = timef[run_input.index("historical")]
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_historical[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_T_historical['time_counter_annual'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'U':
                    da_2 = ds_U_historical[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_U_historical['time_counter_annual'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'V':
                    da_2 = ds_V_historical[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_V_historical['time_counter_annual'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'W':
                    da_2 = ds_W_historical[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_W_historical['time_counter_annual'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
            if run_input[0] == 'projection':
                # Identify time
                timei_this = timei[run_input.index("projection")]
                timef_this = timef[run_input.index("projection")]
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_projection[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_T_projection['time_counter_annual'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'U':
                    da_1 = ds_U_projection[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_U_projection['time_counter_annual'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'V':
                    da_1 = ds_V_projection[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_V_projection['time_counter_annual'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'W':
                    da_1 = ds_W_projection[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_1 = []
                    for t in ds_W_projection['time_counter_annual'][:]:
                        time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
            elif run_input[1] == 'projection':
                # Identify time
                timei_this = timei[run_input.index("projection")]
                timef_this = timef[run_input.index("projection")]
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_projection[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_T_projection['time_counter_annual'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'U':
                    da_2 = ds_U_projection[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_U_projection['time_counter_annual'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'V':
                    da_2 = ds_V_projection[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_V_projection['time_counter_annual'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
                elif grid_input[i] == 'W':
                    da_2 = ds_W_projection[f'{var_input[i]}_annualtimeseries']
                    # Define time
                    time_this_2 = []
                    for t in ds_W_projection['time_counter_annual'][:]:
                        time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
            if run_input[0] == 'reanalysis':
                # Identify time
                timei_this = timei[run_input.index("reanalysis")]
                timef_this = timef[run_input.index("reanalysis")]
                # Get DataArray for current variable
                da_1 = ds_cmems[f'{var_input[i]}_annualtimeseries']
                # Define time
                time_this_1 = []
                for t in ds_cmems['time_counter_annual'][:]:
                    time_this_1 = np.append(time_this_1,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))
            elif run_input[1] == 'reanalysis':
                # Identify time
                timei_this = timei[run_input.index("reanalysis")]
                timef_this = timef[run_input.index("reanalysis")]
                # Get DataArray for current variable
                da_2 = ds_cmems[f'{var_input[i]}_annualtimeseries']
                # Define time
                time_this_2 = []
                for t in ds_cmems['time_counter_annual'][:]:
                    time_this_2 = np.append(time_this_2,date(int(timei_this[0:4]),int(timei_this[4:]),1) + timedelta(float(t)+182))

            fig = plt.figure()
            fig.set_size_inches(8, 4)
            ax = fig.add_axes([0.1,0.24,0.87,0.54])
            plt.plot(time_this_1, da_1[:], color='steelblue', linewidth=1.5)
            plt.plot(time_this_2, da_2[:], color='darkred', linewidth=1.5)
            ax.set_ylabel(cbar_title[i])
            ax.set_xlabel('Time')
            mean_this = [str(np.round(np.nanmean(da_1),2)),str(np.round(np.nanmean(da_2),2))]
            std_this = [str(np.round(np.nanstd(da_1),2)),str(np.round(np.nanstd(da_2),2))]
            var_this = [str(np.round(np.nanvar(da_1),2)),str(np.round(np.nanvar(da_2),2))]
            corr_this = str(np.round(ma.corrcoef([ma.masked_invalid(da_1),ma.masked_invalid(da_2)]),2)[1][0])
            bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1,da_2))),2))
            rmse_this = str(np.round(mean_squared_error(da_1, da_2, squared=False),2))
            fig.text(0.55, 0.95, f'Annual mean {var_long[i]} ({area_input[0]}°N-{area_input[1]}°N, {area_input[2]}°E-{area_input[3]}°E)', ha="center", va="bottom", size="medium",color="black")
            fig.text(0.55, 0.90, f'Corr = {corr_this}, Bias = {bias_this}, RMSE = {rmse_this}', ha="center", va="bottom", size="medium", color="black") 
            fig.text(0.55, 0.85, f'Mean = {mean_this[0]}, Std = {std_this[0]}, ({run_input[0]})', ha="center", va="bottom", size="medium", color="steelblue")
            fig.text(0.55, 0.80, f'Mean = {mean_this[1]}, Std = {std_this[1]}, ({run_input[1]})', ha="center", va="bottom", size="medium",color="darkred")
            plt.savefig(f'{path_output_this}timeseries/{var_input[i]}_annualmean_{run_input[0]}_{run_input[1]}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
            plt.close(fig)

            print(f'    {var_long[i]} OK.')

    # Close datasets
    if 'historical' in run_input:
        ds_T_historical.close()
        ds_U_historical.close()
        ds_V_historical.close()
        ds_W_historical.close()
    if 'projection' in run_input:
        ds_T_projection.close()
        ds_U_projection.close()
        ds_V_projection.close()
        ds_W_projection.close()
    if 'reanalysis' in run_input:
        ds_cmems.close()

# Function to plot mean maps
def plot_meanmaps(var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, cmap_list, path_file):

    # Load datasets
    if 'historical' in run_input:
        # Identify time
        timei_this = timei[run_input.index("historical")]
        timef_this = timef[run_input.index("historical")]
        ds_T_historical = nc.Dataset(f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_historical = nc.Dataset(f'{path_file}historical_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_historical = nc.Dataset(f'{path_file}historical_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_historical = nc.Dataset(f'{path_file}historical_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'projection' in run_input:
        # Identify time
        timei_this = timei[run_input.index("projection")]
        timef_this = timef[run_input.index("projection")]
        ds_T_projection = nc.Dataset(f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_projection = nc.Dataset(f'{path_file}projection_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_projection = nc.Dataset(f'{path_file}projection_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_projection = nc.Dataset(f'{path_file}projection_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'reanalysis' in run_input:
        # Identify time
        timei_this = timei[run_input.index("reanalysis")]
        timef_this = timef[run_input.index("reanalysis")]
        # Load dataset
        ds_cmems = nc.Dataset(f'{path_file}reanalysis_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        # Load interpolated dataset if it exists (only used for comparison)
        path_ds_cmems_interp = f'{path_file}reanalysis_interp_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc'
        if os.path.exists(path_ds_cmems_interp):
            ds_cmems_interp = nc.Dataset(path_ds_cmems_interp)

    for i in range(0,len(var_input)):

        # Single product
        if len(run_input)==1:

            if 'historical' in run_input:
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_historical[f'{var_input[i]}_meanmap']
                    da_lat = ds_T_historical['nav_lat']
                    da_lon = ds_T_historical['nav_lon']
                elif grid_input[i] == 'U':
                    da = ds_U_historical[f'{var_input[i]}_meanmap']
                    da_lat = ds_U_historical['nav_lat']
                    da_lon = ds_U_historical['nav_lon']
                elif grid_input[i] == 'V':
                    da = ds_V_historical[f'{var_input[i]}_meanmap']
                    da_lat = ds_V_historical['nav_lat']
                    da_lon = ds_V_historical['nav_lon']
                elif grid_input[i] == 'W':
                    da = ds_W_historical[f'{var_input[i]}_meanmap']
                    da_lat = ds_W_historical['nav_lat']
                    da_lon = ds_W_historical['nav_lon']
            if 'projection' in run_input:
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_projection[f'{var_input[i]}_meanmap']
                    da_lat = ds_T_projection['nav_lat']
                    da_lon = ds_T_projection['nav_lon']
                elif grid_input[i] == 'U':
                    da = ds_U_projection[f'{var_input[i]}_meanmap']
                    da_lat = ds_U_projection['nav_lat']
                    da_lon = ds_U_projection['nav_lon']
                elif grid_input[i] == 'V':
                    da = ds_V_projection[f'{var_input[i]}_meanmap']
                    da_lat = ds_V_projection['nav_lat']
                    da_lon = ds_V_projection['nav_lon']
                elif grid_input[i] == 'W':
                    da = ds_W_projection[f'{var_input[i]}_meanmap']
                    da_lat = ds_W_projection['nav_lat']
                    da_lon = ds_W_projection['nav_lon']
            if 'reanalysis' in run_input:
                # Get DataArray for current variable
                da = ds_cmems[f'{var_input[i]}_meanmap']
                da_lat = ds_cmems['nav_lat']
                da_lon = ds_cmems['nav_lon']

            # Define range of colormap
            vmax, vmin = [np.nanmax(da), np.nanmin(da)]

            # If range is not between -1 and 1, round values
            if vmax > 1 or vmax < -1:
                vmax = math.ceil(vmax)
            if vmin > 1 or vmin < -1:
                vmin = math.floor(vmin)

            # Define box
            #box=datasetBoundingBox(da_1)

            # Define meridians and parallels
            #meridians, parallels = getMeridansNParallels(box)

            # Define cmap (shift midpoint if the map contains positive and negative values)
            #if vmax > 0 and vmin < 0:
            #    midpoint_this = 1 - vmax/(vmax - vmin)
            #    cmap_this = shiftedColorMap(eval(cmap_list[i]), midpoint=midpoint_this, name = f'cmapshifted_{str(i)}')
            #else:
            cmap_this = cmap_list[i]

            # Calculate statistics
            mean_this = str(np.round(np.nanmean(da),2))
            std_this = str(np.round(np.nanstd(da),2))
            var_this = str(np.round(np.nanvar(da),2))

            # Plot mean map
            fig = plt.figure()
            fig.set_size_inches(8, 6)
            ax = fig.add_subplot(111)
            ax.set_title(f'Mean {var_long[i]} ({timei_this[4:6]}-{timei_this[0:4]} to {timef_this[4:6]}-{timef_this[0:4]}) \n '
                         f'Mean = {mean_this}, '
                         f'Std = {std_this}, '
                         f'Var = {var_this}')
            m = Basemap(llcrnrlon=np.nanmin(da_lon), llcrnrlat=np.nanmin(da_lat), 
                        urcrnrlat=np.nanmax(da_lat), urcrnrlon=np.nanmax(da_lon), resolution='f')
            m.drawcoastlines()
            m.fillcontinents('Whitesmoke')
            #m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
            #m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
            im = ax.imshow(da, origin='lower', cmap=cmap_this, vmin=vmin,
                           vmax=vmax,
                           extent=[np.nanmin(da_lon),np.nanmax(da_lon), 
                                   np.nanmin(da_lat),np.nanmax(da_lat)])
            cbar=plt.colorbar(im)
            cbar.set_label(cbar_title[i])
            plt.savefig(f'{path_output_this}maps/{var_input[i]}_mean_{run_input[0]}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
            plt.close(fig)

            print(f'    {var_long[i]} OK.')

        # Comparison between two products
        if len(run_input)==2:

            if run_input[0] == 'historical':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_historical[f'{var_input[i]}_meanmap']
                    da_1_lat = ds_T_historical['nav_lat']
                    da_1_lon = ds_T_historical['nav_lon']
                elif grid_input[i] == 'U':
                    da_1 = ds_U_historical[f'{var_input[i]}_meanmap']
                    da_1_lat = ds_U_historical['nav_lat']
                    da_1_lon = ds_U_historical['nav_lon']
                elif grid_input[i] == 'V':
                    da_1 = ds_V_historical[f'{var_input[i]}_meanmap']
                    da_1_lat = ds_V_historical['nav_lat']
                    da_1_lon = ds_V_historical['nav_lon']
                elif grid_input[i] == 'W':
                    da_1 = ds_W_historical[f'{var_input[i]}_meanmap']
                    da_1_lat = ds_W_historical['nav_lat']
                    da_1_lon = ds_W_historical['nav_lon']
            elif run_input[1] == 'historical':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_historical[f'{var_input[i]}_meanmap']
                    da_2_lat = ds_T_historical['nav_lat']
                    da_2_lon = ds_T_historical['nav_lon']
                elif grid_input[i] == 'U':
                    da_2 = ds_U_historical[f'{var_input[i]}_meanmap']
                    da_2_lat = ds_U_historical['nav_lat']
                    da_2_lon = ds_U_historical['nav_lon']
                elif grid_input[i] == 'V':
                    da_2 = ds_V_historical[f'{var_input[i]}_meanmap']
                    da_2_lat = ds_V_historical['nav_lat']
                    da_2_lon = ds_V_historical['nav_lon']
                elif grid_input[i] == 'W':
                    da_2 = ds_W_historical[f'{var_input[i]}_meanmap']
                    da_2_lat = ds_W_historical['nav_lat']
                    da_2_lon = ds_W_historical['nav_lon']
            if run_input[0] == 'projection':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_projection[f'{var_input[i]}_meanmap']
                    da_1_lat = ds_T_projection['nav_lat']
                    da_1_lon = ds_T_projection['nav_lon']
                elif grid_input[i] == 'U':
                    da_1 = ds_U_projection[f'{var_input[i]}_meanmap']
                    da_1_lat = ds_U_projection['nav_lat']
                    da_1_lon = ds_U_projection['nav_lon']
                elif grid_input[i] == 'V':
                    da_1 = ds_V_projection[f'{var_input[i]}_meanmap']
                    da_1_lat = ds_V_projection['nav_lat']
                    da_1_lon = ds_V_projection['nav_lon']
                elif grid_input[i] == 'W':
                    da_1 = ds_W_projection[f'{var_input[i]}_meanmap']
                    da_1_lat = ds_W_projection['nav_lat']
                    da_1_lon = ds_W_projection['nav_lon']
            elif run_input[1] == 'projection':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_projection[f'{var_input[i]}_meanmap']
                    da_2_lat = ds_T_projection['nav_lat']
                    da_2_lon = ds_T_projection['nav_lon']
                elif grid_input[i] == 'U':
                    da_2 = ds_U_projection[f'{var_input[i]}_meanmap']
                    da_2_lat = ds_U_projection['nav_lat']
                    da_2_lon = ds_U_projection['nav_lon']
                elif grid_input[i] == 'V':
                    da_2 = ds_V_projection[f'{var_input[i]}_meanmap']
                    da_2_lat = ds_V_projection['nav_lat']
                    da_2_lon = ds_V_projection['nav_lon']
                elif grid_input[i] == 'W':
                    da_2 = ds_W_projection[f'{var_input[i]}_meanmap']
                    da_2_lat = ds_W_projection['nav_lat']
                    da_2_lon = ds_W_projection['nav_lon']
            if run_input[0] == 'reanalysis':
                # Get DataArray for current variable
                da_1 = ds_cmems[f'{var_input[i]}_meanmap']
                da_1_lat = ds_cmems['nav_lat']
                da_1_lon = ds_cmems['nav_lon']
                da_1_interp = ds_cmems_interp[f'{var_input[i]}_meanmap']
            elif run_input[1] == 'reanalysis':
                da_2 = ds_cmems[f'{var_input[i]}_meanmap']
                da_2_lat = ds_cmems['nav_lat']
                da_2_lon = ds_cmems['nav_lon']
                da_2_interp = ds_cmems_interp[f'{var_input[i]}_meanmap']

            # Define range of colormap
            vmax, vmin = [np.nanmax(da_1), np.nanmin(da_1)]

            # If range is not between -1 and 1, round values
            if vmax > 1 or vmax < -1:
                vmax = math.ceil(vmax)
            if vmin > 1 or vmin < -1:
                vmin = math.floor(vmin)

            # Define box
            #box=datasetBoundingBox(da_1)

            # Define meridians and parallels
            #meridians, parallels = getMeridansNParallels(box)

            # Define cmap (shift midpoint if the map contains positive and negative values)
            #if vmax > 0 and vmin < 0:
            #    midpoint_this = 1 - vmax/(vmax - vmin)
            #    cmap_this = shiftedColorMap('viridis', midpoint=midpoint_this, name = f'cmapshifted_{str(i)}')
                #cmap_this = shiftedColorMap(eval(cmap_list[i]), midpoint=midpoint_this, name = f'cmapshifted_{str(i)}')
            #else:
            cmap_this = cmap_list[i]

            # Calculate difference array
            if 'da_1_interp' in locals():
                da_diff = da_2[:].data - da_1_interp[:].data
            elif 'da_2_interp' in locals():
                da_diff = da_2_interp[:].data - da_1[:].data
            else:
                da_diff = da_2[:].data - da_1[:].data

            # Calculate statistics
            mean_this = [str(np.round(np.nanmean(da_1),2)),str(np.round(np.nanmean(da_2),2))]
            std_this = [str(np.round(np.nanstd(da_1),2)),str(np.round(np.nanstd(da_2),2))]
            var_this = [str(np.round(np.nanvar(da_1),2)),str(np.round(np.nanvar(da_2),2))]
            if 'da_1_interp' in locals():
                corr_this = str(np.round(ma.corrcoef(ma.masked_invalid(da_1_interp[:].data.flatten()),ma.masked_invalid(da_2[:].data.flatten())),2)[1][0])
                bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1_interp,da_2))),2))
                rmse_this = str(np.round(mean_squared_error(np.nan_to_num(da_1_interp), np.nan_to_num(da_2), squared=False),2))
            elif 'da_2_interp' in locals():
                corr_this = str(np.round(ma.corrcoef(ma.masked_invalid(da_1[:].data.flatten()),ma.masked_invalid(da_2_interp[:].data.flatten())),2)[1][0])
                bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1,da_2_interp))),2))
                rmse_this = str(np.round(mean_squared_error(np.nan_to_num(da_1), np.nan_to_num(da_2_interp), squared=False),2))
            else:
                corr_this = str(np.round(ma.corrcoef(ma.masked_invalid(da_1[:].data.flatten()),ma.masked_invalid(da_2[:].data.flatten())),2)[1][0])
                bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1,da_2))),2))
                rmse_this = str(np.round(mean_squared_error(np.nan_to_num(da_1), np.nan_to_num(da_2), squared=False),2))

            # Plot mean map
            fig = plt.figure()
            fig.set_size_inches(12, 4.5)

            fig.suptitle(f'Mean {var_long[i]} ({timei[0][4:6]}-{timei[0][0:4]} to {timef[0][4:6]}-{timef[0][0:4]}, {timei[1][4:6]}-{timei[1][0:4]} to {timef[1][4:6]}-{timef[1][0:4]})')

            ax = fig.add_subplot(131)
            ax.set_title(f'{run_input[0]}, '
                         f'Mean = {mean_this[0]}, \n '
                         f'Std = {std_this[0]}, '
                         f'Var = {var_this[0]}')
            if 'da_1_interp' in locals():
                m = Basemap(llcrnrlon=np.nanmin(da_2_lon), llcrnrlat=np.nanmin(da_2_lat), 
                            urcrnrlat=np.nanmax(da_2_lat), urcrnrlon=np.nanmax(da_2_lon), resolution='f')
            else:
                m = Basemap(llcrnrlon=np.nanmin(da_1_lon), llcrnrlat=np.nanmin(da_1_lat), 
                            urcrnrlat=np.nanmax(da_1_lat), urcrnrlon=np.nanmax(da_1_lon), resolution='f')
            m.drawcoastlines()
            m.fillcontinents('Whitesmoke')
            #m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
            #m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
            if 'da_1_interp' in locals():
                im = ax.imshow(da_1_interp, origin='lower', cmap=cmap_this, vmin=vmin,
                               vmax=vmax,
                               extent=[np.nanmin(da_2_lon),np.nanmax(da_2_lon), 
                                       np.nanmin(da_2_lat),np.nanmax(da_2_lat)])
            else:
                im = ax.imshow(da_1, origin='lower', cmap=cmap_this, vmin=vmin,
                               vmax=vmax,
                               extent=[np.nanmin(da_1_lon),np.nanmax(da_1_lon), 
                                       np.nanmin(da_1_lat),np.nanmax(da_1_lat)])
            cbar=plt.colorbar(im, location="bottom")
            cbar.set_label(cbar_title[i])

            ax_2 = fig.add_subplot(132)
            ax_2.set_title(f'{run_input[1]}, '
                           f'Mean = {mean_this[1]}, \n '
                           f'Std = {std_this[1]}, '
                           f'Var = {var_this[1]}')
            if 'da_2_interp' in locals():
                m = Basemap(llcrnrlon=np.nanmin(da_1_lon), llcrnrlat=np.nanmin(da_1_lat), 
                            urcrnrlat=np.nanmax(da_1_lat), urcrnrlon=np.nanmax(da_1_lon), resolution='f')
            else:
                m = Basemap(llcrnrlon=np.nanmin(da_2_lon), llcrnrlat=np.nanmin(da_2_lat), 
                            urcrnrlat=np.nanmax(da_2_lat), urcrnrlon=np.nanmax(da_2_lon), resolution='f')
            m.drawcoastlines()
            m.fillcontinents('Whitesmoke')
            #m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
            #m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
            if 'da_2_interp' in locals():
                im = ax_2.imshow(da_2_interp, origin='lower', cmap=cmap_this, vmin=vmin,
                               vmax=vmax,
                               extent=[np.nanmin(da_1_lon),np.nanmax(da_1_lon), 
                                       np.nanmin(da_1_lat),np.nanmax(da_1_lat)])
            else:
                im = ax_2.imshow(da_2, origin='lower', cmap=cmap_this, vmin=vmin,
                               vmax=vmax,
                               extent=[np.nanmin(da_2_lon),np.nanmax(da_2_lon), 
                                       np.nanmin(da_2_lat),np.nanmax(da_2_lat)])
            cbar=plt.colorbar(im, location="bottom")
            cbar.set_label(cbar_title[i])

            ax_3 = fig.add_subplot(133)
            ax_3.set_title('Difference, '
                           f'Corr = {corr_this}, \n '
                           f'Bias = {bias_this}, '
                           f'RMSE = {rmse_this}')
            if 'da_2_interp' in locals():
                m = Basemap(llcrnrlon=np.nanmin(da_1_lon), llcrnrlat=np.nanmin(da_1_lat), 
                            urcrnrlat=np.nanmax(da_1_lat), urcrnrlon=np.nanmax(da_1_lon), resolution='f')
            else:
                m = Basemap(llcrnrlon=np.nanmin(da_2_lon), llcrnrlat=np.nanmin(da_2_lat), 
                            urcrnrlat=np.nanmax(da_2_lat), urcrnrlon=np.nanmax(da_2_lon), resolution='f')
            m.drawcoastlines()
            m.fillcontinents('Whitesmoke')
            #m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
            #m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
            # define scale, with white at zero
            vmin_diff = np.nanmin(da_diff) 
            vmax_diff = np.nanmax(da_diff) 
            norm = colors.TwoSlopeNorm(vcenter=0)
            if 'da_2_interp' in locals():
                im = ax_3.imshow(da_diff, origin='lower', cmap=cm.balance, norm=norm,
                                 extent=[np.nanmin(da_1_lon),np.nanmax(da_1_lon), 
                                         np.nanmin(da_1_lat),np.nanmax(da_1_lat)])
            else:
                im = ax_3.imshow(da_diff, origin='lower', cmap=cm.balance, norm=norm,
                                 extent=[np.nanmin(da_2_lon),np.nanmax(da_2_lon), 
                                         np.nanmin(da_2_lat),np.nanmax(da_2_lat)])
            cbar=plt.colorbar(im, location="bottom")
            cbar.set_label(f'Diff ({cbar_title[i]})')

            plt.savefig(f'{path_output_this}maps/{var_input[i]}_mean_{run_input[0]}_{run_input[1]}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
            plt.close(fig)

            print(f'    {var_long[i]} OK.')

    # Close datasets
    if 'historical' in run_input:
        ds_T_historical.close()
        ds_U_historical.close()
        ds_V_historical.close()
        ds_W_historical.close()
    if 'projection' in run_input:
        ds_T_projection.close()
        ds_U_projection.close()
        ds_V_projection.close()
        ds_W_projection.close()
    if 'reanalysis' in run_input:
        ds_cmems.close()

# Function to plot monthly mean maps
def plot_monthlymeanmaps(var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, cmap_list, cmaprange_input, path_file):

    # Load datasets
    if 'historical' in run_input:
        # Identify time
        timei_this = timei[run_input.index("historical")]
        timef_this = timef[run_input.index("historical")]
        ds_T_historical = nc.Dataset(f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_historical = nc.Dataset(f'{path_file}historical_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_historical = nc.Dataset(f'{path_file}historical_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_historical = nc.Dataset(f'{path_file}historical_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'projection' in run_input:
        # Identify time
        timei_this = timei[run_input.index("projection")]
        timef_this = timef[run_input.index("projection")]
        ds_T_projection = nc.Dataset(f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_projection = nc.Dataset(f'{path_file}projection_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_projection = nc.Dataset(f'{path_file}projection_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_projection = nc.Dataset(f'{path_file}projection_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'reanalysis' in run_input:
        # Identify time
        timei_this = timei[run_input.index("reanalysis")]
        timef_this = timef[run_input.index("reanalysis")]
        # Load dataset
        ds_cmems = nc.Dataset(f'{path_file}reanalysis_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        # Load interpolated dataset if it exists (only used for comparison)
        path_ds_cmems_interp = f'{path_file}reanalysis_interp_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc'
        if os.path.exists(path_ds_cmems_interp):
            ds_cmems_interp = nc.Dataset(path_ds_cmems_interp)

    for i in range(0,len(var_input)):

        # Single product
        if len(run_input)==1:

            if 'historical' in run_input:
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_historical[f'{var_input[i]}_monthlymap']
                    da_lat = ds_T_historical['nav_lat']
                    da_lon = ds_T_historical['nav_lon']
                elif grid_input[i] == 'U':
                    da = ds_U_historical[f'{var_input[i]}_monthlymap']
                    da_lat = ds_U_historical['nav_lat']
                    da_lon = ds_U_historical['nav_lon']
                elif grid_input[i] == 'V':
                    da = ds_V_historical[f'{var_input[i]}_monthlymap']
                    da_lat = ds_V_historical['nav_lat']
                    da_lon = ds_V_historical['nav_lon']
                elif grid_input[i] == 'W':
                    da = ds_W_historical[f'{var_input[i]}_monthlymap']
                    da_lat = ds_W_historical['nav_lat']
                    da_lon = ds_W_historical['nav_lon']
            if 'projection' in run_input:
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_projection[f'{var_input[i]}_monthlymap']
                    da_lat = ds_T_projection['nav_lat']
                    da_lon = ds_T_projection['nav_lon']
                elif grid_input[i] == 'U':
                    da = ds_U_projection[f'{var_input[i]}_monthlymap']
                    da_lat = ds_U_projection['nav_lat']
                    da_lon = ds_U_projection['nav_lon']
                elif grid_input[i] == 'V':
                    da = ds_V_projection[f'{var_input[i]}_monthlymap']
                    da_lat = ds_V_projection['nav_lat']
                    da_lon = ds_V_projection['nav_lon']
                elif grid_input[i] == 'W':
                    da = ds_W_projection[f'{var_input[i]}_monthlymap']
                    da_lat = ds_W_projection['nav_lat']
                    da_lon = ds_W_projection['nav_lon']
            if 'reanalysis' in run_input:
                # Get DataArray for current variable
                da = ds_cmems[f'{var_input[i]}_monthlymap']
                da_lat = ds_cmems['nav_lat']
                da_lon = ds_cmems['nav_lon']

            # Create month dimension
            da_month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            da_month_str = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            for mo in range(0,len(da_month)):

                # Create variable for current month
                da_this = da[mo, :, :].data

                # Define range of colormap
                vmax, vmin = [np.nanmax(da_this), np.nanmin(da_this)]

                # If range is not between -1 and 1, round values
                if vmax > 1 or vmax < -1:
                    vmax = math.ceil(vmax)
                if vmin > 1 or vmin < -1:
                    vmin = math.floor(vmin)

                # Define box
                #box=datasetBoundingBox(da_1)

                # Define meridians and parallels
                #meridians, parallels = getMeridansNParallels(box)

                # Define cmap (shift midpoint if the map contains positive and negative values)
                #if vmax > 0 and vmin < 0:
                #    midpoint_this = 1 - vmax/(vmax - vmin)
                #    cmap_this = shiftedColorMap(eval(cmap_list[i]), midpoint=midpoint_this, name = f'cmapshifted_{str(i)}')
                #else:
                cmap_this = cmap_list[i]

                # Calculate statistics
                mean_this = str(np.round(np.nanmean(da_this),2))
                std_this = str(np.round(np.nanstd(da_this),2))
                var_this = str(np.round(np.nanvar(da_this),2))

                # Plot mean map
                fig = plt.figure()
                fig.set_size_inches(8, 6)
                ax = fig.add_subplot(111)
                ax.set_title(f'{da_month_str[mo]} Mean {var_long[i]} ({timei_this[4:6]}-{timei_this[0:4]} to {timef_this[4:6]}-{timef_this[0:4]}) \n '
                             f'Mean = {mean_this}, '
                             f'Std = {std_this}, '
                             f'Var = {var_this}')
                m = Basemap(llcrnrlon=np.nanmin(da_lon), llcrnrlat=np.nanmin(da_lat), 
                            urcrnrlat=np.nanmax(da_lat), urcrnrlon=np.nanmax(da_lon), resolution='f')
                m.drawcoastlines()
                m.fillcontinents('Whitesmoke')
                #m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
                #m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
                im = ax.imshow(da_this, origin='lower', cmap=cmap_this, vmin=vmin,
                               vmax=vmax,
                               extent=[np.nanmin(da_lon),np.nanmax(da_lon), 
                                       np.nanmin(da_lat),np.nanmax(da_lat)])
                cbar=plt.colorbar(im)
                cbar.set_label(cbar_title[i])
                plt.savefig(f'{path_output_this}maps/{var_input[i]}_{da_month_str[mo]}_mean_{run_input[0]}_{timei}-{timef}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
                plt.close(fig)

                print(f'    {var_long[i]} ({da_month_str[mo]}) OK.')

        # Comparison between two products
        if len(run_input)==2:

            if run_input[0] == 'historical':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_historical[f'{var_input[i]}_monthlymap']
                    da_1_lat = ds_T_historical['nav_lat']
                    da_1_lon = ds_T_historical['nav_lon']
                elif grid_input[i] == 'U':
                    da_1 = ds_U_historical[f'{var_input[i]}_monthlymap']
                    da_1_lat = ds_U_historical['nav_lat']
                    da_1_lon = ds_U_historical['nav_lon']
                elif grid_input[i] == 'V':
                    da_1 = ds_V_historical[f'{var_input[i]}_monthlymap']
                    da_1_lat = ds_V_historical['nav_lat']
                    da_1_lon = ds_V_historical['nav_lon']
                elif grid_input[i] == 'W':
                    da_1 = ds_W_historical[f'{var_input[i]}_monthlymap']
                    da_1_lat = ds_W_historical['nav_lat']
                    da_1_lon = ds_W_historical['nav_lon']
            elif run_input[1] == 'historical':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_historical[f'{var_input[i]}_monthlymap']
                    da_2_lat = ds_T_historical['nav_lat']
                    da_2_lon = ds_T_historical['nav_lon']
                elif grid_input[i] == 'U':
                    da_2 = ds_U_historical[f'{var_input[i]}_monthlymap']
                    da_2_lat = ds_U_historical['nav_lat']
                    da_2_lon = ds_U_historical['nav_lon']
                elif grid_input[i] == 'V':
                    da_2 = ds_V_historical[f'{var_input[i]}_monthlymap']
                    da_2_lat = ds_V_historical['nav_lat']
                    da_2_lon = ds_V_historical['nav_lon']
                elif grid_input[i] == 'W':
                    da_2 = ds_W_historical[f'{var_input[i]}_monthlymap']
                    da_2_lat = ds_W_historical['nav_lat']
                    da_2_lon = ds_W_historical['nav_lon']
            if run_input[0] == 'projection':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_projection[f'{var_input[i]}_monthlymap']
                    da_1_lat = ds_T_projection['nav_lat']
                    da_1_lon = ds_T_projection['nav_lon']
                elif grid_input[i] == 'U':
                    da_1 = ds_U_projection[f'{var_input[i]}_monthlymap']
                    da_1_lat = ds_U_projection['nav_lat']
                    da_1_lon = ds_U_projection['nav_lon']
                elif grid_input[i] == 'V':
                    da_1 = ds_V_projection[f'{var_input[i]}_monthlymap']
                    da_1_lat = ds_V_projection['nav_lat']
                    da_1_lon = ds_V_projection['nav_lon']
                elif grid_input[i] == 'W':
                    da_1 = ds_W_projection[f'{var_input[i]}_monthlymap']
                    da_1_lat = ds_W_projection['nav_lat']
                    da_1_lon = ds_W_projection['nav_lon']
            elif run_input[1] == 'projection':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_projection[f'{var_input[i]}_monthlymap']
                    da_2_lat = ds_T_projection['nav_lat']
                    da_2_lon = ds_T_projection['nav_lon']
                elif grid_input[i] == 'U':
                    da_2 = ds_U_projection[f'{var_input[i]}_monthlymap']
                    da_2_lat = ds_U_projection['nav_lat']
                    da_2_lon = ds_U_projection['nav_lon']
                elif grid_input[i] == 'V':
                    da_2 = ds_V_projection[f'{var_input[i]}_monthlymap']
                    da_2_lat = ds_V_projection['nav_lat']
                    da_2_lon = ds_V_projection['nav_lon']
                elif grid_input[i] == 'W':
                    da_2 = ds_W_projection[f'{var_input[i]}_monthlymap']
                    da_2_lat = ds_W_projection['nav_lat']
                    da_2_lon = ds_W_projection['nav_lon']
            if run_input[0] == 'reanalysis':
                # Get DataArray for current variable
                da_1 = ds_cmems[f'{var_input[i]}_monthlymap']
                da_1_lat = ds_cmems['nav_lat']
                da_1_lon = ds_cmems['nav_lon']
                da_1_interp = ds_cmems_interp[f'{var_input[i]}_monthlymap']
            elif run_input[1] == 'reanalysis':
                da_2 = ds_cmems[f'{var_input[i]}_monthlymap']
                da_2_lat = ds_cmems['nav_lat']
                da_2_lon = ds_cmems['nav_lon']
                da_2_interp = ds_cmems_interp[f'{var_input[i]}_monthlymap']

            # Create month dimension
            da_1_month = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
            da_1_month_str = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            for mo in range(0,len(da_1_month)):

                # Create variables for current month
                if 'da_1_interp' in locals():
                    da_1_this = da_1[mo, :, :].data
                    da_1_interp_this = da_1_interp[mo, :, :].data
                    da_2_this = da_2[mo, :, :].data
                elif 'da_2_interp' in locals():
                    da_1_this = da_1[mo, :, :].data
                    da_2_this = da_2[mo, :, :].data
                    da_2_interp_this = da_2_interp[mo, :, :].data
                else:
                    da_1_this = da_1[mo, :, :].data
                    da_2_this = da_2[mo, :, :].data

                vmax, vmin = [np.nanmax(da_1_this), np.nanmin(da_1_this)]

                # If range is not between -1 and 1, round values
                if vmax > 1 or vmax < -1:
                    vmax = math.ceil(vmax)
                if vmin > 1 or vmin < -1:
                    vmin = math.floor(vmin)

                # Define box
                #box=datasetBoundingBox(da_1)

                # Define meridians and parallels
                #meridians, parallels = getMeridansNParallels(box)

                # Define cmap (shift midpoint if the map contains positive and negative values)
                #if vmax > 0 and vmin < 0:
                #    midpoint_this = 1 - vmax/(vmax - vmin)
                #    cmap_this = shiftedColorMap(eval(cmap_list[i]), midpoint=midpoint_this, name = f'cmapshifted_{str(i)}')
                #else:
                cmap_this = cmap_list[i]

                # Calculate difference array
                if 'da_1_interp_this' in locals():
                    da_diff = da_2_this - da_1_interp_this
                elif 'da_2_interp_this' in locals():
                    da_diff = da_2_interp_this - da_1_this
                else:
                    da_diff = da_2_this - da_1_this

                # Calculate statistics
                mean_this = [str(np.round(np.nanmean(da_1_this),2)),str(np.round(np.nanmean(da_2_this),2))]
                std_this = [str(np.round(np.nanstd(da_1_this),2)),str(np.round(np.nanstd(da_2_this),2))]
                var_this = [str(np.round(np.nanvar(da_1_this),2)),str(np.round(np.nanvar(da_2_this),2))]
                if 'da_1_interp_this' in locals():
                    corr_this = str(np.round(ma.corrcoef(ma.masked_invalid(da_1_interp_this.flatten()),ma.masked_invalid(da_2_this.flatten())),2)[1][0])
                    bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1_interp_this,da_2_this))),2))
                    rmse_this = str(np.round(mean_squared_error(np.nan_to_num(da_1_interp_this), np.nan_to_num(da_2_this), squared=False),2))
                elif 'da_2_interp_this' in locals():
                    corr_this = str(np.round(ma.corrcoef(ma.masked_invalid(da_1_this.flatten()),ma.masked_invalid(da_2_interp_this.flatten())),2)[1][0])
                    bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1_this,da_2_interp_this))),2))
                    rmse_this = str(np.round(mean_squared_error(np.nan_to_num(da_1_this), np.nan_to_num(da_2_interp_this), squared=False),2))
                else:
                    corr_this = str(np.round(ma.corrcoef(ma.masked_invalid(da_1_this.flatten()),ma.masked_invalid(da_2_this.flatten())),2)[1][0])
                    bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1_this,da_2_this))),2))
                    rmse_this = str(np.round(mean_squared_error(np.nan_to_num(da_1_this), np.nan_to_num(da_2_this), squared=False),2))

                # Plot mean map
                fig = plt.figure()
                fig.set_size_inches(12, 4.5)

                fig.suptitle(f'{da_1_month_str[mo]} Mean {var_long[i]} ({timei[0][4:6]}-{timei[0][0:4]} to {timef[0][4:6]}-{timef[0][0:4]}, {timei[1][4:6]}-{timei[1][0:4]} to {timef[1][4:6]}-{timef[1][0:4]})')

                ax = fig.add_subplot(131)
                ax.set_title(f'{run_input[0]}, '
                             f'Mean = {mean_this[0]}, \n '
                             f'Std = {std_this[0]}, '
                             f'Var = {var_this[0]}')
                if 'da_1_interp_this' in locals():
                    m = Basemap(llcrnrlon=np.nanmin(da_2_lon), llcrnrlat=np.nanmin(da_2_lat), 
                                urcrnrlat=np.nanmax(da_2_lat), urcrnrlon=np.nanmax(da_2_lon), resolution='f')
                else:
                    m = Basemap(llcrnrlon=np.nanmin(da_1_lon), llcrnrlat=np.nanmin(da_1_lat), 
                                urcrnrlat=np.nanmax(da_1_lat), urcrnrlon=np.nanmax(da_1_lon), resolution='f')
                m.drawcoastlines()
                m.fillcontinents('Whitesmoke')
                #m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
                #m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
                if 'da_1_interp_this' in locals():
                    im = ax.imshow(da_1_interp_this, origin='lower', cmap=cmap_this, vmin=vmin,
                                   vmax=vmax,
                                   extent=[np.nanmin(da_2_lon),np.nanmax(da_2_lon), 
                                           np.nanmin(da_2_lat),np.nanmax(da_2_lat)])
                else:
                    im = ax.imshow(da_1_this, origin='lower', cmap=cmap_this, vmin=vmin,
                                   vmax=vmax,
                                   extent=[np.nanmin(da_1_lon),np.nanmax(da_1_lon), 
                                           np.nanmin(da_1_lat),np.nanmax(da_1_lat)])
                cbar=plt.colorbar(im, location="bottom")
                cbar.set_label(cbar_title[i])

                ax_2 = fig.add_subplot(132)
                ax_2.set_title(f'{run_input[1]}, '
                               f'Mean = {mean_this[1]}, \n '
                               f'Std = {std_this[1]}, '
                               f'Var = {var_this[1]}')
                if 'da_2_interp_this' in locals():
                    m = Basemap(llcrnrlon=np.nanmin(da_1_lon), llcrnrlat=np.nanmin(da_1_lat), 
                                urcrnrlat=np.nanmax(da_1_lat), urcrnrlon=np.nanmax(da_1_lon), resolution='f')
                else:
                    m = Basemap(llcrnrlon=np.nanmin(da_2_lon), llcrnrlat=np.nanmin(da_2_lat), 
                                urcrnrlat=np.nanmax(da_2_lat), urcrnrlon=np.nanmax(da_2_lon), resolution='f')
                m.drawcoastlines()
                m.fillcontinents('Whitesmoke')
                #m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
                #m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
                if 'da_2_interp_this' in locals():
                    im = ax_2.imshow(da_2_interp_this, origin='lower', cmap=cmap_this, vmin=vmin,
                                   vmax=vmax,
                                   extent=[np.nanmin(da_1_lon),np.nanmax(da_1_lon), 
                                           np.nanmin(da_1_lat),np.nanmax(da_1_lat)])
                else:
                    im = ax_2.imshow(da_2_this, origin='lower', cmap=cmap_this, vmin=vmin,
                                   vmax=vmax,
                                   extent=[np.nanmin(da_2_lon),np.nanmax(da_2_lon), 
                                           np.nanmin(da_2_lat),np.nanmax(da_2_lat)])
                cbar=plt.colorbar(im, location="bottom")
                cbar.set_label(cbar_title[i])

                ax_3 = fig.add_subplot(133)
                ax_3.set_title('Difference, '
                               f'Corr = {corr_this}, \n '
                               f'Bias = {bias_this}, '
                               f'RMSE = {rmse_this}')
                if 'da_2_interp' in locals():
                    m = Basemap(llcrnrlon=np.nanmin(da_1_lon), llcrnrlat=np.nanmin(da_1_lat), 
                                urcrnrlat=np.nanmax(da_1_lat), urcrnrlon=np.nanmax(da_1_lon), resolution='f')
                else:
                    m = Basemap(llcrnrlon=np.nanmin(da_2_lon), llcrnrlat=np.nanmin(da_2_lat), 
                                urcrnrlat=np.nanmax(da_2_lat), urcrnrlon=np.nanmax(da_2_lon), resolution='f')
                m.drawcoastlines()
                m.fillcontinents('Whitesmoke')
                #m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
                #m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
                # define scale, with white at zero
                vmin_diff = np.nanmin(da_diff) 
                vmax_diff = np.nanmax(da_diff) 
                norm = colors.TwoSlopeNorm(vcenter=0)
                if 'da_2_interp' in locals():
                    im = ax_3.imshow(da_diff, origin='lower', cmap=cm.balance, norm=norm,
                                     extent=[np.nanmin(da_1_lon),np.nanmax(da_1_lon), 
                                             np.nanmin(da_1_lat),np.nanmax(da_1_lat)])
                else:
                    im = ax_3.imshow(da_diff, origin='lower', cmap=cm.balance, norm=norm,
                                     extent=[np.nanmin(da_2_lon),np.nanmax(da_2_lon), 
                                             np.nanmin(da_2_lat),np.nanmax(da_2_lat)])
                cbar=plt.colorbar(im, location="bottom")
                cbar.set_label(f'Diff ({cbar_title[i]})')

                plt.savefig(f'{path_output_this}maps/{var_input[i]}_{da_1_month_str[mo]}_mean_{run_input[0]}_{run_input[1]}_{timei[0]}-{timef[0]}_{timei[1]}-{timef[1]}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
                plt.close(fig)

                print(f'    {var_long[i]} ({da_1_month_str[mo]}) OK.')

    # Close datasets
    if 'historical' in run_input:
        ds_T_historical.close()
        ds_U_historical.close()
        ds_V_historical.close()
        ds_W_historical.close()
    if 'projection' in run_input:
        ds_T_projection.close()
        ds_U_projection.close()
        ds_V_projection.close()
        ds_W_projection.close()
    if 'reanalysis' in run_input:
        ds_cmems.close()

# Function to plot seasonal mean maps
def plot_seasonalmeanmaps(var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, cmap_list, cmaprange_input, path_file):    

    # Load datasets
    if 'historical' in run_input:
        # Identify time
        timei_this = timei[run_input.index("historical")]
        timef_this = timef[run_input.index("historical")]
        ds_T_historical = nc.Dataset(f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_historical = nc.Dataset(f'{path_file}historical_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_historical = nc.Dataset(f'{path_file}historical_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_historical = nc.Dataset(f'{path_file}historical_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'projection' in run_input:
        # Identify time
        timei_this = timei[run_input.index("projection")]
        timef_this = timef[run_input.index("projection")]
        ds_T_projection = nc.Dataset(f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_projection = nc.Dataset(f'{path_file}projection_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_projection = nc.Dataset(f'{path_file}projection_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_projection = nc.Dataset(f'{path_file}projection_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'reanalysis' in run_input:
        # Identify time
        timei_this = timei[run_input.index("reanalysis")]
        timef_this = timef[run_input.index("reanalysis")]
        # Load dataset
        ds_cmems = nc.Dataset(f'{path_file}reanalysis_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        # Load interpolated dataset if it exists (only used for comparison)
        path_ds_cmems_interp = f'{path_file}reanalysis_interp_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc'
        if os.path.exists(path_ds_cmems_interp):
            ds_cmems_interp = nc.Dataset(path_ds_cmems_interp)

    for i in range(0,len(var_input)):

        # Single product
        if len(run_input)==1:

            if 'historical' in run_input:
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_historical[f'{var_input[i]}_seasonalmap']
                    da_lat = ds_T_historical['nav_lat']
                    da_lon = ds_T_historical['nav_lon']
                elif grid_input[i] == 'U':
                    da = ds_U_historical[f'{var_input[i]}_seasonalmap']
                    da_lat = ds_U_historical['nav_lat']
                    da_lon = ds_U_historical['nav_lon']
                elif grid_input[i] == 'V':
                    da = ds_V_historical[f'{var_input[i]}_seasonalmap']
                    da_lat = ds_V_historical['nav_lat']
                    da_lon = ds_V_historical['nav_lon']
                elif grid_input[i] == 'W':
                    da = ds_W_historical[f'{var_input[i]}_seasonalmap']
                    da_lat = ds_W_historical['nav_lat']
                    da_lon = ds_W_historical['nav_lon']
            if 'projection' in run_input:
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_projection[f'{var_input[i]}_seasonalmap']
                    da_lat = ds_T_projection['nav_lat']
                    da_lon = ds_T_projection['nav_lon']
                elif grid_input[i] == 'U':
                    da = ds_U_projection[f'{var_input[i]}_seasonalmap']
                    da_lat = ds_U_projection['nav_lat']
                    da_lon = ds_U_projection['nav_lon']
                elif grid_input[i] == 'V':
                    da = ds_V_projection[f'{var_input[i]}_seasonalmap']
                    da_lat = ds_V_projection['nav_lat']
                    da_lon = ds_V_projection['nav_lon']
                elif grid_input[i] == 'W':
                    da = ds_W_projection[f'{var_input[i]}_seasonalmap']
                    da_lat = ds_W_projection['nav_lat']
                    da_lon = ds_W_projection['nav_lon']
            if 'reanalysis' in run_input:
                # Get DataArray for current variable
                da = ds_cmems[f'{var_input[i]}_seasonalmap']
                da_lat = ds_cmems['nav_lat']
                da_lon = ds_cmems['nav_lon']

            # Create month dimension
            da_season = [1, 2, 3, 4]
            da_season_str = ['DJF', 'MAM', 'JJA', 'SON']

            for mo in range(0,len(da_season)):

                # Create variable for current month
                da_this = da[mo, :, :].data

                # Define range of colormap
                vmax, vmin = [np.nanmax(da_this), np.nanmin(da_this)]

                # If range is not between -1 and 1, round values
                if vmax > 1 or vmax < -1:
                    vmax = math.ceil(vmax)
                if vmin > 1 or vmin < -1:
                    vmin = math.floor(vmin)

                # Define box
                #box=datasetBoundingBox(da_1)

                # Define meridians and parallels
                #meridians, parallels = getMeridansNParallels(box)

                # Define cmap (shift midpoint if the map contains positive and negative values)
                #if vmax > 0 and vmin < 0:
                #    midpoint_this = 1 - vmax/(vmax - vmin)
                #    cmap_this = shiftedColorMap(eval(cmap_list[i]), midpoint=midpoint_this, name = f'cmapshifted_{str(i)}')
                #else:
                cmap_this = cmap_list[i]

                # Calculate statistics
                mean_this = str(np.round(np.nanmean(da_this),2))
                std_this = str(np.round(np.nanstd(da_this),2))
                var_this = str(np.round(np.nanvar(da_this),2))

                # Plot mean map
                fig = plt.figure()
                fig.set_size_inches(8, 6)
                ax = fig.add_subplot(111)
                ax.set_title(f'{da_season_str[mo]} Mean {var_long[i]} ({timei_this[4:6]}-{timei_this[0:4]} to {timef_this[4:6]}-{timef_this[0:4]}) \n '
                             f'Mean = {mean_this}, '
                             f'Std = {std_this}, '
                             f'Var = {var_this}')
                m = Basemap(llcrnrlon=np.nanmin(da_lon), llcrnrlat=np.nanmin(da_lat), 
                            urcrnrlat=np.nanmax(da_lat), urcrnrlon=np.nanmax(da_lon), resolution='f')
                m.drawcoastlines()
                m.fillcontinents('Whitesmoke')
                #m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
                #m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
                im = ax.imshow(da_this, origin='lower', cmap=cmap_this, vmin=vmin,
                               vmax=vmax,
                               extent=[np.nanmin(da_lon),np.nanmax(da_lon), 
                                       np.nanmin(da_lat),np.nanmax(da_lat)])
                cbar=plt.colorbar(im)
                cbar.set_label(cbar_title[i])
                plt.savefig(f'{path_output_this}maps/{var_input[i]}_{da_season_str[mo]}_mean_{run_input[0]}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
                plt.close(fig)

                print(f'    {var_long[i]} ({da_season_str[mo]}) OK.')

        # Comparison between two products
        if len(run_input)==2:

            if run_input[0] == 'historical':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_historical[f'{var_input[i]}_seasonalmap']
                    da_1_lat = ds_T_historical['nav_lat']
                    da_1_lon = ds_T_historical['nav_lon']
                elif grid_input[i] == 'U':
                    da_1 = ds_U_historical[f'{var_input[i]}_seasonalmap']
                    da_1_lat = ds_U_historical['nav_lat']
                    da_1_lon = ds_U_historical['nav_lon']
                elif grid_input[i] == 'V':
                    da_1 = ds_V_historical[f'{var_input[i]}_seasonalmap']
                    da_1_lat = ds_V_historical['nav_lat']
                    da_1_lon = ds_V_historical['nav_lon']
                elif grid_input[i] == 'W':
                    da_1 = ds_W_historical[f'{var_input[i]}_seasonalmap']
                    da_1_lat = ds_W_historical['nav_lat']
                    da_1_lon = ds_W_historical['nav_lon']
            elif run_input[1] == 'historical':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_historical[f'{var_input[i]}_seasonalmap']
                    da_2_lat = ds_T_historical['nav_lat']
                    da_2_lon = ds_T_historical['nav_lon']
                elif grid_input[i] == 'U':
                    da_2 = ds_U_historical[f'{var_input[i]}_seasonalmap']
                    da_2_lat = ds_U_historical['nav_lat']
                    da_2_lon = ds_U_historical['nav_lon']
                elif grid_input[i] == 'V':
                    da_2 = ds_V_historical[f'{var_input[i]}_seasonalmap']
                    da_2_lat = ds_V_historical['nav_lat']
                    da_2_lon = ds_V_historical['nav_lon']
                elif grid_input[i] == 'W':
                    da_2 = ds_W_historical[f'{var_input[i]}_seasonalmap']
                    da_2_lat = ds_W_historical['nav_lat']
                    da_2_lon = ds_W_historical['nav_lon']
            if run_input[0] == 'projection':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_projection[f'{var_input[i]}_seasonalmap']
                    da_1_lat = ds_T_projection['nav_lat']
                    da_1_lon = ds_T_projection['nav_lon']
                elif grid_input[i] == 'U':
                    da_1 = ds_U_projection[f'{var_input[i]}_seasonalmap']
                    da_1_lat = ds_U_projection['nav_lat']
                    da_1_lon = ds_U_projection['nav_lon']
                elif grid_input[i] == 'V':
                    da_1 = ds_V_projection[f'{var_input[i]}_seasonalmap']
                    da_1_lat = ds_V_projection['nav_lat']
                    da_1_lon = ds_V_projection['nav_lon']
                elif grid_input[i] == 'W':
                    da_1 = ds_W_projection[f'{var_input[i]}_seasonalmap']
                    da_1_lat = ds_W_projection['nav_lat']
                    da_1_lon = ds_W_projection['nav_lon']
            elif run_input[1] == 'projection':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_projection[f'{var_input[i]}_seasonalmap']
                    da_2_lat = ds_T_projection['nav_lat']
                    da_2_lon = ds_T_projection['nav_lon']
                elif grid_input[i] == 'U':
                    da_2 = ds_U_projection[f'{var_input[i]}_seasonalmap']
                    da_2_lat = ds_U_projection['nav_lat']
                    da_2_lon = ds_U_projection['nav_lon']
                elif grid_input[i] == 'V':
                    da_2 = ds_V_projection[f'{var_input[i]}_seasonalmap']
                    da_2_lat = ds_V_projection['nav_lat']
                    da_2_lon = ds_V_projection['nav_lon']
                elif grid_input[i] == 'W':
                    da_2 = ds_W_projection[f'{var_input[i]}_seasonalmap']
                    da_2_lat = ds_W_projection['nav_lat']
                    da_2_lon = ds_W_projection['nav_lon']
            if run_input[0] == 'reanalysis':
                # Get DataArray for current variable
                da_1 = ds_cmems[f'{var_input[i]}_seasonalmap']
                da_1_lat = ds_cmems['nav_lat']
                da_1_lon = ds_cmems['nav_lon']
                da_1_interp = ds_cmems_interp[f'{var_input[i]}_seasonalmap']
            elif run_input[1] == 'reanalysis':
                da_2 = ds_cmems[f'{var_input[i]}_seasonalmap']
                da_2_lat = ds_cmems['nav_lat']
                da_2_lon = ds_cmems['nav_lon']
                da_2_interp = ds_cmems_interp[f'{var_input[i]}_seasonalmap']

            # Create month dimension
            da_1_season = [1, 2, 3, 4]
            da_1_season_str = ['DJF', 'MAM', 'JJA', 'SON']

            for mo in range(0,len(da_1_season)):

                # Create variables for current month
                if 'da_1_interp' in locals():
                    da_1_this = da_1[mo, :, :].data
                    da_1_interp_this = da_1_interp[mo, :, :].data
                    da_2_this = da_2[mo, :, :].data
                elif 'da_2_interp' in locals():
                    da_1_this = da_1[mo, :, :].data
                    da_2_this = da_2[mo, :, :].data
                    da_2_interp_this = da_2_interp[mo, :, :].data
                else:
                    da_1_this = da_1[mo, :, :].data
                    da_2_this = da_2[mo, :, :].data

                vmax, vmin = [np.nanmax(da_1_this), np.nanmin(da_1_this)]

                # If range is not between -1 and 1, round values
                if vmax > 1 or vmax < -1:
                    vmax = math.ceil(vmax)
                if vmin > 1 or vmin < -1:
                    vmin = math.floor(vmin)

                # Define box
                #box=datasetBoundingBox(da_1)

                # Define meridians and parallels
                #meridians, parallels = getMeridansNParallels(box)

                # Define cmap (shift midpoint if the map contains positive and negative values)
                #if vmax > 0 and vmin < 0:
                #    midpoint_this = 1 - vmax/(vmax - vmin)
                #    cmap_this = shiftedColorMap(eval(cmap_list[i]), midpoint=midpoint_this, name = f'cmapshifted_{str(i)}')
                #else:
                cmap_this = cmap_list[i]

                # Calculate difference array
                if 'da_1_interp_this' in locals():
                    da_diff = da_2_this - da_1_interp_this
                elif 'da_2_interp_this' in locals():
                    da_diff = da_2_interp_this - da_1_this
                else:
                    da_diff = da_2_this - da_1_this

                # Calculate statistics
                mean_this = [str(np.round(np.nanmean(da_1_this),2)),str(np.round(np.nanmean(da_2_this),2))]
                std_this = [str(np.round(np.nanstd(da_1_this),2)),str(np.round(np.nanstd(da_2_this),2))]
                var_this = [str(np.round(np.nanvar(da_1_this),2)),str(np.round(np.nanvar(da_2_this),2))]
                if 'da_1_interp_this' in locals():
                    corr_this = str(np.round(ma.corrcoef(ma.masked_invalid(da_1_interp_this.flatten()),ma.masked_invalid(da_2_this.flatten())),2)[1][0])
                    bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1_interp_this,da_2_this))),2))
                    rmse_this = str(np.round(mean_squared_error(np.nan_to_num(da_1_interp_this), np.nan_to_num(da_2_this), squared=False),2))
                elif 'da_2_interp_this' in locals():
                    corr_this = str(np.round(ma.corrcoef(ma.masked_invalid(da_1_this.flatten()),ma.masked_invalid(da_2_interp_this.flatten())),2)[1][0])
                    bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1_this,da_2_interp_this))),2))
                    rmse_this = str(np.round(mean_squared_error(np.nan_to_num(da_1_this), np.nan_to_num(da_2_interp_this), squared=False),2))
                else:
                    corr_this = str(np.round(ma.corrcoef(ma.masked_invalid(da_1_this.flatten()),ma.masked_invalid(da_2_this.flatten())),2)[1][0])
                    bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1_this,da_2_this))),2))
                    rmse_this = str(np.round(mean_squared_error(np.nan_to_num(da_1_this), np.nan_to_num(da_2_this), squared=False),2))

                # Plot mean map
                fig = plt.figure()
                fig.set_size_inches(12, 4.5)

                fig.suptitle(f'{da_1_season_str[mo]} Mean {var_long[i]} ({timei[0][4:6]}-{timei[0][0:4]} to {timef[0][4:6]}-{timef[0][0:4]}, {timei[1][4:6]}-{timei[1][0:4]} to {timef[1][4:6]}-{timef[1][0:4]})')

                ax = fig.add_subplot(131)
                ax.set_title(f'{run_input[0]}, '
                             f'Mean = {mean_this[0]}, \n '
                             f'Std = {std_this[0]}, '
                             f'Var = {var_this[0]}')
                if 'da_1_interp_this' in locals():
                    m = Basemap(llcrnrlon=np.nanmin(da_2_lon), llcrnrlat=np.nanmin(da_2_lat), 
                                urcrnrlat=np.nanmax(da_2_lat), urcrnrlon=np.nanmax(da_2_lon), resolution='f')
                else:
                    m = Basemap(llcrnrlon=np.nanmin(da_1_lon), llcrnrlat=np.nanmin(da_1_lat), 
                                urcrnrlat=np.nanmax(da_1_lat), urcrnrlon=np.nanmax(da_1_lon), resolution='f')
                m.drawcoastlines()
                m.fillcontinents('Whitesmoke')
                #m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
                #m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
                if 'da_1_interp_this' in locals():
                    im = ax.imshow(da_1_interp_this, origin='lower', cmap=cmap_this, vmin=vmin,
                                   vmax=vmax,
                                   extent=[np.nanmin(da_2_lon),np.nanmax(da_2_lon), 
                                           np.nanmin(da_2_lat),np.nanmax(da_2_lat)])
                else:
                    im = ax.imshow(da_1_this, origin='lower', cmap=cmap_this, vmin=vmin,
                                   vmax=vmax,
                                   extent=[np.nanmin(da_1_lon),np.nanmax(da_1_lon), 
                                           np.nanmin(da_1_lat),np.nanmax(da_1_lat)])
                cbar=plt.colorbar(im, location="bottom")
                cbar.set_label(cbar_title[i])

                ax_2 = fig.add_subplot(132)
                ax_2.set_title(f'{run_input[1]}, '
                               f'Mean = {mean_this[1]}, \n '
                               f'Std = {std_this[1]}, '
                               f'Var = {var_this[1]}')
                if 'da_2_interp_this' in locals():
                    m = Basemap(llcrnrlon=np.nanmin(da_1_lon), llcrnrlat=np.nanmin(da_1_lat), 
                                urcrnrlat=np.nanmax(da_1_lat), urcrnrlon=np.nanmax(da_1_lon), resolution='f')
                else:
                    m = Basemap(llcrnrlon=np.nanmin(da_2_lon), llcrnrlat=np.nanmin(da_2_lat), 
                                urcrnrlat=np.nanmax(da_2_lat), urcrnrlon=np.nanmax(da_2_lon), resolution='f')
                m.drawcoastlines()
                m.fillcontinents('Whitesmoke')
                #m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
                #m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
                if 'da_2_interp_this' in locals():
                    im = ax_2.imshow(da_2_interp_this, origin='lower', cmap=cmap_this, vmin=vmin,
                                   vmax=vmax,
                                   extent=[np.nanmin(da_1_lon),np.nanmax(da_1_lon), 
                                           np.nanmin(da_1_lat),np.nanmax(da_1_lat)])
                else:
                    im = ax_2.imshow(da_2_this, origin='lower', cmap=cmap_this, vmin=vmin,
                                   vmax=vmax,
                                   extent=[np.nanmin(da_2_lon),np.nanmax(da_2_lon), 
                                           np.nanmin(da_2_lat),np.nanmax(da_2_lat)])
                cbar=plt.colorbar(im, location="bottom")
                cbar.set_label(cbar_title[i])

                ax_3 = fig.add_subplot(133)
                ax_3.set_title('Difference, '
                               f'Corr = {corr_this}, \n '
                               f'Bias = {bias_this}, '
                               f'RMSE = {rmse_this}')
                if 'da_2_interp' in locals():
                    m = Basemap(llcrnrlon=np.nanmin(da_1_lon), llcrnrlat=np.nanmin(da_1_lat), 
                                urcrnrlat=np.nanmax(da_1_lat), urcrnrlon=np.nanmax(da_1_lon), resolution='f')
                else:
                    m = Basemap(llcrnrlon=np.nanmin(da_2_lon), llcrnrlat=np.nanmin(da_2_lat), 
                                urcrnrlat=np.nanmax(da_2_lat), urcrnrlon=np.nanmax(da_2_lon), resolution='f')
                m.drawcoastlines()
                m.fillcontinents('Whitesmoke')
                #m.drawparallels(parallels, labels=[True, False, False, True], linewidth=0.1)
                #m.drawmeridians(meridians, labels=[True, False, False, True], linewidth=0.1)
                # define scale, with white at zero
                vmin_diff = np.nanmin(da_diff) 
                vmax_diff = np.nanmax(da_diff) 
                norm = colors.TwoSlopeNorm(vcenter=0)
                if 'da_2_interp' in locals():
                    im = ax_3.imshow(da_diff, origin='lower', cmap=cm.balance, norm=norm,
                                     extent=[np.nanmin(da_1_lon),np.nanmax(da_1_lon), 
                                             np.nanmin(da_1_lat),np.nanmax(da_1_lat)])
                else:
                    im = ax_3.imshow(da_diff, origin='lower', cmap=cm.balance, norm=norm,
                                     extent=[np.nanmin(da_2_lon),np.nanmax(da_2_lon), 
                                             np.nanmin(da_2_lat),np.nanmax(da_2_lat)])
                cbar=plt.colorbar(im, location="bottom")
                cbar.set_label(f'Diff ({cbar_title[i]})')

                plt.savefig(f'{path_output_this}maps/{var_input[i]}_{da_1_season_str[mo]}_mean_{run_input[0]}_{run_input[1]}_{timei[0]}-{timef[0]}_{timei[1]}-{timef[1]}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
                plt.close(fig)

                print(f'    {var_long[i]} ({da_1_season_str[mo]}) OK.')

    # Close datasets
    if 'historical' in run_input:
        ds_T_historical.close()
        ds_U_historical.close()
        ds_V_historical.close()
        ds_W_historical.close()
    if 'projection' in run_input:
        ds_T_projection.close()
        ds_U_projection.close()
        ds_V_projection.close()
        ds_W_projection.close()
    if 'reanalysis' in run_input:
        ds_cmems.close()

# Function to plot monthly mean time series
def plot_meanyear(path_mask, file_mask, var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, path_file):
        
    # Load datasets
    if 'historical' in run_input:
        # Identify time
        timei_this = timei[run_input.index("historical")]
        timef_this = timef[run_input.index("historical")]
        ds_T_historical = nc.Dataset(f'{path_file}historical_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_historical = nc.Dataset(f'{path_file}historical_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_historical = nc.Dataset(f'{path_file}historical_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_historical = nc.Dataset(f'{path_file}historical_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'projection' in run_input:
        # Identify time
        timei_this = timei[run_input.index("projection")]
        timef_this = timef[run_input.index("projection")]
        ds_T_projection = nc.Dataset(f'{path_file}projection_T_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_U_projection = nc.Dataset(f'{path_file}projection_U_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_V_projection = nc.Dataset(f'{path_file}projection_V_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
        ds_W_projection = nc.Dataset(f'{path_file}projection_W_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')
    if 'reanalysis' in run_input:
        # Identify time
        timei_this = timei[run_input.index("reanalysis")]
        timef_this = timef[run_input.index("reanalysis")]
        ds_cmems = nc.Dataset(f'{path_file}reanalysis_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.nc')

    for i in range(0,len(var_input)):

        # Single product
        if len(run_input)==1:
            if 'historical' in run_input:
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_historical[f'{var_input[i]}_dailymeanyear']
                    da2 = ds_T_historical[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'U':
                    da = ds_U_historical[f'{var_input[i]}_dailymeanyear']
                    da2 = ds_U_historical[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'V':
                    da = ds_V_historical[f'{var_input[i]}_dailymeanyear']
                    da2 = ds_V_historical[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'W':
                    da = ds_W_historical[f'{var_input[i]}_dailymeanyear']
                    da2 = ds_W_historical[f'{var_input[i]}_monthlymeanyear']
            if 'projection' in run_input:
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da = ds_T_projection[f'{var_input[i]}_dailymeanyear']
                    da2 = ds_T_projection[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'U':
                    da = ds_U_projection[f'{var_input[i]}_dailymeanyear']
                    da2 = ds_U_projection[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'V':
                    da = ds_V_projection[f'{var_input[i]}_dailymeanyear']
                    da2 = ds_V_projection[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'W':
                    da = ds_W_projection[f'{var_input[i]}_dailymeanyear']
                    da2 = ds_W_projection[f'{var_input[i]}_monthlymeanyear']
            if 'reanalysis' in run_input:
                # Get DataArray for current variable
                da = ds_cmems[f'{var_input[i]}_dailymeanyear']
                da2 = ds_cmems[f'{var_input[i]}_monthlymeanyear']

            # Plot figure
            fig = plt.figure()
            fig.set_size_inches(8, 4)
            ax = fig.add_axes([0.17,0.24,0.8,0.66])
            plt.plot(da[:], color='black', linewidth=1.5)
            ax.set_ylabel(cbar_title[i])
            ax.set_xlabel('Day of Year')
            mean_this = str(np.round(np.nanmean(da),2))
            std_this = str(np.round(np.nanstd(da),2))
            var_this = str(np.round(np.nanvar(da),2))
            ax.set_title(f'Daily mean-year {var_long[i]} ({area_input[0]}°N-{area_input[1]}°N, {area_input[2]}°E-{area_input[3]}°E) \n '
                         f'Mean = {mean_this}, '
                         f'Std = {std_this}, '
                         f'Var = {var_this}')
            plt.savefig(f'{path_output_this}timeseries/{var_input[i]}_meanyeardaily_{run_input}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
            plt.close(fig)

            # Plot figure
            fig = plt.figure()
            fig.set_size_inches(8, 4)
            ax = fig.add_axes([0.17,0.24,0.8,0.66])
            plt.plot(da2[:], color='black', linewidth=1.5)
            ax.set_ylabel(cbar_title[i])
            ax.set_xlabel('Month')
            mean_this = str(np.round(np.nanmean(da2),2))
            std_this = str(np.round(np.nanstd(da2),2))
            var_this = str(np.round(np.nanvar(da2),2))
            ax.set_title(f'Monthly mean-year {var_long[i]} ({area_input[0]}°N-{area_input[1]}°N, {area_input[2]}°E-{area_input[3]}°E) \n '
                         f'Mean = {mean_this}, '
                         f'Std = {std_this}, '
                         f'Var = {var_this}, '
                         f'({run_input})')
            plt.savefig(f'{path_output_this}timeseries/{var_input[i]}_meanyearmonthly_{run_input[0]}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
            plt.close(fig)

            print(f'    {var_long[i]} OK.')

        # Comparison between two products
        if len(run_input)==2:

            if run_input[0] == 'historical':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_historical[f'{var_input[i]}_dailymeanyear']
                    da2_1 = ds_T_historical[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'U':
                    da_1 = ds_U_historical[f'{var_input[i]}_dailymeanyear']
                    da2_1 = ds_U_historical[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'V':
                    da_1 = ds_V_historical[f'{var_input[i]}_dailymeanyear']
                    da2_1 = ds_V_historical[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'W':
                    da_1 = ds_W_historical[f'{var_input[i]}_dailymeanyear']
                    da2_1 = ds_W_historical[f'{var_input[i]}_monthlymeanyear']
            elif run_input[1] == 'historical':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_historical[f'{var_input[i]}_dailymeanyear']
                    da2_2 = ds_T_historical[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'U':
                    da_2 = ds_U_historical[f'{var_input[i]}_dailymeanyear']
                    da2_2 = ds_U_historical[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'V':
                    da_2 = ds_V_historical[f'{var_input[i]}_dailymeanyear']
                    da2_2 = ds_V_historical[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'W':
                    da_2 = ds_W_historical[f'{var_input[i]}_dailymeanyear']
                    da2_2 = ds_W_historical[f'{var_input[i]}_monthlymeanyear']
            if run_input[0] == 'projection':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_1 = ds_T_projection[f'{var_input[i]}_dailymeanyear']
                    da2_1 = ds_T_projection[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'U':
                    da_1 = ds_U_projection[f'{var_input[i]}_dailymeanyear']
                    da2_1 = ds_U_projection[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'V':
                    da_1 = ds_V_projection[f'{var_input[i]}_dailymeanyear']
                    da2_1 = ds_V_projection[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'W':
                    da_1 = ds_W_projection[f'{var_input[i]}_dailymeanyear']
                    da2_1 = ds_W_projection[f'{var_input[i]}_monthlymeanyear']
            elif run_input[1] == 'projection':
                # Get DataArray for current variable
                if grid_input[i] == 'T':
                    da_2 = ds_T_projection[f'{var_input[i]}_dailymeanyear']
                    da2_2 = ds_T_projection[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'U':
                    da_2 = ds_U_projection[f'{var_input[i]}_dailymeanyear']
                    da2_2 = ds_U_projection[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'V':
                    da_2 = ds_V_projection[f'{var_input[i]}_dailymeanyear']
                    da2_2 = ds_V_projection[f'{var_input[i]}_monthlymeanyear']
                elif grid_input[i] == 'W':
                    da_2 = ds_W_projection[f'{var_input[i]}_dailymeanyear']
                    da2_2 = ds_W_projection[f'{var_input[i]}_monthlymeanyear']
            if run_input[0] == 'reanalysis':
                # Get DataArray for current variable
                da_1 = ds_cmems[f'{var_input[i]}_dailymeanyear']
                da2_1 = ds_cmems[f'{var_input[i]}_monthlymeanyear']
            elif run_input[1] == 'reanalysis':
                da_2 = ds_cmems[f'{var_input[i]}_dailymeanyear']
                da2_2 = ds_cmems[f'{var_input[i]}_monthlymeanyear']

            # Plot figure
            fig = plt.figure()
            fig.set_size_inches(8, 4)
            ax = fig.add_axes([0.1,0.24,0.87,0.54])
            plt.plot(da_1[:], color='steelblue', linewidth=1.5)
            plt.plot(da_2[:], color='darkred', linewidth=1.5)
            ax.set_ylabel(cbar_title[i])
            ax.set_xlabel('Day of Year')
            mean_this = [str(np.round(np.nanmean(da_1),2)),str(np.round(np.nanmean(da_2),2))]
            std_this = [str(np.round(np.nanstd(da_1),2)),str(np.round(np.nanstd(da_2),2))]
            var_this = [str(np.round(np.nanvar(da_1),2)),str(np.round(np.nanvar(da_2),2))]
            corr_this = str(np.round(ma.corrcoef([ma.masked_invalid(da_1),ma.masked_invalid(da_2)]),2)[1][0])
            bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da_1,da_2))),2))
            rmse_this = str(np.round(mean_squared_error(da_1, da_2, squared=False),2))
            fig.text(0.55, 0.95, f'Daily mean year {var_long[i]} ({area_input[0]}°N-{area_input[1]}°N, {area_input[2]}°E-{area_input[3]}°E)', ha="center", va="bottom", size="medium",color="black")
            fig.text(0.55, 0.90, f'Corr = {corr_this}, Bias = {bias_this}, RMSE = {rmse_this}', ha="center", va="bottom", size="medium", color="black")                
            fig.text(0.55, 0.85, f'Mean = {mean_this[0]}, Std = {std_this[0]}, ({run_input[0]})', ha="center", va="bottom", size="medium", color="steelblue")
            fig.text(0.55, 0.80, f'Mean = {mean_this[1]}, Std = {std_this[1]}, ({run_input[1]})', ha="center", va="bottom", size="medium",color="darkred")
            plt.savefig(f'{path_output_this}timeseries/{var_input[i]}_meanyeardaily_{run_input[0]}_{run_input[1]}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
            plt.close(fig)

            # Plot figure
            fig = plt.figure()
            fig.set_size_inches(8, 4)
            ax = fig.add_axes([0.1,0.24,0.87,0.54])
            plt.plot(da2_1[:], color='steelblue', linewidth=1.5)
            plt.plot(da2_2[:], color='darkred', linewidth=1.5)
            ax.set_ylabel(cbar_title[i])
            ax.set_xlabel('Month')
            mean_this = [str(np.round(np.nanmean(da2_1),2)),str(np.round(np.nanmean(da2_2),2))]
            std_this = [str(np.round(np.nanstd(da2_1),2)),str(np.round(np.nanstd(da2_2),2))]
            var_this = [str(np.round(np.nanvar(da2_1),2)),str(np.round(np.nanvar(da2_2),2))]
            corr_this = str(np.round(ma.corrcoef([ma.masked_invalid(da2_1),ma.masked_invalid(da2_2)]),2)[1][0])
            bias_this = str(np.round(np.nanmean(np.absolute(np.subtract(da2_1,da2_2))),2))
            rmse_this = str(np.round(mean_squared_error(da2_1, da2_2, squared=False),2))
            fig.text(0.55, 0.95, f'Monthly mean year {var_long[i]} ({area_input[0]}°N-{area_input[1]}°N, {area_input[2]}°E-{area_input[3]}°E)', ha="center", va="bottom", size="medium",color="black")
            fig.text(0.55, 0.90, f'Corr = {corr_this}, Bias = {bias_this}, RMSE = {rmse_this}', ha="center", va="bottom", size="medium", color="black")        
            fig.text(0.55, 0.85, f'Mean = {mean_this[0]}, Std = {std_this[0]}, Var = {var_this[0]}, ({run_input[0]})', ha="center", va="bottom", size="medium", color="steelblue")
            fig.text(0.55, 0.80, f'Mean = {mean_this[1]}, Std = {std_this[1]}, Var = {var_this[1]}, ({run_input[1]})', ha="center", va="bottom", size="medium",color="darkred")
            plt.savefig(f'{path_output_this}timeseries/{var_input[i]}_meanyearmonthly_{run_input[0]}_{run_input[1]}_{timei_this}-{timef_this}_{area_input[0]}N-{area_input[1]}N_{area_input[2]}E-{area_input[3]}E.png')
            plt.close(fig)

            print(f'    {var_long[i]} OK.')

    # Close datasets
    if 'historical' in run_input:
        ds_T_historical.close()
        ds_U_historical.close()
        ds_V_historical.close()
        ds_W_historical.close()
    if 'projection' in run_input:
        ds_T_projection.close()
        ds_U_projection.close()
        ds_V_projection.close()
        ds_W_projection.close()
    if 'reanalysis' in run_input:
        ds_cmems.close()

# Function to copy ndv_input.py to output directory
def save_input(path_output_this):
    os.system('cp ndv_input.py ' + os.path.join(path_output_this,'ndv_input.py'))
