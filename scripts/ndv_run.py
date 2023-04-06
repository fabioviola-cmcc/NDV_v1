##################################################################
##################################################################
# Nemo Diagnostics & Validation (NDV) package
# ndv_run.py
#
# Afonso Gonçalves Neto - CMCC, OPA division
# afonso.goncalves@cmcc.it
#
# Lecce, IT, December, 29 2022
##################################################################
##################################################################
## Import packages
from ndv_tools import *
from ndv_jobs import *
from ndv_input import *
from ndv_checkinput import *
##################################################################                             

## Check for errors in ndv_input.py
print('1. Checking for errors in the input parameters')
check_inputpath(path_input)
check_filename(file_input)
check_maskpath(path_mask)
check_maskfile(file_mask)
check_outputpath(path_output)
check_outputsubpath(subpath_output)
check_run(run_input)
check_freq(freq_input)
#check_date(timei, timef, run_input)
check_area(area_input)
check_cmaprange(cmaprange_input)
check_var(var_input, run_input)
#check_plot(timei, timef, plot_input)

## Load input data and store in a Dataset
print('2. Loading input files')
ds_T_historical, ds_U_historical, ds_V_historical, ds_W_historical, var_list2, var_long, grid_input, cbar_title, cmap_list, path_output_this = load_data_historical(path_input, file_input, path_mask, file_mask, run_input, freq_input, timei, timef, area_input, var_input, path_output, subpath_output, path_file)
ds_T_projection, ds_U_projection, ds_V_projection, ds_W_projection, var_list2, var_long, grid_input, cbar_title, cmap_list, path_output_this = load_data_projection(path_input, file_input, path_mask, file_mask, run_input, freq_input, timei, timef, area_input, var_input, path_output, subpath_output, path_output_this, path_file)
ds_cmems, var_list2_cmems, var_long, cbar_title, cmap_list, path_output_this = load_data_reanalysis(path_input, file_input, path_mask, file_mask, run_input, freq_input, timei, timef, area_input, var_input, path_output, subpath_output, path_output_this, path_file)
    
## Save intermediate files
print('3. Saving intermediate files')
save_intermediatefiles(ds_T_historical, ds_U_historical, ds_V_historical, ds_W_historical, ds_T_projection, ds_U_projection, ds_V_projection, ds_W_projection, ds_cmems, path_mask, file_mask, var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, path_file)

# Job index
n=3

## Plot daily time series of spatial mean
if 1 in plot_input:
    n+=1
    print(f'{str(n)}. Plotting daily time series of spatial mean')
    plot_dailytimeseries(path_mask, file_mask, var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, path_file)

## Plot monthly time series of spatial mean
if 2 in plot_input:
    n+=1
    print(f'{str(n)}. Plotting monthly time series of spatial mean')
    plot_monthlytimeseries(path_mask, file_mask, var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, path_file)

## Plot annual time series of spatial mean
if 3 in plot_input:
    n+=1
    print(f'{str(n)}. Plotting annual time series of spatial mean')
    plot_annualtimeseries(path_mask, file_mask, var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, path_file)

## Plot 2D maps of mean fields
if 4 in plot_input:
    n+=1
    print(f'{str(n)}. Plotting 2D maps of mean fields')
    plot_meanmaps(var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, cmap_list, path_file)

## Plot 2D maps of monthly mean fields
if 5 in plot_input:
    n+=1
    print(f'{str(n)}. Plotting 2D maps of monthly mean fields')
    plot_monthlymeanmaps(var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, cmap_list, cmaprange_input, path_file)

## Plot 2D maps of seasonal mean fields
if 6 in plot_input:
    n+=1
    print(f'{str(n)}. Plotting 2D maps of seasonal mean fields')
    plot_seasonalmeanmaps(var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, cmap_list, cmaprange_input, path_file)

# Plot mean year
if 7 in plot_input:
    n+=1
    print(f'{str(n)}. Plotting mean year')
    plot_meanyear(path_mask, file_mask, var_list2, var_list2_cmems, var_long, run_input, grid_input, var_input, area_input, path_output_this, timei, timef, cbar_title, path_file)
    
# Save current ndv_input.py to output directory
n+=1
print(f'{str(n)}. Saving ndv_input.py to output directory')
save_input(path_output_this)
