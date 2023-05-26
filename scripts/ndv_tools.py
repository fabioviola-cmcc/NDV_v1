##################################################################
##################################################################
# Nemo Diagnostics & Validation (NDV) package
# ndv_tools.py
#
# Afonso Gonçalves Neto - CMCC, OPA division
# afonso.goncalves@cmcc.it
#
# Lecce, IT, December 29, 2022
##################################################################
##################################################################
## Import packages
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import datetime
from munch import Munch
import math
##################################################################

## Function to define list of months to be loaded
def create_time_list(timei, timef):
    # Create empty variable
    time_list = []
    # If only a single year will be loaded
    if timef[0:4] == timei[0:4]:
        y = timei[0:4]
        for m in range(int(timei[4:6]),int(timef[4:6])+1):
            if m < 10:
                time_list.append(y+'0'+str(m))
            else:
                time_list.append(y+str(m))
    # If multiple years will be loaded
    elif timef[0:4] > timei[0:4]:
        for y in range(int(timei[0:4]),int(timef[0:4])+1):
            if y == int(timei[0:4]):
                for m in range(int(timei[4:6]),13):
                    if m < 10:
                        time_list.append(str(y)+'0'+str(m))
                    else:
                        time_list.append(str(y)+str(m))
            elif y == int(timef[0:4]):
                for m in range(1,int(timef[4:6])+1):
                    if m < 10:
                        time_list.append(str(y)+'0'+str(m))
                    else:
                        time_list.append(str(y)+str(m))
            else:
                for m in range(1,13):
                    if m < 10:
                        time_list.append(str(y)+'0'+str(m))
                    else:
                        time_list.append(str(y)+str(m))
    return(time_list)

# Function to determine the current time (used in load_data to define the name of the output directory)
def define_current_time():
    current_time_aux = datetime.datetime.now()
    if current_time_aux.month < 10:
        current_time = f'{str(current_time_aux.year)}0{str(current_time_aux.month)}'
    else:
        current_time = f'{str(current_time_aux.year)}{str(current_time_aux.month)}'
    if current_time_aux.day < 10:
        current_time = f'{current_time}0{str(current_time_aux.day)}'
    else:
        current_time = f'{current_time}{str(current_time_aux.day)}'
    if current_time_aux.hour < 10:
        current_time = f'{current_time}_0{str(current_time_aux.hour)}'
    else:
        current_time = f'{current_time}_{str(current_time_aux.hour)}'
    if current_time_aux.minute < 10:
        current_time = f'{current_time}0{str(current_time_aux.minute)}'
    else:
        current_time = f'{current_time}{str(current_time_aux.minute)}'
    return(current_time)

# Function to define the latitude and longitude limits of the plots (called when plotting maps)
def datasetBoundingBox(ds):
    box = Munch()
    box.xmin = math.floor(ds.nav_lon.values.min())
    box.ymin = math.floor(ds.nav_lat.values.min())
    box.xmax = math.ceil(ds.nav_lon.values.max())
    box.ymax = math.ceil(ds.nav_lat.values.max())
    return box

# Function to define the interval between Meridians and Parallels (called when plotting maps)
def meridParallelsStepper(diff):
    if diff<2.5:
        return  0.5
    else:
        if diff<5:
            return 1
        else:
            if diff<10:
                return 2
            else:
                if diff < 25:
                    return 5
                else:
                    return 10

# Function to define the Meridians and Parallels (called when plotting maps)
def getMeridansNParallels(box):
    start_m=int(box.xmin)
    end_m=int(box.xmax)
    step_m=meridParallelsStepper(end_m-start_m)

    start_p=int(box.ymin)
    end_p=int(box.ymax)
    step_p=meridParallelsStepper(end_p-start_p)

    step=min([step_p,step_m])

    meridians = np.arange(start_m-step,end_m+step, step)
    parallels = np.arange(start_p-step,end_p+step, step)
    return meridians,parallels

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

# Function to define the variables' NEMO names to be loaded (called in load_data)
def create_var_list(var_load):
    # Create list of empty strings with the same length as var_input
    var_list = [''] * len(var_load)
    # Attribute the original NEMO var name to each loaded variable
    if 'T' in var_load:
        var_list[var_load.index('T')] = 'votemper'
    if 'S' in var_load:
        var_list[var_load.index('S')] = 'vosaline'
    if 'SST' in var_load:
        var_list[var_load.index('SST')] = 'sosstsst'
    if 'SSS' in var_load:
        var_list[var_load.index('SSS')] = 'sosaline'
    if 'SSH' in var_load:
        var_list[var_load.index('SSH')] = 'sossheig'
    if 'WaterFlux' in var_load:
        var_list[var_load.index('WaterFlux')] = 'sowaflup'
    if 'SaltFlux' in var_load:
        var_list[var_load.index('SaltFlux')] = 'sosfldow' 
    if 'HeatFlux' in var_load:
        var_list[var_load.index('HeatFlux')] = 'sohefldo' 
    if 'MLD' in var_load:
        var_list[var_load.index('MLD')] = 'somxl010'
    if 'TurboclineDepth' in var_load:
        var_list[var_load.index('TurboclineDepth')] = 'somixhgt'
    if 'U' in var_load:
        var_list[var_load.index('U')] = 'vozocrtx'
    if 'EIVU' in var_load:
        var_list[var_load.index('EIVU')] = 'uocetr_eff'
    if 'TauX' in var_load:
        var_list[var_load.index('TauX')] = 'sozotaux'
    if 'V' in var_load:
        var_list[var_load.index('V')] = 'vomecrty'
    if 'EIVV' in var_load:
        var_list[var_load.index('EIVV')] = 'vocetr_eff'
    if 'TauY' in var_load:
        var_list[var_load.index('TauY')] = 'sometauy'
    if 'W' in var_load:
        var_list[var_load.index('W')] = 'vovecrtz'
    if 'EIVW' in var_load:
        var_list[var_load.index('EIVW')] = 'wocetr_eff'
    if 'VertEddyVisc' in var_load:
        var_list[var_load.index('VertEddyVisc')] = 'voddmavs'
    if 'LatEddyDiff' in var_load:
        var_list[var_load.index('LatEddyDiff')] = 'soleahtw'
        
    return(var_list)

# Function to define the variables' NEMO names to be plotted (called in plotting functions)
def create_var_list2(var_input):
    # Create list of empty strings with the same length as var_input
    var_list2 = [''] * len(var_input)
    # Attribute the original NEMO var name to each plotted variable
    # Loaded variables
    if 'T' in var_input:
        var_list2[var_input.index('T')] = 'votemper'
    if 'S' in var_input:
        var_list2[var_input.index('S')] = 'vosaline'
    if 'SST' in var_input:
        var_list2[var_input.index('SST')] = 'sosstsst'
    if 'SSS' in var_input:
        var_list2[var_input.index('SSS')] = 'sosaline'
    if 'SSH' in var_input:
        var_list2[var_input.index('SSH')] = 'sossheig'
    if 'WaterFlux' in var_input:
        var_list2[var_input.index('WaterFlux')] = 'sowaflup'
    if 'SaltFlux' in var_input:
        var_list2[var_input.index('SaltFlux')] = 'sosfldow' 
    if 'HeatFlux' in var_input:
        var_list2[var_input.index('HeatFlux')] = 'sohefldo' 
    if 'MLD' in var_input:
        var_list2[var_input.index('MLD')] = 'somxl010'
    if 'TurboclineDepth' in var_input:
        var_list2[var_input.index('TurboclineDepth')] = 'somixhgt'
    if 'U' in var_input:
        var_list2[var_input.index('U')] = 'vozocrtx'
    if 'EIVU' in var_input:
        var_list2[var_input.index('EIVU')] = 'uocetr_eff'
    if 'TauX' in var_input:
        var_list2[var_input.index('TauX')] = 'sozotaux'
    if 'V' in var_input:
        var_list2[var_input.index('V')] = 'vomecrty'
    if 'EIVV' in var_input:
        var_list2[var_input.index('EIVV')] = 'vocetr_eff'
    if 'TauY' in var_input:
        var_list2[var_input.index('TauY')] = 'sometauy'
    if 'W' in var_input:
        var_list2[var_input.index('W')] = 'vovecrtz'
    if 'EIVW' in var_input:
        var_list2[var_input.index('EIVW')] = 'wocetr_eff'
    if 'VertEddyVisc' in var_input:
        var_list2[var_input.index('VertEddyVisc')] = 'voddmavs'
    if 'LatEddyDiff' in var_input:
        var_list2[var_input.index('LatEddyDiff')] = 'soleahtw'
    # Calculated variables
    if 'SurfVel' in var_input:
        var_list2[var_input.index('SurfVel')] = 'SurfVel'
    if 'T_VertInt' in var_input:
        var_list2[var_input.index('T_VertInt')] = 'T_VertInt'
        
    return(var_list2)

# Function to define the variables' CMEMS names to be loaded (called in load_data)
def create_var_list_cmems(var_load):
    # Create list of empty strings with the same length as var_input
    var_list_cmems = [''] * len(var_load)
    # Attribute the original CMEMS var name to each loaded variable
    if 'T' in var_load:
        var_list_cmems[var_load.index('T')] = 'thetao'
    if 'S' in var_load:
        var_list_cmems[var_load.index('S')] = 'so'
    if 'zos' in var_load:
        var_list_cmems[var_load.index('zos')] = 'zos'
    if 'U' in var_load:
        var_list_cmems[var_load.index('U')] = 'uo'
    if 'V' in var_load:
        var_list_cmems[var_load.index('V')] = 'vo'
        
    return(var_list_cmems)

# Function to define the variables' CMEMS names to be plotted (called in plotting functions)
def create_var_list2_cmems(var_input):
    # Create list of empty strings with the same length as var_input
    var_list2_cmems = [''] * len(var_input)
    # Attribute the original CMEMS var name to each plotted variable
    # Loaded variables
    if 'T' in var_input:
        var_list2_cmems[var_input.index('T')] = 'thetao'
    if 'S' in var_input:
        var_list2_cmems[var_input.index('S')] = 'so'
    if 'zos' in var_input:
        var_list2_cmems[var_input.index('zos')] = 'zos'
    if 'U' in var_input:
        var_list2_cmems[var_input.index('U')] = 'uo'
    if 'V' in var_input:
        var_list2_cmems[var_input.index('V')] = 'vo'
    # Calculated variables
    if 'SST' in var_input:
        var_list2_cmems[var_input.index('SST')] = 'SST'
    if 'SSS' in var_input:
        var_list2_cmems[var_input.index('SSS')] = 'SSS'
    if 'SurfVel' in var_input:
        var_list2_cmems[var_input.index('SurfVel')] = 'SurfVel'
    if 'T_VertInt' in var_input:
        var_list2_cmems[var_input.index('T_VertInt')] = 'T_VertInt'
    if 'SSH' in var_input:
        var_list2_cmems[var_input.index('SSH')] = 'SSH'
        
    return(var_list2_cmems)

# Function to associate grid to each variable (called in load_data)
def create_grid_list(var_load):
    grid_list = [''] * len(var_load)
    if 'T' in var_load:
        grid_list[var_load.index('T')] = 'T'
    if 'S' in var_load:
        grid_list[var_load.index('S')] = 'T'
    if 'SST' in var_load:
        grid_list[var_load.index('SST')] = 'T'
    if 'SSS' in var_load:
        grid_list[var_load.index('SSS')] = 'T'
    if 'SSH' in var_load:
        grid_list[var_load.index('SSH')] = 'T'
    if 'WaterFlux' in var_load:
        grid_list[var_load.index('WaterFlux')] = 'T'
    if 'SaltFlux' in var_load:
        grid_list[var_load.index('SaltFlux')] = 'T' 
    if 'HeatFlux' in var_load:
        grid_list[var_load.index('HeatFlux')] = 'T' 
    if 'MLD' in var_load:
        grid_list[var_load.index('MLD')] = 'T'
    if 'TurboclineDepth' in var_load:
        grid_list[var_load.index('TurboclineDepth')] = 'T'
    if 'U' in var_load:
        grid_list[var_load.index('U')] = 'U'
    if 'EIVU' in var_load:
        grid_list[var_load.index('EIVU')] = 'U'
    if 'TauX' in var_load:
        grid_list[var_load.index('TauX')] = 'U'
    if 'V' in var_load:
        grid_list[var_load.index('V')] = 'V'
    if 'EIVV' in var_load:
        grid_list[var_load.index('EIVV')] = 'V'
    if 'TauY' in var_load:
        grid_list[var_load.index('TauY')] = 'V'
    if 'W' in var_load:
        grid_list[var_load.index('W')] = 'W'
    if 'EIVW' in var_load:
        grid_list[var_load.index('EIVW')] = 'W'
    if 'VertEddyVisc' in var_load:
        grid_list[var_load.index('VertEddyVisc')] = 'W'
    if 'LatEddyDiff' in var_load:
        grid_list[var_load.index('LatEddyDiff')] = 'W'
        
    return(grid_list)

# Function to associate grid to each variable (called in load_data)
def create_grid_input(var_input):
    grid_input = [''] * len(var_input)
    # Loaded Variables
    if 'T' in var_input:
        grid_input[var_input.index('T')] = 'T'
    if 'S' in var_input:
        grid_input[var_input.index('S')] = 'T'
    if 'SST' in var_input:
        grid_input[var_input.index('SST')] = 'T'
    if 'SSS' in var_input:
        grid_input[var_input.index('SSS')] = 'T'
    if 'SSH' in var_input:
        grid_input[var_input.index('SSH')] = 'T'
    if 'WaterFlux' in var_input:
        grid_input[var_input.index('WaterFlux')] = 'T'
    if 'SaltFlux' in var_input:
        grid_input[var_input.index('SaltFlux')] = 'T' 
    if 'HeatFlux' in var_input:
        grid_input[var_input.index('HeatFlux')] = 'T' 
    if 'MLD' in var_input:
        grid_input[var_input.index('MLD')] = 'T'
    if 'TurboclineDepth' in var_input:
        grid_input[var_input.index('TurboclineDepth')] = 'T'
    if 'U' in var_input:
        grid_input[var_input.index('U')] = 'U'
    if 'EIVU' in var_input:
        grid_input[var_input.index('EIVU')] = 'U'
    if 'TauX' in var_input:
        grid_input[var_input.index('TauX')] = 'U'
    if 'V' in var_input:
        grid_input[var_input.index('V')] = 'V'
    if 'EIVV' in var_input:
        grid_input[var_input.index('EIVV')] = 'V'
    if 'TauY' in var_input:
        grid_input[var_input.index('TauY')] = 'V'
    if 'W' in var_input:
        grid_input[var_input.index('W')] = 'W'
    if 'EIVW' in var_input:
        grid_input[var_input.index('EIVW')] = 'W'
    if 'VertEddyVisc' in var_input:
        grid_input[var_input.index('VertEddyVisc')] = 'W'
    if 'LatEddyDiff' in var_input:
        grid_input[var_input.index('LatEddyDiff')] = 'W'
    # Calculated variables
    if 'SurfVel' in var_input:
        grid_input[var_input.index('SurfVel')] = 'U'
    if 'T_VertInt' in var_input:
        grid_input[var_input.index('T_VertInt')] = 'T'
        
    return(grid_input)

# Function to associate grid to each variable (called in load_data)
def create_grid_list_cmems(var_load):
    grid_list_cmems = [''] * len(var_load)
    if 'T' in var_load:
        grid_list_cmems[var_load.index('T')] = 'TEMP'
    if 'S' in var_load:
        grid_list_cmems[var_load.index('S')] = 'PSAL'
    if 'zos' in var_load:
        grid_list_cmems[var_load.index('zos')] = 'ASLV'
    if 'U' in var_load:
        grid_list_cmems[var_load.index('U')] = 'RFVL'
    if 'V' in var_load:
        grid_list_cmems[var_load.index('V')] = 'RFVL'
        
    return(grid_list_cmems)

# Function to associate grid to each variable (called in load_data)
def create_grid_input_cmems(var_input):
    grid_input_cmems = [''] * len(var_input)
    # Loaded Variables
    if 'T' in var_input:
        grid_input_cmems[var_input.index('T')] = 'TEMP'
    if 'S' in var_input:
        grid_input_cmems[var_input.index('S')] = 'PSAL'
    if 'zos' in var_input:
        grid_input_cmems[var_input.index('zos')] = 'ASLV'
    if 'U' in var_input:
        grid_input_cmems[var_input.index('U')] = 'RFVL'
    if 'V' in var_input:
        grid_input_cmems[var_input.index('V')] = 'RFVL'
    # Calculated variables
    if 'SST' in var_input:
        grid_input_cmems[var_input.index('SST')] = 'TEMP'
    if 'SSS' in var_input:
        grid_input_cmems[var_input.index('SSS')] = 'PSAL'
    if 'SurfVel' in var_input:
        grid_input_cmems[var_input.index('SurfVel')] = 'RFVL'
    if 'T_VertInt' in var_input:
        grid_input_cmems[var_input.index('T_VertInt')] = 'TEMP'
    if 'SSH' in var_input:
        grid_input_cmems[var_input.index('SSH')] = 'ASLV'
        
    return(grid_input_cmems)

# Function to define the variables' long names (called in load_data)
def create_var_long(var_input):
    # Create list of empty strings with the same length as var_input
    var_long = [''] * len(var_input)
    # Attribute the long name to each loaded variable
    # Loaded Variables
    if 'T' in var_input:
        var_long[var_input.index('T')] = 'Temperature'
    if 'S' in var_input:
        var_long[var_input.index('S')] = 'Salinity'
    if 'SST' in var_input:
        var_long[var_input.index('SST')] = 'Sea Surface Temperature'
    if 'SSS' in var_input:
        var_long[var_input.index('SSS')] = 'Sea Surface Salinity'
    if 'SSH' in var_input:
        var_long[var_input.index('SSH')] = 'Sea Surface Height'
    if 'WaterFlux' in var_input:
        var_long[var_input.index('WaterFlux')] = 'Net Upward Water Flux'
    if 'SaltFlux' in var_input:
        var_long[var_input.index('SaltFlux')] = 'Surface Salt Flux'
    if 'HeatFlux' in var_input:
        var_long[var_input.index('HeatFlux')] = 'Net Downward Heat Flux'
    if 'MLD' in var_input:
        var_long[var_input.index('MLD')] = 'Mixed Layer Depth 0.01'
    if 'TurboclineDepth' in var_input:
        var_long[var_input.index('TurboclineDepth')] = 'Turbocline Depth'
    if 'U' in var_input:
        var_long[var_input.index('U')] = 'Zonal Current'
    if 'EIVU' in var_input:
        var_long[var_input.index('EIVU')] = 'Zonal EIV Current'
    if 'TauX' in var_input:
        var_long[var_input.index('TauX')] = 'Wind Stress along i-axis'
    if 'V' in var_input:
        var_long[var_input.index('V')] = 'Meridional Current'
    if 'EIVV' in var_input:
        var_long[var_input.index('EIVV')] = 'Meridional EIV Current'
    if 'TauY' in var_input:
        var_long[var_input.index('TauY')] = 'Wind Stress along j-axis'
    if 'W' in var_input:
        var_long[var_input.index('W')] = 'Vertical Velocity'
    if 'EIVW' in var_input:
        var_long[var_input.index('EIVW')] = 'Vertical EIV Velocity'
    if 'VertEddyVisc' in var_input:
        var_long[var_input.index('VertEddyVisc')] = 'Vertical Eddy Viscosity'
    if 'LatEddyDiff' in var_input:
        var_long[var_input.index('LatEddyDiff')] = 'Lateral Eddy Diffusivity'
    # Calculated variables
    if 'SurfVel' in var_input:
        var_long[var_input.index('SurfVel')] = 'Surface Velocity'
    if 'T_VertInt' in var_input:
        var_long[var_input.index('T_VertInt')] = 'Vertically-Integrated Temperature'
        
    return(var_long)

# Function to define colorbar titles (called in load_data)
def create_cbar_title(var_input):
    cbar_title = [''] * len(var_input)
    # Loaded Variables
    if 'T' in var_input:
        cbar_title[var_input.index('T')] = 'T (°C)'
    if 'S' in var_input:
        cbar_title[var_input.index('S')] = 'Salinity'
    if 'SST' in var_input:
        cbar_title[var_input.index('SST')] = 'SST (°C)'
    if 'SSS' in var_input:
        cbar_title[var_input.index('SSS')] = 'SSS'
    if 'SSH' in var_input:
        cbar_title[var_input.index('SSH')] = 'SSH (m)'
    if 'WaterFlux' in var_input:
        cbar_title[var_input.index('WaterFlux')] = 'Water Flux (Kg/m2/s)'
    if 'SaltFlux' in var_input:
        cbar_title[var_input.index('SaltFlux')] = 'Salt Flux (Kg/m2/s)'
    if 'HeatFlux' in var_input:
        cbar_title[var_input.index('HeatFlux')] = 'Heat Flux (W/m2)'
    if 'MLD' in var_input:
        cbar_title[var_input.index('MLD')] = 'MLD (m)'
    if 'TurboclineDepth' in var_input:
        cbar_title[var_input.index('TurboclineDepth')] = 'Turbocline Depth (m)'
    if 'U' in var_input:
        cbar_title[var_input.index('U')] = 'Zonal Current (m/s)'
    if 'EIVU' in var_input:
        cbar_title[var_input.index('EIVU')] = 'Zonal EIV Current (m/s)'
    if 'TauX' in var_input:
        cbar_title[var_input.index('TauX')] = 'Wind Stress along i-axis (N/m2)'
    if 'V' in var_input:
        cbar_title[var_input.index('V')] = 'Meridional Current (m/s)'
    if 'EIVV' in var_input:
        cbar_title[var_input.index('EIVV')] = 'Meridional EIV Current (m/s)'
    if 'TauY' in var_input:
        cbar_title[var_input.index('TauY')] = 'Wind Stress along j-axis (N/m2)'
    if 'W' in var_input:
        cbar_title[var_input.index('W')] = 'Vertical Velocity (m/s)'
    if 'EIVW' in var_input:
        cbar_title[var_input.index('EIVW')] = 'Vertical EIV Velocity (m/s)'
    if 'VertEddyVisc' in var_input:
        cbar_title[var_input.index('VertEddyVisc')] = 'Vertical Eddy Viscosity (m2/s)'
    if 'LatEddyDiff' in var_input:
        cbar_title[var_input.index('LatEddyDiff')] = 'Lateral Eddy Diffusivity (m2/s)'
    # Calculated variables
    if 'SurfVel' in var_input:
        cbar_title[var_input.index('SurfVel')] = 'Surface Velocity (m/s)'
    if 'T_VertInt' in var_input:
        cbar_title[var_input.index('T_VertInt')] = 'Vertically-Integrated Temp. (°C)'
        
    return(cbar_title)

# Function to define colormaps (called in load_data)
def create_cmap_list(var_input):
    cmap_list = [''] * len(var_input)
    # Loaded variables
    if 'T' in var_input:
        cmap_list[var_input.index('T')] = 'cmo.thermal'
    if 'S' in var_input:
        cmap_list[var_input.index('S')] = 'cmo.haline'
    if 'SST' in var_input:
        cmap_list[var_input.index('SST')] = 'cmo.thermal'
    if 'SSS' in var_input:
        cmap_list[var_input.index('SSS')] = 'cmo.haline'
    if 'SSH' in var_input:
        cmap_list[var_input.index('SSH')] = 'cmo.speed'
    if 'WaterFlux' in var_input:
        cmap_list[var_input.index('WaterFlux')] = 'cm.curl'
    if 'SaltFlux' in var_input:
        cmap_list[var_input.index('SaltFlux')] = 'cmo.delta'
    if 'HeatFlux' in var_input:
        cmap_list[var_input.index('HeatFlux')] = 'cm.balance'
    if 'MLD' in var_input:
        cmap_list[var_input.index('MLD')] = 'cmo.turbid'
    if 'TurboclineDepth' in var_input:
        cmap_list[var_input.index('TurboclineDepth')] = 'cmo.dense'
    if 'U' in var_input:
        cmap_list[var_input.index('U')] = 'cmo.speed'
    if 'EIVU' in var_input:
        cmap_list[var_input.index('EIVU')] = 'cmo.speed'
    if 'TauX' in var_input:
        cmap_list[var_input.index('TauX')] = 'cmo.speed'
    if 'V' in var_input:
        cmap_list[var_input.index('V')] = 'cmo.speed'
    if 'EIVV' in var_input:
        cmap_list[var_input.index('EIVV')] = 'cmo.speed'
    if 'TauY' in var_input:
        cmap_list[var_input.index('TauY')] = 'cmo.speed'
    if 'W' in var_input:
        cmap_list[var_input.index('W')] = 'cmo.speed'
    if 'EIVW' in var_input:
        cmap_list[var_input.index('EIVW')] = 'cmo.speed'
    if 'VertEddyVisc' in var_input:
        cmap_list[var_input.index('VertEddyVisc')] = 'cmo.speed'
    if 'LatEddyDiff' in var_input:
        cmap_list[var_input.index('LatEddyDiff')] = 'cmo.speed'
    # Calculated variables
    if 'SurfVel' in var_input:
        cmap_list[var_input.index('SurfVel')] = 'cmo.speed'
    if 'T_VertInt' in var_input:
        cmap_list[var_input.index('T_VertInt')] = 'cmo.thermal'
        
    return(cmap_list)
