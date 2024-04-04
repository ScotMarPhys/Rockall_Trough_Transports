from pathlib import Path

local_dir = Path("C:/Users/sa07kb/Projects/Rockall_Trough_Transports/")

one_drive_data_dir = Path("C:/Users/sa07kb/OneDrive - SAMS/data/")

# local paths
local_data_dir = (local_dir/'data')
local_fig_dir = (local_dir/'figures')
raw_data_dir = (local_data_dir/'raw')

# Rockall Trough data
RT_data_path = (one_drive_data_dir/ 'data_RT_mooring')
RT_mooring_data_path = (RT_data_path / 'Rockall_Trough_moorings_time_series')

# suppl data
GEBCO_data_path = (one_drive_data_dir/ 'data_GEBCO')
GLORYS_data_path = (one_drive_data_dir/ 'data_GLORYS')
ERA5_data_path = (one_drive_data_dir/ 'data_ERA5/Rockall_Trough')

# file names
RT_loc_fn = 'Ellet_array_mooring_location.csv'
GEBCO_fn = 'GEBCO_2014_2D_-22.0_55.0_-7.0_59.0.nc'
GLORYS_fn_re = list(sorted(
        (GLORYS_data_path).glob(f"*reanalyis-phy*Vvel.nc") # reanalysis
    ))
GLORYS_fn_an = list(sorted(
        (GLORYS_data_path).glob(f"*anfc-phy-001_024_Vvel.nc") # analysis/forecasts
    ))
ERA5_fn = 'ERA5_tauxy_RT_2014-2022.nc' 