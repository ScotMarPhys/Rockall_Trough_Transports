# Rockall Trough Volume, Heat and Freshwater Transports
This repository uses moored and glider observations from the Ellet Array to calculate volume, heat and freshwater transports through the northern Rockall Trough.

The Rockall Trough moorings, located at approximately 57° north 11° west, form the eastern boundary of the international OSNAP (Overturning in the Subpolar North Atlantic Programme) array. This observing system has measured the full depth, basin-wide overturning circulation and associated transport of heat and freshwater since 2014. The Rockall Trough component is funded through the Atlantic Climate and Environment Strategic Science (AtlantiS) (NE/Y005589/1) and Climate Linked Atlantic Sector Science (CLASS) (NE/R015953/1), along with NERC grants U.K. OSNAP Decade (NE/T00858X/1 and NE/T008938/1). It consists of three vertical moorings in the boundaries of the subpolar basin, supplemented by glider missions to provide contemporaneous temperature and salinity information. Data included in this release were sampled between July 2014 and July 2022 and are from single point current meters and moored CTD instruments, providing measurements of water velocity, temperature, salinity, and pressure.

# How to contribute
## As External
Contribute by opening an issue.

## As User
Before using the toolbox and making any changes to branches it is essential that any user gains a basic understanding of git and github, which can be found here: https://docs.github.com/en/get-started

# How to use the repository
Please clone or download the repository and download the following data to the data directory. Please make sure to adapt all paths in src/set_paths.py as well as at the beginning of each jupyter notebook.

Rockall Trough data: 
- [Stacked mooring data](https://thredds.sams.ac.uk/thredds/catalog/osnap/osnap.html?dataset=osnap_rtms_tsuv_v1),
- [Gridded moorgin data](https://thredds.sams.ac.uk/thredds/catalog/osnap/osnap.html?dataset=osnap_rtmg_tsuv_v1)

Ancillary data
GLORYS12v2 data should be automatically downloaded if not in the directory set in src/set_paths.py
- [GLORYS12v2](https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description), doi: https://doi.org/10.48670/moi-00021
- [GEBCO version 20141103](https://www.gebco.net/data_and_products/gridded_bathymetry_data/version_20141103/)
- Derive monthly mean horizotnal wind stress for Ekman transport from [ERA5](https://cds.climate.copernicus.eu/datasets/derived-era5-single-levels-daily-statistics?tab=overview)

## Running order to obtain Rockall Trough Transports
Generally, the notebooks are numbered to indicate the order they need to be run to obtain transport estimates and may depend on data outputs from lower order notebooks.

First run notebooks/0.0_Rockall_Trough_moorings_data_prep_2014_2022_v0.ipynb to prepare the mooring data for the transport calculations. 

### Old methodology after Houpert et al. (2020) and Fraser et al. (2022)
To calculate the Rockall Trough Transport wihtout using glider data run notebooks/1.0_Rockall_Trough_transports_2014_2022_v0.ipynb

Note: To obtain Ekman transport estimate you will have to obatin monthly mean values of ERA5 horizontal wind stress. This function is not yet available in this repository. Alternatively, you can uncommand lines in the notebook regarding the Ekman transport, as it is neglectable (Houpert et al., 2020; Fraser et al., 2022)

### New methodology included Glider data after Burmeister et al. (in prep)
To calculate the Rockall Trough Transport using glider data to obtain a better representation of the European Slope Current run notebooks/3.0_Rockall_Trough_transports_2014_2022_v1.ipynb

Note: To obtain Ekman transport estimate you first will have to obatin monthly mean values of ERA5 horizontal wind stress. This function is not yet available in this repository. Alternatively, you can uncommand lines in the notebook regarding the Ekman transport, as it is neglectable (Houpert et al., 2020; Fraser et al., 2022)

### Glider spacial pattern analysis
To perform an analysis of the dominant spatial pattern in the Glider sections use notebooks/2.0_Rockall_Trough_glider_EW_EOF_2020_2023.ipynb or 2.1_Rockall_Trough_glider_EW_HEOF_2020_2023.pdf. As the new methodology automatically performs an EOF analysis of the glider sections, it is not nessecary to run these if you are only interested in the transports.

The other notebooks are used to produce nice figures for paper and presentations and may include additional data products and analysis. They are dependent on data output of lower order notebooks.
