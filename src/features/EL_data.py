import xarray as xr
import pandas as pd
import numpy as np
import warnings
import cftime

def get_bodc_files_from_metadata(csv_path, csv_file, instr, year):
    """
    Filters a metadata CSV file to find BODC references that match specific criteria.

    Args:
        csv_path (str): The path to the metadata CSV file.
        csv_file (str): The name of the metadata CSV file.
        instr (str): The desired instrument description (case-sensitive).
                     currently only for ladcp, vmadcp and ctd
        year (int): The desired year from the 'Start date' column.

    Returns:
        list: A list of filenames (e.g., '2205407.nc') that match the criteria.
    """
    if instr == 'ladcp':
        instrument = 'Acoustic doppler current profiler'
        platform = 'research vessel'
        positional_definition = 'Fixed point'
    elif instr == 'vmadcp':
        instrument = 'Acoustic doppler current profiler'
        platform = 'research vessel'
        positional_definition = 'Line trajectory'
    elif instr == 'ctd':
        instrument = 'CTD/STD cast'
        platform = 'research vessel'
        positional_definition = 'Fixed point'
    
    try:
        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_path+csv_file, header=23)
    
        # 1. Standardize and filter the 'BODC Oceanogr Instrument' and 'Platform'
        # The 'Platform' column in the example data is a subset of the 'BODC Oceanogr Instrument' column.
        # We can filter based on these columns to find the desired entries.
        df_filtered = df[df['Instrument'].str.contains(instrument, case=False, na=False)]
        df_filtered = df_filtered[df_filtered['Platform'].str.contains(platform, case=False, na=False)]
        
        # 2. Filter by 'Positional definition'
        df_filtered = df_filtered[df_filtered['Positional definition'] == positional_definition]
        
        # 3. Filter by year
        # First, convert the 'Start date' column to datetime objects
        df_filtered['Start date'] = pd.to_datetime(df_filtered['Start date'], format='%d/%m/%Y')
        
        # Then, extract the year and filter
        df_filtered = df_filtered[df_filtered['Start date'].dt.year == year]
        
        # 4. Get the list of BODC references and append '.nc'
        bodc_references = df_filtered['BODC reference'].astype(str).tolist()
        bodc_references.sort()
        file_list = [f"{csv_path}b{ref}.nc" for ref in bodc_references]
        if not file_list:
            # If no matches are found, raise an error
            raise ValueError(f"The file does not contain any data matching criteria for instrument '{instr}' in year {year}.")
        
        return file_list

    except ValueError as ve:
        # Catch the ValueError we just raised or other potential ValueErrors (e.g. from pd.to_datetime format issues)
        print(f"Filtering error: {ve}")
        # Re-raise the error so the calling code knows there were no results
        raise ve
    except Exception as e:
        # Catch any other unexpected errors during filtering
        print(f"An error occurred during data filtering: {e}")
        raise

def preprocess_ladcp_file(ds,dim='TIME'):
    time_strings = [t.strftime('%Y-%m-%dT%H:%M:%S.%f') for t in ds.TIME.values]
    ds['TIME'] = pd.to_datetime(time_strings).values

    depth_indices = np.arange(1, ds.sizes['MAXZ'] + 1)
    ds = ds.assign_coords(MAXZ=depth_indices)
    return ds

