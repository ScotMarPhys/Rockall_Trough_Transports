import pandas as pd
import xarray as xr
from pathlib import Path
import copernicusmarine as cm
import gsw
from pandas.tseries.offsets import DateOffset
import sys; sys.path.append(r'../../')
import src.set_paths as sps
import cftime
import datetime

def load_eap():
    ds_EAP = xr.open_dataset((sps.EAP_path/sps.EAP_fn),decode_times=False)
    datesin = cftime.num2date(ds_EAP.T, ds_EAP.T.units, '360_day')
    datesout = [datetime.datetime(vv.year,vv.month,vv.day) for vv in datesin]
    ds_EAP['T']=datesout
    return ds_EAP
def load_nao():
    df_NAO = pd.read_csv(sps.NAO_path/sps.NAO_fn)

    date = pd.to_datetime(df_NAO.iloc[:, [0,1,2]])

    ds_NAO = xr.Dataset(
        data_vars=dict(
            NAO_index=(['time'],df_NAO['index'])),
        coords=dict(
            time=date),
        attrs=dict(
            description='NOA index after Barnston_1987'))
    return ds_NAO

def change_lon(ds,lon):
    lon_attrs = ds[lon].attrs
    ds = ds.assign_coords({lon:(((ds[lon] + 180) % 360) - 180)}).sortby(lon)
    ds[lon].attrs = lon_attrs
    return ds

def mm_mid_month(ds,tstr):
    units, reference_date = ds[tstr].attrs['units'].split('since')
    
    offset = int(ds.TIME.min('TIME'))
    ds1 = pd.date_range(start=reference_date, periods=ds.sizes[tstr], freq='MS')+DateOffset(months=offset)
    ds2 = pd.date_range(start=reference_date, periods=ds.sizes[tstr], freq='M')+DateOffset(months=offset)
    return ds1+(ds2-ds1)/2-pd.to_timedelta(.5, unit='d')
    
def load_cruise_list():
    df_cruises = pd.read_csv((sps.RT_data_path/'RT_mooring_cruises.csv'))
    df_cruises['date start']= pd.to_datetime(df_cruises['date start'],format='%d/%m/%Y')
    ds_cruises = df_cruises['cruise_id'].to_xarray()
    ds_cruises.coords['TIME']=('index',df_cruises['date start'])
    ds_cruises = ds_cruises.swap_dims({'index':'TIME'}).drop_vars('index')
    return ds_cruises


def load_RT_loc(raw_data_path,RT_mooring_loc):
    moor_ds=pd.read_csv(raw_data_path/RT_mooring_loc)
    moor_ds = moor_ds.set_index(['ID']).to_xarray()
    
    RT_loc = xr.Dataset()
    RT_loc['lon_RTWB1'] = moor_ds.sel(ID='RTWB1').lon.values
    RT_loc['lat_RTWB1'] = moor_ds.sel(ID='RTWB1').lat.values
    RT_loc['lon_RTWB2'] = moor_ds.sel(ID='RTWB2').lon.values
    RT_loc['lat_RTWB2'] = moor_ds.sel(ID='RTWB2').lat.values
    RT_loc['lon_RTWB'] = (RT_loc.lon_RTWB1 + RT_loc.lon_RTWB2)/2
    RT_loc['lat_RTWB'] = (RT_loc.lat_RTWB1 + RT_loc.lat_RTWB2)/2
    RT_loc['lon_RTEB'] = moor_ds.sel(ID='RTEB1').lon.values
    RT_loc['lat_RTEB'] = moor_ds.sel(ID='RTEB1').lat.values
    RT_loc['lon_RTADCP'] = moor_ds.sel(ID='RTADCP').lon.values
    RT_loc['lat_RTADCP'] = moor_ds.sel(ID='RTADCP').lat.values
    RT_loc['lon_RTWS'] = moor_ds.sel(ID='RTWS').lon.values
    RT_loc['lat_RTWS'] = RT_loc.lat_RTWB
    RT_loc['lon_RTES'] = moor_ds.sel(ID='RTES').lon.values
    RT_loc['lat_RTES'] = RT_loc.lat_RTEB
    
    return RT_loc

def make_attr(reg_short,reg_long,end_points,data_used,CT_ref,SA_ref,rho_ref=1027.4):
    attrs_Q = {'name':f'Q_{reg_short}',
                   'long_name':f'{reg_short} Volume Transport',
                    'units':'Sv',
                    'Description':f'{reg_long} volume transport'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'
                  }
    attrs_Qh = {'name': f'Qh_{reg_short}',
                'long_name': f'{reg_short} Heat Flux',
                'units':'PW',
                'Description':f'Heat flux {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'
                   }

    attrs_Qf = {'name': f'Qf_{reg_short}',
                'long_name': f'{reg_short} Freshwater flux',
                'units':'PW',
                'Description':f'Freshwater flux {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'
                }

    attrs_QS = {'name': f'QS_{reg_short}',
                'long_name': f'{reg_short} salt flux',
                'units':'PW',
                'Description':f'salt flux {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'}

    attrs_q = {'name':f'q_{reg_short}',
                   'long_name':f'{reg_short} Volume Transport per cell',
                    'units':'Sv',
                    'Description':f'Volume transport per cell {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'\
                  }
    attrs_qh = {'name': f'qh_{reg_short}',
                'long_name': f'{reg_short} Heat Flux per cell',
                'units':'PW',
                'Description':f'Heat flux per cell {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'\
                f' Reference temperature {CT_ref}degC',}


    attrs_qf = {'name': f'qf_{reg_short}',
                'long_name': f'{reg_short} Freshwater flux per cell',
                'units':'Sv',
                'Description':f'Freshwater flux per cell {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'\
                f' Reference absolute salinity {SA_ref} (g/kg)',
                }

    attrs_qS = {'name': f'qS_{reg_short}',
                'long_name': f'{reg_short} salt flux per cell',
                'units':'Sv',
                'Description':f'salt flux per cell {reg_long}'\
                    f' derived from dynamic height difference between {end_points}'\
                    f' using hydrography from moorings and EN4 and satellite-derive adt'\
                f' Reference density {rho_ref} (kg/m^3)',}
    return attrs_Q,attrs_Qh,attrs_Qf,attrs_QS,attrs_q,attrs_qh,attrs_qf,attrs_qS

def get_OSTIA_sst(moor,tmin,tmax):
    raw_data_path = Path('../data/raw')
    OSTIA_data_path = (raw_data_path/'OSTIA_sst')
    RT_mooring_loc = 'Ellet_array_mooring_location.csv'

    RT_loc = load_RT_loc(raw_data_path,RT_mooring_loc)

    lon = RT_loc[f'lon_{moor}'].values
    lat = RT_loc[f'lat_{moor}'].values

    dlon = 0.2
    dlat = 0.2
    tmin_str= f'{tmin[:4]}{tmin[5:7]}'
    tmax_str= f'{tmax[:4]}{tmax[5:7]}'

    OSTIA_fn = f"METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2_{moor}_{tmin_str}_{tmax_str}.nc"
    
    if (OSTIA_data_path/OSTIA_fn).is_file()==0:
    
        cm.subset(
          dataset_id="METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
          variables=["analysed_sst", "analysis_error", "mask", "sea_ice_fraction"],
          minimum_longitude=lon-dlon,
          maximum_longitude=lon+dlon,
          minimum_latitude=lat-dlat,
          maximum_latitude=lat+dlat,
          start_datetime=tmin,
          end_datetime=tmax,
          output_filename = OSTIA_fn,
          output_directory = OSTIA_data_path
        )
    
    ds = xr.open_dataset((OSTIA_data_path/OSTIA_fn))
    ds = ds.where(ds.mask==1)
    ds = ds.where(ds.sea_ice_fraction==0.0)
    ds = ds.interp(latitude=lat,longitude=lon)
    attrs_sst = ds['analysed_sst'].attrs
    ds['analysed_sst']=ds['analysed_sst']-243.15
    ds.analysed_sst.attrs=attrs_sst
    ds.analysed_sst.attrs['units']='°C'
    return ds[['analysed_sst','analysis_error']]

def load_ARGO_RG_subset(moor,tmin,tmax):
    global_data_path = Path("C:/Users/sa07kb/OneDrive - SAMS/data/")
    raw_data_path = Path('../data/raw')
    Argo_path = global_data_path/'data_ARGO/Roemmich_Gilson_Argo_climatology'
    RT_mooring_loc = 'Ellet_array_mooring_location.csv'
    files = list(sorted(Argo_path.glob(f"RG_ArgoClim_2*")))

    RT_loc = load_RT_loc(raw_data_path,RT_mooring_loc)
    lon = RT_loc[f'lon_{moor}'].values
    lat = RT_loc[f'lat_{moor}'].values

    ds_temp_2019 = xr.open_dataset((Argo_path/'RG_ArgoClim_Temperature_2019.nc.gz'),decode_times=False)
    ds_sal_2019 = xr.open_dataset((Argo_path/'RG_ArgoClim_Salinity_2019.nc.gz'),decode_times=False)
    ds = xr.open_mfdataset(files,decode_times=False)
    
    ds_argo_2019 = xr.merge([ds_temp_2019,ds_sal_2019])
    
    ds_argo_2019 = change_lon(ds_argo_2019,'LONGITUDE')
    ds = change_lon(ds,'LONGITUDE')
    
    ds_argo_2019 = ds_argo_2019.interp(LATITUDE=lat,LONGITUDE=lon)
    ds_argo_2019['TIME']=mm_mid_month(ds_argo_2019,'TIME')
    
    ds = ds.interp(LATITUDE=lat,LONGITUDE=lon)
    ds['TIME']=mm_mid_month(ds,'TIME')
    
    ds_argo = xr.concat([ds_argo_2019,ds],'TIME',data_vars=['ARGO_TEMPERATURE_ANOMALY','ARGO_SALINITY_ANOMALY'])
    ds_argo = ds_argo.where(ds_argo.BATHYMETRY_MASK==0)
    
    ds_argo['TEMP'] = ds_argo.ARGO_TEMPERATURE_ANOMALY+ds_argo.ARGO_TEMPERATURE_MEAN
    ds_argo['PSAL'] = ds_argo.ARGO_SALINITY_ANOMALY+ds_argo.ARGO_SALINITY_MEAN
    
    
    CT_attrs = {'long_name':'Conservative temperature',
              'description':'conservative temperature TEOS-10',
              'units':'degC'}
    SA_attrs = {'long_name':'Absolute salinity',
            'description':'Absolute salinity TEOS-10',
             'units':'g/kg'}

    ds_argo['SA'] = xr.apply_ufunc(gsw.SA_from_SP,
                  ds_argo.PSAL,ds_argo.PRESSURE,ds_argo.LONGITUDE,ds_argo.LATITUDE,
                  dask = 'parallelized',output_dtypes=[float,])
    ds_argo.SA.attrs = SA_attrs

    ds_argo['CT'] = xr.apply_ufunc(gsw.CT_from_t,
                  ds_argo.SA,ds_argo.TEMP,ds_argo.PRESSURE,
                  dask = 'parallelized',output_dtypes=[float,])
    ds_argo.CT.attrs = CT_attrs
    
    return ds_argo[['CT','SA']].sel(TIME=slice(tmin,tmax))