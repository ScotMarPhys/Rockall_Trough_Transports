import pandas as pd
import mat73
import xarray as xr

def mt2dt(matlab_datenum):
        return pd.to_datetime(matlab_datenum-719529, unit='D')
    
def load_glider_mat(glider_data_path,glider_fn):
    df_glider = mat73.loadmat((glider_data_path/glider_fn))

    glider_tgrid = mt2dt(df_glider['t_grid'])
    # df_glider['x2_grid'] # xgrid, m with respect to EB1
    # df_glider['z_grid'] # z_grid, depth in m
    # df_glider['v_grid'] # meridional velocity, m/s
    # lat_grid, lon_grid will give position of idealize sections

    ds_glider = xr.Dataset(
        data_vars=dict(
            vcur=(['x','depth','time'],df_glider['v_grid'])),
        coords=dict(
            time=glider_tgrid,
            x=df_glider['x2_grid'],
            depth=df_glider['z_grid'],
            xT=df_glider['x_grid'],
            lat=('xT',df_glider['lat_grid']),
            lon=('xT',df_glider['lon_grid'])),
        attrs=dict(
            description='Meridional velocities derived from repeated glider section following Frase et al. (2022), https://doi.org/10.1029/2022JC019291'))
    ds_glider = ds_glider.interp(xT=ds_glider.x).drop_vars('xT')
    ds_glider = ds_glider.swap_dims({'x':'lon'})
    ds_glider.vcur.attrs = {'name':'vcur',
                            'long_name':'Meridional Velocities',
                            'units':'m/s'}
    ds_glider.lat.attrs = {'units':'degN',
                           'long_name':'Latitude'}
    ds_glider.lon.attrs = {'units':'degE',
                           'long_name':'Longitude'}
    ds_glider.depth.attrs= {'units':'m',
                            'long_name':'Depth'}
    ds_glider.x.attrs= {'units':'m',
                        'long_name':'Distance from RTEB1, positive towards east'}
    return ds_glider