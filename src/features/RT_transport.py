import dask
import datetime
import gsw
import scipy
import scipy.io
import cmocean as cm
import numpy as np
import pandas as pd
import seawater as sw
import scipy.signal as signal
import palettable.colorbrewer as cb
import xarray as xr
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from xhistogram.xarray import histogram as xhist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

##local functions
import sys; sys.path.append(r'../')
import src.RT_parameters as rtp
import src.set_paths as sps
import src.features.RT_functions as rtf
import src.features.RT_data as rtd
import src.features.matfile_functions as matlab_fct
import src.features.RT_EOF_functions as rt_eof


# Get dx
def get_dx(lon,lat,dim='lon'):
    
    dx = xr.apply_ufunc(
            gsw.distance,
            lon,lat,
            input_core_dims=[[dim],[dim]],
            output_core_dims=[[dim]],
            exclude_dims=set((dim,)), 
    )
    dx1 = xr.concat([dx.shift({dim:1},fill_value=0),dx[-1]],dim=dim)
    dx2 = xr.concat([dx[0],dx.shift({dim:-1},fill_value=0)],dim=dim)
    return (dx1+dx2)/2

def get_dz(z):
    z1 = (z[0]-(z[:2].diff('depth')/2))
    if z1<0:
        dz1 = z.shift(depth=1,fill_value=0)
    else:
        dz1 = z.shift(depth=1,fill_value=(z[0]-(z[:2].diff('depth'))))
    dz2 = z.shift(depth=-1,fill_value=z[-1])
    return (dz2-dz1)/2

def create_horizontal_grid(version):
    # grid = create_horizontal_grid(version)
    # input
    # version = 'v0': Uses grid for EW from Fraser et al., 2022
    # version = 'v1': Uses grid for EW from glider sections
    # output
    # lon_WW,lat_WW,lon_MB,lat_MB,lon_EW,lat_EW,dx_EW,dx_MB,dx_WW,x_WW,x_MB,x_EW
    
    ds_RT_loc=rtd.load_RT_loc()
    
    # create lat and lon grid
    lon_WW = xr.DataArray(np.linspace(ds_RT_loc.lon_RTWS,ds_RT_loc.lon_RTWB,rtp.NX_WW),dims='lon_WW')
    lat_WW = xr.DataArray(np.linspace(ds_RT_loc.lat_RTWS,ds_RT_loc.lat_RTWB,rtp.NX_WW),dims='lon_WW')
    lon_MB = xr.DataArray(np.linspace(ds_RT_loc.lon_RTWB,ds_RT_loc.lon_RTEB,rtp.NX_MB),dims='lon_MB')
    lat_MB = xr.DataArray(np.linspace(ds_RT_loc.lat_RTWB,ds_RT_loc.lat_RTEB,rtp.NX_MB),dims='lon_MB')
    
    if version=='v0':
        lon_EW = xr.DataArray(np.linspace(ds_RT_loc.lon_RTEB,ds_RT_loc.lon_RTES,rtp.NX_EW),dims='lon_EW')
        lat_EW = xr.DataArray(np.linspace(ds_RT_loc.lat_RTEB,ds_RT_loc.lat_RTES,rtp.NX_EW),dims='lon_EW')
    elif version=='v1':
        ds_glider = matlab_fct.load_glider_mat(sps.glider_data_path,sps.glider_fn)
        lon_EW = xr.DataArray(ds_glider.lon.values,dims='lon_EW')
        lat_EW = xr.DataArray(ds_glider.lat.values,dims='lon_EW')

    lon_WW.coords['lon_WW']=lon_WW.values
    lat_WW.coords['lon_WW']=lon_WW.values

    lon_MB.coords['lon_MB']=lon_MB.values
    lat_MB.coords['lon_MB']=lon_MB.values

    lon_EW.coords['lon_EW']=lon_EW.values
    lat_EW.coords['lon_EW']=lon_EW.values
    
    #Get dx
    dx_EW = get_dx(lon_EW,lat_EW,dim='lon_EW')
    dx_MB = xr.apply_ufunc(
                gsw.distance,
                lon_MB,lat_MB,
                input_core_dims=[['lon_MB'],['lon_MB']],
                output_core_dims=[['lon_MB']],
                exclude_dims=set(('lon_MB',)), 
        )
    
    dx_WW = get_dx(lon_WW,lat_WW,dim='lon_WW')
    
    lat_MB = xr.DataArray(lat_MB.values,dims='lon_MB_1')
    
    RT_hor_grid = xr.merge([dx_WW.rename('dx_WW'),
                            dx_MB.rename('dx_MB'),
                            dx_EW.rename('dx_EW'),
                            lat_WW.rename('lat_WW'),
                            lat_MB.rename('lat_MB'),
                            lat_EW.rename('lat_EW')])
    RT_hor_grid.coords['lon_WW']=lon_WW.values
    RT_hor_grid.coords['lon_MB_1']=lon_MB.values
    RT_hor_grid.coords['lon_EW']=lon_EW.values

    return RT_hor_grid

#######################################

# gsw library functions

def gsw_geo_strf_dyn_height(SA,CT,P,P_ref):
    y = xr.apply_ufunc(
        gsw.geo_strf_dyn_height,
        SA,CT,P,P_ref,
        input_core_dims=[['depth','TIME'],['depth','TIME'],['depth'],[]],
        output_core_dims=[['depth','TIME']],
        dask='parallelized')
    return y


def calc_MB_transport(ds_RT,ds_RT_loc,sens_analysis=True,check_plots=True):
    # Get Coriolis freq
    f = gsw.f(np.mean([ds_RT_loc.lat_RTWB,ds_RT_loc.lat_RTEB]))

    # Get reference pressure
    P_ref = ds_RT.where((ds_RT.SG_WEST+ds_RT.SG_EAST).notnull(),drop=True).PRES.max().values

    # # Calculate baroclinic streamfunction
    SF_RTWB = gsw_geo_strf_dyn_height(ds_RT.SG_WEST,ds_RT.TG_WEST,ds_RT.PRES,P_ref)
    SF_RTEB = gsw_geo_strf_dyn_height(ds_RT.SG_EAST,ds_RT.TG_EAST,ds_RT.PRES,P_ref)

    # Calculate geostrophic transport (Sv) in each cell as function of (z,t)
    q_MB = ds_RT.dz*((SF_RTEB - SF_RTWB)/f)
    q_MB.coords['lat_MB']=np.mean([ds_RT_loc.lat_RTWB,ds_RT_loc.lat_RTEB])
    q_MB.coords['lon_MB']=np.mean([ds_RT_loc.lon_RTWB,ds_RT_loc.lon_RTEB])

    # Perform vertical integral for timeseries (Sv)
    Q_MB = q_MB.sum(['depth'],min_count=1)/1e6

    Q_MB.attrs['name']= 'RT_Q_MB'
    Q_MB.attrs['long_name']= 'RT MB Volume Transport'
    Q_MB.attrs['units']='Sv'
    Q_MB.attrs['description']='Mid basin volume transport in Rockall Trough'\
    ' derived from dynamic height difference between the moorings RTEB1 and RTWB1+2'\
    f' with reference pressure {P_ref}'

    if sens_analysis:
        ## Some sensitivity analysis - extra

        # Calculate temporal mean over entire timeseries and broadcast over TIME dimension
        (ds_RT_ave,_) = xr.broadcast(ds_RT.mean('TIME'),ds_RT.TIME)

        # % Do it again isolating CT changes/ keeping SA fixed
        SF_RTWB_SAfix = gsw_geo_strf_dyn_height(ds_RT_ave.SG_WEST,ds_RT.TG_WEST,ds_RT.PRES,P_ref)
        SF_RTEB_SAfix = gsw_geo_strf_dyn_height(ds_RT_ave.SG_EAST,ds_RT.TG_EAST,ds_RT.PRES,P_ref)
        q_MB_SAfix = ds_RT.dz*((SF_RTEB_SAfix - SF_RTWB_SAfix)/f)
        Q_MB_SAfix = q_MB_SAfix.sum('depth',min_count=1)/1e6
        Q_MB_SAfix.attrs['name']= 'RT_Q_MB_SAfix'
        Q_MB_SAfix.attrs['long_name']= 'RT MB Volume Transport SA fixed'
        Q_MB_SAfix.attrs['units']='Sv'
        Q_MB_SAfix.attrs['description']='As Q_MB but holding salinity fixed at temporal mean at all moorings'

        # % Do it again isolating SA changes / keeping CT fixed
        SF_RTWB_CTfix = gsw_geo_strf_dyn_height(ds_RT.SG_WEST,ds_RT_ave.TG_WEST,ds_RT.PRES,P_ref)
        SF_RTEB_CTfix = gsw_geo_strf_dyn_height(ds_RT.SG_EAST,ds_RT_ave.TG_EAST,ds_RT.PRES,P_ref)
        q_MB_CTfix = ds_RT.dz*((SF_RTEB_CTfix - SF_RTWB_CTfix)/f)
        Q_MB_CTfix = q_MB_CTfix.sum('depth',min_count=1)/1e6
        Q_MB_CTfix.attrs['name']= 'RT_Q_MB_CTfix'
        Q_MB_CTfix.attrs['long_name']= 'RT MB Volume Transport CT fixed'
        Q_MB_CTfix.attrs['units']='Sv'
        Q_MB_CTfix.attrs['description']= 'As Q_MB but holding temperature fixed at temporal mean at all moorings'

        # % Do it again holding both fixed
        SF_RTWB_SAfix_CTfix = gsw_geo_strf_dyn_height(ds_RT_ave.SG_WEST,ds_RT_ave.TG_WEST,ds_RT.PRES,P_ref)
        SF_RTEB_SAfix_CTfix = gsw_geo_strf_dyn_height(ds_RT_ave.SG_EAST,ds_RT_ave.TG_EAST,ds_RT.PRES,P_ref)
        q_MB_SAfix_CTfix = ds_RT.dz*((SF_RTEB_SAfix_CTfix - SF_RTWB_SAfix_CTfix)/f)
        Q_MB_SAfix_CTfix = q_MB_SAfix_CTfix.sum('depth',min_count=1)/1e6
        Q_MB_SAfix_CTfix.attrs['name']= 'RT_Q_MB_SAfix_CTfix'
        Q_MB_SAfix_CTfix.attrs['long_name']= 'RT MB Volume Transport SA and CT fixed'
        Q_MB_SAfix_CTfix.attrs['units']='Sv'
        Q_MB_SAfix_CTfix.attrs['description']= 'As Q_MB but holding temperature and salinity fixed at temporal mean at all moorings'

        # % Do it again isolating CT_RTWB        
        q_MB_WB_CTvar = ds_RT.dz*((SF_RTEB_SAfix_CTfix - SF_RTWB_SAfix)/f)
        Q_MB_WB_CTvar = q_MB_WB_CTvar.sum('depth',min_count=1)/1e6
        Q_MB_WB_CTvar.attrs['name']= 'RT_Q_MB_WB_CTvar'
        Q_MB_WB_CTvar.attrs['long_name']= 'RT MB Volume Transport isolating CT changes at RTWB'
        Q_MB_WB_CTvar.attrs['units']='Sv'
        Q_MB_WB_CTvar.attrs['description']= 'As Q_MB but isolating temperature changes at RTWB'\
                                ' and holding temperature at RTEB and salinity at RTWB & RTEB'\
                                ' fixed at temporal mean'

        # % Do it again isolating CT_RTEB
        q_MB_EB_CTvar = ds_RT.dz*((SF_RTEB_SAfix - SF_RTWB_SAfix_CTfix)/f)
        Q_MB_EB_CTvar = q_MB_EB_CTvar.sum('depth',min_count=1)/1e6
        Q_MB_EB_CTvar.attrs['name']= 'RT_Q_MB_EB_CTvar'
        Q_MB_EB_CTvar.attrs['long_name']= 'RT MB Volume Transport isolating CT changes at RTEB'
        Q_MB_EB_CTvar.attrs['units']='Sv'
        Q_MB_EB_CTvar.attrs['description']= 'As Q_MB but isolating temperature changes at RTEB'\
                            ' and holding temperature at RTWB and salinity at RTWB & RTEB'\
                            ' fixed at temporal mean'

    # Merge to dataset
    if sens_analysis:
        RT_Q_MB = xr.merge([Q_MB.rename(Q_MB.attrs['name']),
                        Q_MB_SAfix.rename(Q_MB_SAfix.attrs['name']),
                        Q_MB_CTfix.rename(Q_MB_CTfix.attrs['name']),
                        Q_MB_SAfix_CTfix.rename(Q_MB_SAfix_CTfix.attrs['name']),
                        Q_MB_WB_CTvar.rename(Q_MB_WB_CTvar.attrs['name']),
                        Q_MB_EB_CTvar.rename(Q_MB_EB_CTvar.attrs['name'])]
                          ).drop_vars(['lat_MB','lon_MB'])
    else:
        RT_Q_MB = Q_MB.drop_vars(['lat_MB','lon_MB'])
        
    #check plots
    if check_plots:
        if sens_analysis:
            fig,axs=plt.subplots(3,2,figsize=[12,8])
            ax = axs[0,0]
            ds = RT_Q_MB.RT_Q_MB
            ds.plot(ax=ax)
            ax.set_title(ds.long_name)
            ax = axs[1,0]
            ds = RT_Q_MB.RT_Q_MB_SAfix
            ds.plot(ax=ax)
            ax.set_title(ds.long_name)
            ax = axs[2,0]
            ds = RT_Q_MB.RT_Q_MB_CTfix
            ds.plot(ax=ax)
            ax.set_title(ds.long_name)
            ax = axs[0,1]
            ds = RT_Q_MB.RT_Q_MB_SAfix_CTfix
            ds.plot(ax=ax)
            ax.set_title(ds.long_name)
            ax = axs[1,1]
            ds = RT_Q_MB.RT_Q_MB_WB_CTvar
            ds.plot(ax=ax)
            ax.set_title(ds.long_name)
            ax = axs[2,1]
            ds = RT_Q_MB.RT_Q_MB_EB_CTvar
            ds.plot(ax=ax)
            ax.set_title(ds.long_name)
            for ax in axs.flat:
                ax.grid()
                ax.set_ylim([-10,25])
                ax.set_ylabel('Volume \nTransport (Sv)')
            fig.tight_layout()
        else:
            fig,axs=plt.subplots(1,1,figsize=[6,4])
            RT_Q_MB.plot()
            axs.grid()
            axs.set_title(RT_Q_MB.long_name)
            axs.set_ylim([-10,25])
            axs.set_ylabel('Volume \nTransport (Sv)')
            fig.tight_layout()
    
    return RT_Q_MB,q_MB.rename({'lat_MB':'lat','lon_MB':'lon'})


#########################################################

def calc_MB_3D_sections(ds_RT,ds_RT_loc,RT_hor_grid):
    
    lon_MB = xr.DataArray(RT_hor_grid.lon_MB_1.values,dims='lon_MB')
    
    # Get Coriolis freq
    f = gsw.f(np.mean([ds_RT_loc.lat_RTWB,ds_RT_loc.lat_RTEB]))

    # Get reference pressure
    P_ref = ds_RT.where((ds_RT.SG_WEST+ds_RT.SG_EAST).notnull(),drop=True).PRES.max().values

    SG_MB_grid = xr.concat([ds_RT.SG_WEST.expand_dims('lon_MB'),
          ds_RT.SG_EAST.expand_dims('lon_MB')],dim='lon_MB')
    SG_MB_grid['lon_MB']=[ds_RT_loc.lon_RTWB,ds_RT_loc.lon_RTEB]
    SG_MB_grid.coords['lat_MB']= ('lon_MB',[ds_RT_loc.lat_RTWB,ds_RT_loc.lat_RTEB])

    TG_MB_grid = xr.concat([ds_RT.TG_WEST.expand_dims('lon_MB'),
              ds_RT.TG_EAST.expand_dims('lon_MB')],dim='lon_MB')
    TG_MB_grid['lon_MB']=[ds_RT_loc.lon_RTWB,ds_RT_loc.lon_RTEB]
    TG_MB_grid.coords['lat_MB']= ('lon_MB',[ds_RT_loc.lat_RTWB,ds_RT_loc.lat_RTEB])

    SG_MB_grid = SG_MB_grid.interp(lon_MB=lon_MB)
    TG_MB_grid = TG_MB_grid.interp(lon_MB=lon_MB)

    # Calculate baroclinic streamfunction
    SF_MB_grid = xr.apply_ufunc(
            gsw.geo_strf_dyn_height,
            SG_MB_grid,TG_MB_grid,TG_MB_grid.PRES,P_ref,
            kwargs={'axis':2},
            input_core_dims=[['lon_MB', 'TIME', 'depth'],['lon_MB', 'TIME', 'depth'],['depth'],[]],
            output_core_dims=[['lon_MB', 'TIME', 'depth']],
            dask = 'parallelized')

    # Calculate geostrophic transport (Sv) in each cell as function of (z,t)
    q_MB_grid = ds_RT.dz*(SF_MB_grid.diff('lon_MB')/f)
    lon_MB = xr.DataArray(((SF_MB_grid.lon_MB[:-1].values+SF_MB_grid.lon_MB[1:].values)/2),dims='lon_MB')
    q_MB_grid['lon_MB']=lon_MB
    lat_MB = xr.DataArray(((SF_MB_grid.lat_MB[:-1].values+SF_MB_grid.lat_MB[1:].values)/2),dims='lon_MB')
    q_MB_grid.coords['lat_MB']= lat_MB
    q_MB_grid['dx_MB'] = RT_hor_grid.dx_MB
    v = q_MB_grid/(RT_hor_grid.dx_MB*ds_RT.dz)

    ds_RT_MB_grid = xr.merge([q_MB_grid.rename('q'),
                              v.rename('v'),
                          TG_MB_grid.rename('CT').drop('lat_MB').interp(lon_MB=q_MB_grid.lon_MB),
                          SG_MB_grid.rename('SA').drop('lat_MB').interp(lon_MB=q_MB_grid.lon_MB)]
                            )
    ds_RT_MB_grid.coords['mask'] = q_MB_grid.notnull()
    ds_RT_MB_grid = ds_RT_MB_grid.rename({'lat_MB':'lat','lon_MB':'lon','dx_MB':'dx'})
    
    
    ds_RT_MB_grid.q.attrs={'long_name':f'Volume transport per grid cell',
            'units':'Sv',
            'description':f'Volume transport per grid cell for RT MB'}
    ds_RT_MB_grid.v.attrs = {'long_name':'Across section velocity',
                 'units':'m/s'}
    ds_RT_MB_grid.CT.attrs = {'long_name':'Conservative temperature',
              'description':'conservative temperature TEOS-10',
              'units':'degC'}
    ds_RT_MB_grid.SA.attrs = {'long_name':'Absolute salinity',
            'description':'Absolute salinity TEOS-10',
             'units':'g/kg'}
    
    return ds_RT_MB_grid
#########################

def calc_WW_transport(ds_RT,ds_RT_loc,RT_hor_grid,ds_GEBCO,check_plots=True):
    # Get section bathymetry
    bathy_WW = ds_GEBCO.elevation.interp(
            lon=RT_hor_grid.lon_WW, lat=RT_hor_grid.lat_WW
            ).drop(['lon','lat'])
    bathy_WW.coords['lon_WW']=RT_hor_grid.lon_WW
    
    # use m/s
    ds_RT['v_RTWB']=ds_RT.v_RTWB/1e2
    
    # get meridional velocity
    v_WW = (ds_RT.v_RTWB).rename('v_WW').compute()
    (v_WW,_) = xr.broadcast(v_WW,xr.DataArray(RT_hor_grid.lon_WW, dims="lon_WW"))
    v_WW.coords['lon_WW']=RT_hor_grid.lon_WW
    v_WW.coords['lat_WW']=RT_hor_grid.lat_WW
    v_WW = v_WW.where(RT_hor_grid.lon_WW>=ds_RT_loc.lon_RTWB1)

    # Upper 250 m: Linear decay from WB1 value to 0, western limit is cut off lon
    i_bathy = (RT_hor_grid.lon_WW<ds_RT_loc.lon_RTWB1) & (RT_hor_grid.lon_WW>=rtp.lon_cutoff_WW)
    y = xr.apply_ufunc(
        np.linspace,
        0,ds_RT.v_RTWB.where((ds_RT.depth<rtp.z_cutoff_WW)),sum(i_bathy),
        input_core_dims=[[],['depth','TIME'],[]],
        output_core_dims=[['lon_WW','depth','TIME']],
        dask='parallelized')

    y.coords['lon_WW']= RT_hor_grid.lon_WW[(RT_hor_grid.lon_WW<ds_RT_loc.lon_RTWB1) & (RT_hor_grid.lon_WW>=rtp.lon_cutoff_WW)]

    (v_WW,y)=xr.broadcast(v_WW,y)
    mask = (v_WW.notnull()+y.notnull())
    v_WW = (v_WW.fillna(0)+y.fillna(0)).where(mask)

    # Below 250 m: Linear decay from WB1 value to 0, western limit is bathymetry
    i_bathy =(-1*bathy_WW.where(
        RT_hor_grid.lon_WW<ds_RT_loc.lon_RTWB1)>ds_RT.depth.where(ds_RT.depth>rtp.z_cutoff_WW))

    for idx,bathy in enumerate(i_bathy.sum('lon_WW')):
        if bathy>0:
            y1 = xr.apply_ufunc(
            np.linspace,
            0,ds_RT.v_RTWB.isel(depth=[idx]),bathy,
            input_core_dims=[[],['TIME','depth'],[]],
            output_core_dims=[['lon_WW','TIME','depth']], 
            dask='parallelized')

            y1.coords['lon_WW']= RT_hor_grid.lon_WW.values[i_bathy[:,idx]]
            (_,y1)=xr.broadcast(v_WW,y1)
            mask = (v_WW.notnull()+y1.notnull())
            v_WW = (v_WW.fillna(0)+y1.fillna(0)).where(mask)
    v_WW = v_WW.fillna(0).where(v_WW.depth<-1*bathy_WW)

    # Transport in each cell
    q_WW = RT_hor_grid.dx_WW*ds_RT.dz*(v_WW)
    
    # get CT and SA
    mask = q_WW.notnull()
    qCT = (ds_RT.TG_WEST*q_WW.notnull()).where(mask)
    qSA = (ds_RT.SG_WEST*q_WW.notnull()).where(mask)
    
    q_WW = q_WW.rename('q').to_dataset()
    q_WW['v']=v_WW
    q_WW['CT'] = qCT
    q_WW['SA'] = qSA
    q_WW.coords['dx'] = RT_hor_grid.dx_WW
    q_WW.coords['mask'] = mask
    
    # Add attributes
    q_WW.q.attrs={'long_name':f'Volume transport per grid cell',
            'units':'Sv',
            'description':f'Volume transport per grid cell for RT WW'}
    q_WW.v.attrs = {'long_name':'Across section velocity',
                 'units':'m/s'}
    q_WW.CT.attrs = {'long_name':'Conservative temperature',
              'description':'conservative temperature TEOS-10',
              'units':'degC'}
    q_WW.SA.attrs = {'long_name':'Absolute salinity',
            'description':'Absolute salinity TEOS-10',
             'units':'g/kg'}

    # Integrate for transport timeseries (Sv)
    Q_WW = q_WW.q.sum(['depth','lon_WW'],min_count=1)/1e6 
    Q_WW.coords['mask_WW'] = ds_RT.v_RTWB.isel(depth=50).notnull()
    Q_WW.attrs['name']= 'RT_Q_WW'
    Q_WW.attrs['long_name']= 'RT WW Volume Transport'
    Q_WW.attrs['units']='Sv'
    Q_WW.attrs['description']='Volume transport at western wedge of Rockall Trough'\
    ' derived from moored velocity measurements at RTWB1+2'
    Q_WW = Q_WW.drop_vars(['depth','PRES'])
    
    if check_plots:
        fig,axs=plt.subplots(1,1,figsize=[6,4])
        Q_WW.plot()
        axs.grid()
        axs.set_title(Q_WW.long_name)
        axs.set_ylabel('Volume \nTransport (Sv)')
        fig.tight_layout()

    return Q_WW, q_WW.rename({'lon_WW':'lon','lat_WW':'lat'})
##########################
def calc_EW_F22_transport(ds_RT,ds_RT_loc,RT_hor_grid,ds_GEBCO,ds_GLORYS,check_plots=True):
    
    # Get section bathymetry
    lon_EW = RT_hor_grid.lon_EW
    lat_EW = RT_hor_grid.lat_EW
    bathy_EW = ds_GEBCO.elevation.interp(
                lon=lon_EW, lat=lat_EW
                ).drop(['lon','lat'])
    bathy_EW.coords['lon_EW']=lon_EW
    
    # Get velocity from GLORYS output
    v_GLO_RTADCP = ds_GLORYS.vo.interp(longitude=ds_RT_loc.lon_RTADCP,
                    latitude=ds_RT_loc.lat_RTADCP,
                    time=('TIME',ds_RT.TIME.data),
                    depth=('depth',ds_RT.depth.data)) + rtp.corr_model

    # Duplicate top and bottom GLORYS-ADCP values
    v_GLO_RTADCP = v_GLO_RTADCP.interpolate_na(dim="depth", method="nearest", fill_value="extrapolate")

    # create velocity matrix with mooring velocity at western border
    (v_EW,_) = xr.broadcast(ds_RT.V_EAST/100,xr.DataArray(lon_EW, dims="lon_EW"))
    v_EW.coords['lon_EW'] = RT_hor_grid.lon_EW
    v_EW0 = v_EW.where((ds_RT.depth<rtp.z_cutoff_EW)).where(lon_EW==lon_EW[-1])*0
    v_EW1 = v_EW.where((ds_RT.depth>rtp.z_cutoff_EW))
    v_EW = v_EW.where((ds_RT.depth<rtp.z_cutoff_EW)).where(lon_EW<lon_EW[1])

    # combine both
    mask = (v_EW.notnull()+v_EW1.notnull()+v_EW0.notnull())
    v_EW = (v_EW.fillna(0)+v_EW1.fillna(0)+v_EW0.fillna(0)).where(mask)


    # create velocity matrix with GLORYS-ADCP closest to the position of RTADCP
    (v_EW2,_) = xr.broadcast(v_GLO_RTADCP,xr.DataArray(lon_EW, dims="lon_EW"))
    v_EW2.coords['lon_EW'] = lon_EW
    v_EW2.coords['lat_EW'] = lat_EW
    v_EW3 = v_EW2
    v_EW2 = v_EW2.where((ds_RT.depth<rtp.z_cutoff_EW)
                       ).where(lon_EW==v_EW2.lon_EW.sel(lon_EW=ds_RT_loc.lon_RTADCP,method='nearest').values)

    mask = (v_EW.notnull()+v_EW2.notnull())
    v_EW = (v_EW.fillna(0)+v_EW2.fillna(0)).where(mask)

    # Linear interpolation from mooring velocity to GLORYS-ADCP in depth shallower than 750m 
    v_EW = v_EW.interpolate_na(dim="lon_EW", method="linear")

    # Mask bathy
    v_EW = v_EW.where(v_EW.depth<-1*bathy_EW)

    # Transport in each cell
    q_EW = (RT_hor_grid.dx_EW*ds_RT.dz*(v_EW)).rename('q').to_dataset(
            )
    
    #get tracer
    mask = q_EW.q.notnull()
    qCT = (ds_RT.TG_EAST*q_EW.q.notnull()).where(mask)
    qSA = (ds_RT.SG_EAST*q_EW.q.notnull()).where(mask)
    
    q_EW['v']=v_EW
    q_EW['CT'] = qCT
    q_EW['SA'] = qSA
    q_EW.coords['dx'] = RT_hor_grid.dx_EW
    q_EW.coords['mask'] = mask
    
    # Add attributes
    q_EW.q.attrs={'long_name':f'Volume transport per grid cell',
            'units':'Sv',
            'description':f'Volume transport per grid cell for RT EW F22'}
    q_EW.v.attrs = {'long_name':'Across section velocity',
                 'units':'m/s'}
    q_EW.CT.attrs = {'long_name':'Conservative temperature',
              'description':'conservative temperature TEOS-10',
              'units':'degC'}
    q_EW.SA.attrs = {'long_name':'Absolute salinity',
            'description':'Absolute salinity TEOS-10',
             'units':'g/kg'}
    q_EW = q_EW.drop_vars(['longitude','latitude','time'])

    # Integrate for transport timeseries (Sv)
    Q_EW = q_EW.q.sum(['depth','lon_EW'],min_count=1)/1e6
    Q_EW.coords['mask_EW'] = ds_RT.V_EAST.isel(depth=50).notnull()
    Q_EW.attrs['name']= 'RT_Q_EW'
    Q_EW.attrs['long_name']= 'RT EW Volume Transport'
    Q_EW.attrs['units']='Sv'
    Q_EW.attrs['description']='Volume transport at eastern wedge of Rockall Trough'\
    ' derived from moored velocity measurements at RTEB1 and GLORY12V1 output'
    Q_EW = Q_EW.drop_vars(['depth','PRES'])

    if check_plots:
        fig,axs=plt.subplots(1,1,figsize=[6,4])
        Q_EW.plot()
        axs.grid()
        axs.set_title(Q_EW.long_name)
        axs.set_ylabel('Volume \nTransport (Sv)')
        fig.tight_layout()
    
    return Q_EW, q_EW.rename({'lon_EW':'lon','lat_EW':'lat'})

#########################
def calc_EW_transport(ds_RT,ds_RT_loc,RT_hor_grid,ds_glider,ds_GEBCO,ds_GLORYS,check_plots=True):
    
    lon_EW = RT_hor_grid.lon_EW
    lat_EW = RT_hor_grid.lat_EW    
    bathy_EW = ds_GEBCO.elevation.interp(
                lon=lon_EW, lat=lat_EW
                ).drop(['lon','lat'])
    bathy_EW.coords['lon_EW']=lon_EW
    bathy_EW=bathy_EW.rename({'lon_EW':'lon'})
    
    # Interpolate glider to EB1 depths
    ds_glider['depth']=np.abs(ds_glider.depth)
    ds_glider = ds_glider.interp(depth=('depth',ds_RT.depth.data)).rename({'time':'TIME'})
    
    # Calculate glider EOF
    v_anomaly = ds_glider.vcur.resample(TIME="15D").mean()
    v_anomaly = v_anomaly - v_anomaly.mean('TIME',keep_attrs=True)
    glider_EOF = rt_eof.EOF_func(v_anomaly,n_modes=1,plot_out=False,dim='lon',TIME_dim='TIME')
    
    # Get RTEB1 meridional velocity
    v_RTEB1=(ds_RT.V_EAST/1e2)
    
    #get mean glider section at mooring positions
    glider_locs = ds_glider.vcur.sel(lon=[ds_RT_loc.lon_RTEB,ds_RT_loc.lon_RTADCP],method='nearest'
                                    ).mean('TIME').compute()
    
    # Get Glorys data at ADCP station and apply corretions
    v_GLO_RTADCP = ds_GLORYS.vo.interp(longitude=glider_locs.lon.sel(
                        lon=[ds_RT_loc.lon_RTADCP],method='nearest').data,
                        latitude=ds_RT_loc.lat_RTADCP,
                        time=('time',v_RTEB1.TIME.data),
                        depth=('depth',v_RTEB1.depth.data)) + rtp.corr_model

    # Duplicate top GLORYS-ADCP values
    mask = v_GLO_RTADCP
    mask = (mask.notnull()+mask.shift(depth=-1).notnull())
    v_GLO_RTADCP = v_GLO_RTADCP.interpolate_na(dim="depth", method="nearest", fill_value="extrapolate")
    v_GLO_RTADCP = v_GLO_RTADCP.where(mask).rename({'time':'TIME'}).squeeze(
                    ).drop_vars(['latitude','longitude'])

    # combinde both to one matrix (time,depth,lon)
    ds_y = xr.concat([v_RTEB1,v_GLO_RTADCP], dim="lon")
    ds_y['lon']=glider_locs.lon.data

    # Remove glider sections temporal mean from y
    ds_y = (ds_y-glider_locs).compute()

    # get EOF components at mooring positions as X for linear regression (X'X*alpha=X'y)
    # initial X matrix (mode,lon,depth)
    ds_X = glider_EOF.components().sel(lon=[ds_RT_loc.lon_RTEB,ds_RT_loc.lon_RTADCP],
                                       method='nearest').compute()

    # get alpha & reconstruct velocity fields
    v_rec = rt_eof.rec_v_sec(ds_X,ds_y,glider_EOF,ds_glider.vcur,TIME_dim='TIME')

    zlim = v_rec.depth.where(v_rec.notnull()).max()
    (v_EW,_) = xr.broadcast(v_RTEB1,v_rec.lon)

    
    v_EW = v_rec.fillna(0)+v_EW.where(v_EW.depth>zlim).fillna(0)

    # Mask bathy
    v_EW = v_EW.where(v_EW.depth<-1*bathy_EW)


    # Transport in each cell
    q_EW = (RT_hor_grid.dx_EW.rename({'lon_EW':'lon'})*ds_RT.dz*(v_EW)
           ).rename('q').to_dataset()
    
    #get tracer
    mask = q_EW.q.notnull()
    qCT = (ds_RT.TG_EAST*q_EW.q.notnull()).where(mask)
    qSA = (ds_RT.SG_EAST*q_EW.q.notnull()).where(mask)
    
    q_EW['v']=v_EW
    q_EW['CT'] = qCT
    q_EW['SA'] = qSA
    q_EW.coords['dx'] = RT_hor_grid.dx_EW.rename({'lon_EW':'lon'})
    q_EW.coords['mask'] = mask
    
    # Add attributes
    q_EW.q.attrs={'long_name':f'Volume transport per grid cell',
            'units':'Sv',
            'description':f'Volume transport per grid cell for RT EW F22'}
    q_EW.v.attrs = {'long_name':'Across section velocity',
                 'units':'m/s'}
    q_EW.CT.attrs = {'long_name':'Conservative temperature',
              'description':'conservative temperature TEOS-10',
              'units':'degC'}
    q_EW.SA.attrs = {'long_name':'Absolute salinity',
            'description':'Absolute salinity TEOS-10',
             'units':'g/kg'}
    q_EW = q_EW.drop_vars(['mode','x'])

    # Integrate for transport timeseries (Sv)
    Q_EW = q_EW.q.sum(['depth','lon'],min_count=1)/1e6
    Q_EW.coords['mask_EW'] = ds_RT.V_EAST.isel(depth=50).notnull()
    Q_EW.attrs['name']= 'RT_Q_EW'
    Q_EW.attrs['long_name']= 'RT EW Volume Transport'
    Q_EW.attrs['units']='Sv'
    Q_EW.attrs['description']='Volume transport at eastern wedge of Rockall Trough'\
    ' derived from moored velocity measurements at RTEB1, GLORY12V1 output and glider data'
    Q_EW = Q_EW.drop_vars(['depth','PRES'])
    
    if check_plots:
        fig,axs=plt.subplots(1,1,figsize=[6,4])
        Q_EW.plot()
        axs.grid()
        axs.set_title(Q_EW.long_name)
        axs.set_ylabel('Volume \nTransport (Sv)')
        fig.tight_layout()
   
    return Q_EW,q_EW,ds_glider
    
#########################

def combine_sections_tot_transp(ds_Q_WW,ds_Q_MB,ds_Q_EW):
    ds_Q_tot= (ds_Q_WW.Q.rename('Q_total').fillna(0
                        )+ds_Q_EW.Q.rename('Q_total').fillna(0
                        )+ds_Q_MB.Q.rename('Q_total')
          ).to_dataset()

    ds_Q_tot['Qh_total']= ds_Q_WW.Qh.rename('Qh_total').fillna(0
                            )+ds_Q_EW.Qh.rename('Qh_total').fillna(0
                            )+ds_Q_MB.Qh.rename('Qh_total')
    ds_Q_tot['Qf_total']= ds_Q_WW.Qf.rename('Qf_total').fillna(0
                            )+ds_Q_EW.Qf.rename('Qf_total').fillna(0
                            )+ds_Q_MB.Qf.rename('Qf_total')
    ds_Q_tot = ds_Q_tot
    
    
    desc_str = 'Sum of western, eastern, and mid-basin'
    units = 'Sv'
    name = 'Volume transport'
    ds_Q_tot['Q_total'].attrs = dict(long_name=name, units=units,
                                       description = f'{desc_str} {name}')
    
    units = 'PW'
    name = 'Heat transport'
    ds_Q_tot['Qh_total'].attrs = dict(long_name=name, units=units,
                                       description = f'{desc_str} {name}')

    units ='Sv'
    name = 'Freshwater transport'
    ds_Q_tot['Qf_total'].attrs = dict(long_name=name, units=units,
                                       description = f'{desc_str} {name}')
    
    ds_Q_tot.attrs['description'] = f'{desc_str} volume, heat and freshwater transports'

    return ds_Q_tot
########################

# EKMAN TRANSPORT
# Get normal and tangent to each section
def tau_rot(ds,lat,lon):
    (_,pa) = np.radians(sw.dist(lat[:2],lon[:2],'km'))
    tau_rot = np.cos(pa)*ds.metss + np.sin(pa)*ds.mntss
    return tau_rot

def calc_Ekman_transport(ds_ERA5, RT_hor_grid,ds_RT_loc,check_plots=True):
    lon_MB = xr.DataArray(((RT_hor_grid.lon_MB_1[:-1].values+RT_hor_grid.lon_MB_1[1:].values)/2),dims='lon_MB')
    lat_MB = xr.DataArray(((RT_hor_grid.lat_MB[:-1].values+RT_hor_grid.lat_MB[1:].values)/2),dims='lon_MB')
    
    ds_ERA_MB = ds_ERA5[['metss','mntss']].interp(
        longitude=lon_MB,latitude=lat_MB).drop(['longitude','latitude'])
    ds_ERA_MB.coords['lon_MB']=RT_hor_grid.lon_MB
    
    ds_ERA_WW = ds_ERA5[['metss','mntss']].interp(
        longitude=RT_hor_grid.lon_WW,latitude=RT_hor_grid.lat_WW).drop(['longitude','latitude'])
    ds_ERA_WW.coords['lon_WW']=RT_hor_grid.lon_WW
    
    ds_ERA_EW = ds_ERA5[['metss','mntss']].interp(
        longitude=RT_hor_grid.lon_EW,latitude= RT_hor_grid.lat_EW).drop(['longitude','latitude'])
    ds_ERA_EW.coords['lon_EW']=RT_hor_grid.lon_EW

    # Get tau tangent to section
    tau_para_MB = tau_rot(ds_ERA_MB,lat_MB,lon_MB)
    tau_para_WW = tau_rot(ds_ERA_WW,RT_hor_grid.lat_WW,RT_hor_grid.lon_WW)
    tau_para_EW = tau_rot(ds_ERA_EW,RT_hor_grid.lat_EW,RT_hor_grid.lon_EW)

    # Get Ekman
    f = 2*rtp.Omega*np.sin(np.radians(ds_RT_loc.lat_RTWB))
    V_Ek_WW = -1*tau_para_WW/(f*rtp.rho0)
    V_Ek_MB = -1*tau_para_MB/(f*rtp.rho0)
    V_Ek_EW = -1*tau_para_EW/(f*rtp.rho0)

    # Transport per cell in Sv
    q_Ek_WW = V_Ek_WW*RT_hor_grid.dx_WW/1e6
    q_Ek_MB = V_Ek_MB*RT_hor_grid.dx_MB/1e6
    q_Ek_EW = V_Ek_EW*RT_hor_grid.dx_EW/1e6

    # Integrate transport timeseries
    Q_Ek_WW = q_Ek_WW.sum('lon_WW')
    Q_Ek_WW.attrs['name']= 'Q_Ek_WW'
    Q_Ek_WW.attrs['long_name']= 'RT WW Ekman Transport'
    Q_Ek_WW.attrs['units']='Sv'
    Q_Ek_WW.attrs['description']= 'Ekman Transport at wester wedge of Rockall Trough'\
    ' derived from ERA5 monthly data'

    Q_Ek_MB = q_Ek_MB.sum('lon_MB')
    Q_Ek_MB.attrs['name']= 'Q_Ek_MB'
    Q_Ek_MB.attrs['long_name']= 'RT MB Ekman Transport'
    Q_Ek_MB.attrs['units']='Sv'
    Q_Ek_MB.attrs['description']= 'Ekman Transport across Rockall Trough'\
    ' derived from ERA5 monthly data'

    Q_Ek_EW = q_Ek_EW.sum('lon_EW')
    Q_Ek_EW.attrs['name']= 'Q_Ek_EW'
    Q_Ek_EW.attrs['long_name']= 'RT EW Ekman Transport'
    Q_Ek_EW.attrs['units']='Sv'
    Q_Ek_EW.attrs['description']= 'Ekman Transport at eastern wedge of Rockall Trough'\
    ' derived from ERA5 monthly data'

    # Merge to dataset
    RT_Q_Ek = xr.merge([Q_Ek_WW.rename(Q_Ek_WW.attrs['name']),
                    Q_Ek_MB.rename(Q_Ek_MB.attrs['name']),
                    Q_Ek_EW.rename(Q_Ek_EW.attrs['name'])])
    
    if check_plots:
        (RT_Q_Ek.Q_Ek_WW+RT_Q_Ek.Q_Ek_MB+RT_Q_Ek.Q_Ek_EW).plot(lw=.5,color='k',label='tot',figsize=[6,4])
        RT_Q_Ek.Q_Ek_MB.plot.line(ls='-.',label='MB')
        RT_Q_Ek.Q_Ek_WW.plot.line(ls='-',label='WW')
        RT_Q_Ek.Q_Ek_EW.plot.line(ls='--',label='EW')
        plt.grid()
        plt.legend()
    
    return RT_Q_Ek

#########################
def calc_transports(Q,q,CT,SA,dims,sec_str): 
    
    #calculations
    qh = rtp.rhoCp*q*(CT - rtp.CT_ref)
    qf = -1*q*(SA - rtp.SA_ref)/rtp.SA_ref
    qS = q*SA/rtp.rho0
    
    mask = q.notnull()
    qCT = (CT*q.notnull()).where(mask)
    qSA = (SA*q.notnull()).where(mask)
    
    Qh = qh.sum(dims)/1e15
    Qf = qf.sum(dims)/1e6
    QS = qS.sum(dims)/1e3
    
    #attributes
    q_attrs={'name':f'q',
            'long_name':f'Volume transport per grid cell',
            'units':'Sv',
            'description':f'Volume transport per grid cell for RT {sec_str}'}
    qh_attrs={'name':f'qh',
            'long_name':f'Heat transport per grid cell',
            'units':'PW',
            'description':f'Heat transport per grid cell referenced '\
            f'to temperature of {rtp.CT_ref}degC for RT {sec_str}'}
    qf_attrs = {'name': f'qf',
                'long_name': f'Freshwater transport per grid cell',
                'units':'Sv',
                'description':f'Freshwater transport per grid cell referenced '\
                f'to salinity of {rtp.SA_ref} g/kg for RT {sec_str}'}
    qS_attrs = {'name': f'qS',
                'long_name': f'Salt transport per grid cell',
                'units':'Sv',
                'description':f'Salt transport per grid cell referenced '\
                f'to density time specific heat capacity of of {rtp.rho0}kg m^-3'\
                f'for RT {sec_str}'}
    
    q.attrs =q_attrs
    qh.attrs =qh_attrs
    qf.attrs =qf_attrs
    qS.attrs =qS_attrs
    
    Q.attrs['name']= f'Q'
    Q.attrs['long_name']= f'Volume Transport'
    Q.attrs['units']=Q.units
    Q.attrs['description']=Q.description
    
    Qh.attrs['name']= f'Qh'
    Qh.attrs['long_name']= f'Heat transport'
    Qh.attrs['units']='PW'
    Qh.attrs['description']=f'Heat transport at {sec_str} of Rockall Trough'\
    f' Reference temperature {rtp.CT_ref}degC'
    
    Qf.attrs['name']= f'Qf'
    Qf.attrs['long_name']= f'Freshwater transport'
    Qf.attrs['units']='Sv'
    Qf.attrs['description']=f'Freshwater transport at {sec_str} of Rockall Trough'\
    f' Reference absolute salinity {rtp.SA_ref} (g/kg)'
    
    QS.attrs['name']= f'QS'
    QS.attrs['long_name']= f'Salt transport'
    QS.attrs['units']='Sv'
    QS.attrs['description']=f'Salt transport at {sec_str} of Rockall Trough'\
    f' Reference density {rtp.rho0} (kg/m^3)'
    
    ds_Q = xr.merge([Q.rename(Q.attrs['name']),
                     Qh.rename(Qh.attrs['name']),
                     Qf.rename(Qf.attrs['name']),
                     QS.rename(QS.attrs['name'])])
    ds_q = xr.merge([q.rename(q.attrs['name']),
                     qh.rename(qh.attrs['name']),
                     qf.rename(qf.attrs['name']),
                     qS.rename(qS.attrs['name'])])
    
    if 'mask' in str(Q.coords):
        ds_Q.coords[f'mask_{sec_str[:2]}'] = Q[f'mask_{sec_str[:2]}']
    
    ds_Q.attrs = {'description':f'Volume, heat, freshwater and salt tranport for {sec_str} of Rockall Trough'}
    ds_q.attrs = {'description':f'Volume, heat, freshwater and salt tranport'\
                    f' per grid cell for {sec_str} of Rockall Trough'}
    
    return ds_Q,ds_q

def add_da_to_ds(ds,ds_q,da=['qh','qf']):
    for var in da:
        ds[var]=ds_q[var]
    return ds