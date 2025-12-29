import dask
import datetime
import gsw
import scipy
import scipy.io
import cmocean as cm
import numpy as np
import pandas as pd
import scipy.signal as signal
import palettable.colorbrewer as cb
import xarray as xr
import pymannkendall as mk
from pathlib import Path
from matplotlib import pyplot as plt
from scipy.signal import butter, filtfilt
from xhistogram.xarray import histogram as xhist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from numba import njit

##local functions
import sys; sys.path.append(r'../')
import src.RT_parameters as rtp
import src.set_paths as sps
import src.features.RT_data as rtd
import src.features.matfile_functions as matlab_fct

# which times are duplicated
def check_for_duplicates(ds,dim,remove=True):
    print(ds[dim].to_index().duplicated().any())
    
    if ds[dim].to_index().duplicated().any():
        ds[dim].diff(dim).plot.line('.',ms='7',label='Time series with duplicates') 
        print(ds[dim].to_index()[ds[dim].to_index().duplicated()])
        
        test = ds.sel(T=~ds[dim].to_index().duplicated())
        test[dim].diff(dim).plot.line('.',ms='4',label='Duplicates removed') 
        plt.legend()
        if remove==True:
            return test

def rename_vars(ds,var_str):
    ds_new = xr.Dataset()
    for var in ds.data_vars:
        ds_new[f'{var}_{var_str}']=ds[var]
        ds_new[f'{var}_{var_str}'].attrs['name'] = f'{var}_{var_str}'
    return ds_new

def normalise_and_predict(x,y,dim):
    # first normalise the variable x
    xnorm = (x - x.mean(dim)) / (x.std(dim));

    # then fit to y
    y_pred = (xnorm * y.std(dim)) + y.mean(dim)
    # y_pred = (xnorm) * (y.std(dim))
    return y_pred

def lin_fit_quick(x,y):
    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)
    model = LinearRegression().fit(x,y)
    r_sq = r2_score(x, y)
    error = mean_squared_error(x,y)
    print(f"coefficient of correlation: {np.sqrt(r_sq)}")
    print(f"mean square errpr: {error}")
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")
    return model

def ddspike(da,std_win,stddy_tol,nloop,dim_x,dim_y,graphics=True):
    mda = da.median(dim_x)
    sda = da.std(dim_x)
    ytol= mda + std_win*sda

    mask_ytol = (np.abs(da.fillna(0))<=ytol)
    spikes_ytol = (np.abs(da.fillna(0))>ytol)
    
    if graphics:
        fig,axs = plt.subplots(nloop+1,1,sharey=True,figsize=[10,(nloop+1)/2])
        ax = axs[0]
        spikes_ytol.sum(dim_y).plot(ax=ax,x=dim_x)
        ax.set_ylabel('')

        print(f'{spikes_ytol.sum().values} exceeds max threshold')

    da = da.where(mask_ytol)


    for i in range(nloop):
        dda_p1=da.fillna(0).shift({dim_x:1},fill_value=0)-da.fillna(0)
        dda_m1=da.fillna(0)-da.fillna(0).shift({dim_x:-1},fill_value=0)

        dda_p1_std = dda_p1.std(dim_x)
        dda_m1_std = dda_m1.std(dim_x)

        m_dda_p1 = (np.abs(dda_p1)/dda_p1_std>=stddy_tol)
        m_dda_m1 = (np.abs(dda_m1)/dda_m1_std>=stddy_tol)

        m_dda = (m_dda_p1.fillna(0)+m_dda_m1.fillna(0))==0

        s_dda = (m_dda_p1.fillna(0)+m_dda_m1.fillna(0))!=0
        if graphics:
            ax = axs[i+1]
            s_dda.sum(dim_y).plot(ax=ax,x=dim_x)
            ax.set_ylabel('')
            
            
            print(f'Loop number {i+1}')
            print(f'{s_dda.sum().values} spikes found')
        da = da.where(m_dda)

    return da

def prep_data_for_gab_filling(da,px,py,day_cutoff):
    px_str = f'p{px}m'
    py_str = f'p{py}m'
    px_lp_str = f'p{px}m_lp'
    py_lp_str = f'p{py}m_lp'
    
    x = da.sel(PRES=px,method = 'nearest')
    y = da.sel(PRES=py,method = 'nearest')

    x = x.where(y.notnull(),drop=True)
    y = y.where(y.notnull(),drop=True)

    v_merge = xr.merge([x.rename(px_str),
          y.rename(py_str)])
    v_merge['v_diff'] = v_merge[px_str]-v_merge[py_str]
    v_merge['month'] = v_merge.TIME.dt.month
    
    #low pass filter
    dt = 0.5 # twice per day, time step of sampling
    fs = 1/dt # sample rate (1/day)
    lowcut = 1/day_cutoff # cut off frequency 1/day_cutoff
    v_merge[px_lp_str] = lazy_butter_lp_filter(v_merge[px_str], lowcut, fs,dim='TIME')
    v_merge[py_lp_str] = lazy_butter_lp_filter(v_merge[py_str], lowcut, fs,dim='TIME')
    return v_merge


def print_stats(y,y_pred):
    error = mean_squared_error(y,y_pred)
    r_sq = r2_score(y,y_pred)
    
    print(f"Mean square error: {error}")
    print(f"coefficient of correlation: {np.sqrt(r_sq)}")
    print(f"Mean y {y.mean().values}, mean y pred {y_pred.mean().values}")
    print(f"Std y {y.std().values}, std y pred {y_pred.std().values}")

def lin_fit_depth(da,px,py,day_cutoff,graphics=True,std_scale=True):
    v_merge = prep_data_for_gab_filling(da, px, py,day_cutoff)
    
    px_str = f'p{px}m'
    py_str = f'p{py}m'
    px_lp_str = f'p{px}m_lp'
    py_lp_str = f'p{py}m_lp'
    
    x = v_merge[px_str]
    y = v_merge[py_str]
    model = lin_fit_quick(x,y)
    v_merge['y_pred'] = model.intercept_.item() + model.coef_.item()*x
    
    x_lp = v_merge[px_lp_str]
    y_lp = v_merge[py_lp_str]
    model_lp = lin_fit_quick(x_lp,y_lp)
    v_merge['y_pred_lp'] = model_lp.intercept_.item() + model_lp.coef_.item()*x_lp
    
    print('Lin regression orig')
    print_stats(y,v_merge['y_pred'])
    print('Lin regression lp')
    print_stats(y_lp,v_merge['y_pred_lp'])
     
    
    if std_scale:
        v_merge['y_scaled'] = normalise_and_predict(v_merge[px_str],v_merge[py_str],'TIME')
        v_merge['y_scaled_lp'] = normalise_and_predict(v_merge[px_lp_str],v_merge[py_lp_str],'TIME')
    
        print(f"Scaling orig")
        print_stats(v_merge[py_str],v_merge['y_scaled'])
        print(f"Scaling lp")
        print_stats(v_merge[py_lp_str],v_merge['y_scaled_lp'])
    
    if graphics:
        plot_figure_gap(v_merge,'y_pred',py_str,'y_pred_lp',py_lp_str,period=[None,None])
        if std_scale:
            plot_figure_gap(v_merge,'y_scaled',py_str,'y_scaled_lp',py_lp_str,period=[None,None]) 
            
    return v_merge

def plot_correlation_stacked(ds_RT_stacked,px,py,period):
    x = ds_RT_stacked.sel(TIME=slice(*period)).VS_EAST_1.sel(ZS_EAST_1_UV=px,method='nearest')
    y = ds_RT_stacked.sel(TIME=slice(*period)).VS_EAST_1.sel(ZS_EAST_1_UV=py,method='nearest')

    x = x.where(y.notnull(),drop=True)
    y = y.where(y.notnull(),drop=True)
    y = y.where(x.notnull(),drop=True)
    x = x.where(x.notnull(),drop=True)

    #low pass filter
    # dt = 0.5 # twice per day, time step of sampling
    # fs = 1/dt # sample rate (1/day)
    # lowcut = 1/day_cutoff # cut off frequency 1/day_cutoff
    # x = rtf.lazy_butter_lp_filter(x, lowcut, fs,dim='TIME')
    # y = rtf.lazy_butter_lp_filter(y, lowcut, fs,dim='TIME')

    y_pred = normalise_and_predict(x,y,'TIME')


    x.sel(TIME=slice(*period)).plot(figsize=(15,4),label=f'x={px}')
    y.sel(TIME=slice(*period)).plot(label=f'y={py}')
    y_pred.sel(TIME=slice(*period)).plot(label=f'y pred',color='k',ls='--')
    plt.legend()

    print(x.mean().values,x.std().values)
    print(y.mean().values,y.std().values)
    print(y_pred.mean().values,y_pred.std().values)
    print(f'x,y: {xr.corr(x,y).values}')
    print(f'RME x, y = {np.sqrt((y-x)**2).mean().values}')
    print(f'y,y_pred: {xr.corr(y,y_pred).values}')
    print(f'RME y, y_pred = {np.sqrt((y-y_pred)**2).mean().values}')


        
def CM_linear_upper_values(var,moor,std_win,stddy_tol,nloop,dim_x,dim_y,graphics):
    
    # if moor=='EB1':
    #     var_i = var.interpolate_na(
    #     dim='PRES',
    #     method="linear",
    #     )
    #     tlim = var_i.TIME.sel(TIME='2020-10-09T12:00:00',method='nearest')
    #     mask_2 = var_i.where((var_i.TIME>tlim)).notnull()
    #     mask_2 = mask_2 + var_i.where((var_i.TIME>tlim)).shift(PRES=-12).notnull()

    
    var_i = var.interpolate_na(
        dim='PRES',
        method="linear",
        fill_value="extrapolate",
    )
    
    if moor=='EB1':
        mask = var_i.where(var_i.PRES<=1800).notnull()
    elif moor=='WB2':
        mask = (var.PRES<1800)&(var.PRES>1020)
    elif moor=='WB1':
        mask = var_i.where((var_i.PRES<=1580)).notnull()
        
    var_i = var_i.where(mask)
    mask = var_i.notnull()
    var_i = ddspike(var_i,std_win,stddy_tol,nloop,dim_x,dim_y,graphics)
    var_i = var_i.interpolate_na(
                dim='TIME',
                method="linear",
            ).where(mask)
    return var_i

def repeat_upper_values(var,dim='PRES'):
    mask = var.bfill(dim=dim).notnull()
    var = var.interpolate_na(
        dim=dim,
        method="nearest",
        fill_value="extrapolate",
    ).where(mask)
    return var

def extr_moored_RT_timeseries(ds_RT,dim_x,dim_y,graphics=True):
    ds_RT['V_EAST'] = CM_linear_upper_values(ds_RT.V_EAST,'EB1',
                         rtp.std_win,rtp.stddy_tol,rtp.nloop,dim_x,dim_y,graphics)
    ds_RT['U_EAST'] = CM_linear_upper_values(ds_RT.U_EAST,'EB1',
                             rtp.std_win,rtp.stddy_tol,rtp.nloop,dim_x,dim_y,graphics)
    ds_RT['V_WEST_1'] = CM_linear_upper_values(ds_RT.V_WEST_1,'WB1',
                             rtp.std_win,rtp.stddy_tol,rtp.nloop,dim_x,dim_y,graphics)
    ds_RT['U_WEST_1'] = CM_linear_upper_values(ds_RT.U_WEST_1,'WB1',
                             rtp.std_win,rtp.stddy_tol,rtp.nloop,dim_x,dim_y,graphics)

    # repeat upper values of hydrography
    ds_RT['TG_EAST'] = repeat_upper_values(ds_RT['TG_EAST'])
    ds_RT['SG_EAST'] = repeat_upper_values(ds_RT['SG_EAST'])
    ds_RT['TG_WEST'] = repeat_upper_values(ds_RT['TG_WEST'])
    ds_RT['SG_WEST'] = repeat_upper_values(ds_RT['SG_WEST'])
    return ds_RT

def merge_RT_WB1_2(ds_RT,mean=False):
    ds_RT['v_RTWB'] = ds_RT.V_WEST_2.where(ds_RT.V_WEST_1.isnull())
    ds_RT['v_mask'] = (ds_RT.v_RTWB.notnull()+ds_RT.V_WEST_1.notnull())
    
    if mean:
        ds_RT['v_RTWB'] = ((ds_RT.V_WEST_1.where(
        ds_RT.V_WEST_2.isnull()).fillna(0
        )+ds_RT.V_WEST_2.fillna(0
        )+ds_RT.V_WEST_2.where(
        ds_RT.V_WEST_1.isnull()).fillna(0
        )+ds_RT.V_WEST_1.fillna(0))/2).where(ds_RT.v_mask==1)
    else:
        ds_RT['v_RTWB'] = (ds_RT.v_RTWB.fillna(0)+ds_RT.V_WEST_1.fillna(0)).where(
            ds_RT.v_mask==1)
    return ds_RT

def ds_rt_swap_vert_dim(ds_RT,dim='PRES'):
    ds_RT_swap = ds_RT.swap_dims({dim:'depth'})
    ds_RT_swap['depth']=abs(ds_RT_swap.depth)
    ds_RT_swap = ds_RT_swap.interp(depth=ds_RT[dim].values)
    return ds_RT_swap

def prep_mooring_data_for_transport_calc(ds_RT,ds_RT_loc):
    
    #remove nan at beginning and end
    ds_RT = ds_RT.where(ds_RT.TG_EAST.sel(PRES=500).notnull().drop_vars('PRES'),drop=True)

    # extrapolate upper values of velocity
    dim_x = 'TIME'
    dim_y = 'PRES'
    graphics = False
    ds_RT = extr_moored_RT_timeseries(ds_RT,dim_x,dim_y,graphics=graphics)

    #linearly interpolate over time gaps in velocity fields
    ds_RT = ds_RT.interpolate_na(dim='TIME')

    # Get z from P
    ds_RT.coords['depth'] = gsw.z_from_p(ds_RT.PRES,np.mean([ds_RT_loc.lat_RTWB, ds_RT_loc.lat_RTEB]))
    ds_RT.depth.attrs = {'name' :'depth',
                         'units' :'m',}
    # Create merged WB1/2 CM
    ds_RT = merge_RT_WB1_2(ds_RT)

    ds_RT = ds_rt_swap_vert_dim(ds_RT)
    return ds_RT

#######################################

def calc_sigma0_grid(ds_RT_grid):

    sigma0_attrs = {'long_name':'Potential density referenced to 0dbar',
                       'description':'Potential density TEOS-10', 
                     'units':'kg/m^3'}

    ds_RT_grid['sigma0'] = xr.apply_ufunc(gsw.sigma0,
                      ds_RT_grid.SA,ds_RT_grid.CT,
                      dask = 'parallelized',output_dtypes=[float,])
    ds_RT_grid.sigma0.attrs = sigma0_attrs
    
    return ds_RT_grid

#######################################

def calc_sigma0(ds):
    
    sigma0_attrs = {'long_name':'Potential density referenced to 0dbar',
                   'description':'Potential density TEOS-10', 
                 'units':'kg/m^3'}
    
    ds['sigma0'] = xr.apply_ufunc(gsw.sigma0,
                  ds.SA,ds.CT,
                  dask = 'parallelized',output_dtypes=[float,])
    ds.sigma0.attrs = sigma0_attrs
    return ds



def calc_SA_CT_sigma0(ds, case='moor'):

    if case=='moor':
        ds = ds.rename({'TIME':'time',
                                'LATITUDE':'lat',
                                'LONGITUDE':'lon',
                                'DEPTH':'depth',
                                'VELO':'vel',
                                'TEMP':'temp',
                                'SAL':'psal'})
        ds['lat']=('lon',ds.lat.values)
        dt='12hr'
        ds['time']=ds.time - pd.Timedelta(dt)
        

    CT_attrs = {'long_name':'Conservative temperature',
              'description':'conservative temperature TEOS-10',
              'units':'degC'}
    SA_attrs = {'long_name':'Absolute salinity',
            'description':'Absolute salinity TEOS-10',
             'units':'g/kg'}
    sigma0_attrs = {'long_name':'Potential density referenced to 0dbar',
                   'description':'Potential density TEOS-10', 
                 'units':'kg/m^3'}
    # calculate pressure
    ds['PRES'] = xr.apply_ufunc(
        gsw.p_from_z,
        -abs(ds.psal.depth),ds.lat,
        dask='parallelized', output_dtypes=[float, ]
        )


    ds['SA'] = xr.apply_ufunc(gsw.SA_from_SP,
                  ds.psal,ds.PRES,ds.lon,ds.lat,
                  dask = 'parallelized',output_dtypes=[float,])
    ds.SA.attrs = SA_attrs

    if case=='moor':
        ds['CT'] = xr.apply_ufunc(gsw.CT_from_t,
                  ds.SA,ds.temp,ds.PRES,
                  dask = 'parallelized',output_dtypes=[float,])
    elif case=='ship':
        ds['CT'] = xr.apply_ufunc(gsw.CT_from_pt,
                  ds.SA,ds.ptmp,
                  dask = 'parallelized',output_dtypes=[float,])
    ds.CT.attrs = CT_attrs

    ds['sigma0'] = xr.apply_ufunc(gsw.sigma0,
                  ds.SA,ds.CT,
                  dask = 'parallelized',output_dtypes=[float,])
    ds.sigma0.attrs = sigma0_attrs
    return ds

# bandpass filter
def __butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def __butter_lowpass(lowcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='low')
    return b, a


def __butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = __butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data,method='gust')
    return y

def __butter_lowpass_filter(data, lowcut, fs, order=4):
    b, a = __butter_lowpass(lowcut,  fs, order=order)
    y = filtfilt(b, a, data,method='gust')
    return y

def lazy_butter_bp_filter(data, lowcut, highcut, fs,dim='time_counter'):
    y = xr.apply_ufunc(
        __butter_bandpass_filter,
        data.chunk({dim: -1}), lowcut, highcut, fs,
        input_core_dims=[[dim],[],[],[]],
        output_core_dims=[[dim]],
        dask='parallelized')
    return y


def lazy_butter_lp_filter(data, lowcut, fs,dim='time_counter'):
    y = xr.apply_ufunc(
        __butter_lowpass_filter,
        data.chunk({dim: -1}), lowcut, fs,
        input_core_dims=[[dim],[],[]],
        output_core_dims=[[dim]],
        dask='parallelized')
    if 'long_name' in y.attrs:
        y.attrs['long_name'] = f'{1/lowcut} days low pass filtered {y.long_name}'
    else:
        y.attrs['long_name'] = f'{1/lowcut} days low pass filtered'
    if 'description' in y.attrs:
        y.attrs['description'] = f'{1/lowcut} days low pass filtered {y.description}'
    else:
        y.attrs['description'] = f'{1/lowcut} days low pass filtered'
    return y

def lp_filt_loop(ds,lowcut,fs,dim,check_plots=False):
    ds_vars = ds.data_vars
    var_count = 0
    for var in ds_vars:
        if ('_lp' not in var) and (ds[var].size>1):
            with xr.set_options(keep_attrs=True):
                ds[f'{var}_lp'] = lazy_butter_lp_filter(ds[var],lowcut,fs,dim=dim)
                var_count +=1
        elif '_lp' in var:
            print(f'{var} already in Dataset, will be overwritten')
        elif ds[var].size==1:
            print(f'{var} has length {ds[var].size}, no filter applied')
    
    if check_plots:
        fig,axs = plt.subplots(var_count,1,figsize=[var_count*3,8],sharex=True)
        i=0
        for var in ds_vars:
            if '_lp' not in var:
                ax=axs[i]
                ds[var].plot(ax=ax)
                ds[f'{var}_lp'].plot(ax=ax)
                i+=1
                ax.set_xlabel('')
                ax.set_ylabel(ds[var].units)
                ax.set_title(ds[var].long_name)
                ax.grid()
        plt.tight_layout()
    return ds

##################################################
def xcorr_norm(x,y,dim):
#         """
#         Perform Cross-Correlation on x and y
#         x    : 1st signal
#         y    : 2nd signal

#         returns
#         corr : coefficients of correlation
#         """
        # First normalise the variable
        xnorm = (x - x.mean(dim)) / (x.std(dim)*len(x));
        ynorm = (y - y.mean(dim)) / (y.std(dim))

        corr = signal.correlate(xnorm, ynorm, mode="full")
        lags = signal.correlation_lags(len(xnorm), len(ynorm), mode="full")
        return corr,lags

    ######################################################
def decorrelation(x,y,dim,doplot,precision=2,print_text=False):
    # %% Compute auto-correlation of a variable; determine the weighted integral scale; 
    # % compute degrees of freedom in data series
    # % Compute the integral time scale and degrees of freedom in a timeseries
    # % Emery, W. J. and R. E. Thomson, 1997: Data analysis methods in physical
    # % oceanography. Pergamon, 634 pp. (see p263,265,379-380) define the
    # % integral time scale from C(0) out to the first zero crossing. Here we
    # % define the integral time scale as twice this value.
    # % Integral time scale = 2xsum(C(tau)dt) from C(0) to first zero crossing.
    # % If the autocorrelation decays linearly then the area under this triangle
    # % is 0.5 * distance to zero crossing. So twice this integral is equal to the 
    # % time to the zero crossing.
    # % If the correlation decays fast initially but more slowly later the zero
    # % crossing suggests a longer time than the sum which is really just a
    # % weighted estimate and in this case gives less weight to a long tail with
    # % low correlation.
    # %
    # %
    # % USAGE : dcl,dof = decorrelation(x,y,dim,doplot)
    # INPUT
    # % x: xarray dataarray, is first variable and normalised prior to computing the xcorrelation: xnorm = (x - x.mean(dim)) / (x.std(dim)*len(x));
    # % y: xarray, dataarray, is second and is normalised prior to computing the xcorrelation:(y - y.mean(dim)) / (y.std(dim))
    #     for autorcorrelation x and y are the same
    #   dim: str, name of dimension along with computation should take 
    # % Diagnostic doplot = 1/0 for diagnostic plots.
    #
    # OUTPUT
    # % dcl is the decorrelation length scale in the units of x
    # % dof is the number of degrees of freedom in x. Calculated by length(x) /
    # % dcl.
    # % Stuart Cunningham, July 2017
    # Adapted for python xarray, Kristin Burmeister, 2023

    C,lags = xcorr_norm(x,y,dim) # compute normalised correlation coefficient

    # find lag value of first negative crossing
    idx2 = int((len(C)+1)/2)
    if precision>0:
        idx = next(x[0] for x in enumerate(C[idx2:]) if np.round(x[1],precision) <=0)
    else:
        idx = next(x[0] for x in enumerate(C[idx2:]) if x[1] <=0)
    
    if idx>0:
        # find index of first negative&positive crossing
        Imin,Imax= idx2-idx-1,idx2+idx

        # Integrate correlation between first neg and first pos crossing. This is the decorrelation length
        dcl=np.trapz(C[Imin:Imax],lags[Imin:Imax])

        # Degrees of freedom = length of y / dcl
        dof= len(y)/ dcl
    elif idx==0:
        dcl=np.nan
        dof=len(y)
    if print_text:
        display(f'x has {len(y)} data cycles')
        display(f'Integral time scale (days) = {dcl}')
        display(f' : Degrees of freedom = {dof}')

    if doplot:
        
        plt.plot(lags[Imin:Imax],C[Imin:Imax]*0)
        plt.plot(lags[Imin:Imax],C[Imin:Imax])
        plt.vlines([lags[Imin],lags[Imax]],-0.2,1)
        plt.xlabel('Lags');
        plt.ylabel('Normalised Correlation Coefficient');
        plt.title('Normalised auto-correlation of ynorm')

    return dcl,dof
    
###########################################################################
def harmonic_cycle(t, T=1, phi=0):
    """Create harmonic cycles."""
    return np.sin(2 * np.pi / T * (t + phi))

def normalize(x=None, y=None):
    return y / np.linalg.norm(y) / (x.max() - x.min()) ** 0.5

def harmonic_proj(t=None, T=None, dt=None, signal=None, dims='time'):
    #     harmonic_mode = (
    #         normalize(t, harmonic_cycle(t, T=T, phi=0))
    #         + 1j * normalize(t, harmonic_cycle(t, T=T, phi=T / 4.0))
    #     ) / (2 ** 0.5)
    harmonic_mode = normalize(
        t,
        harmonic_cycle(t, T=T, phi=0) + 1j * harmonic_cycle(t, T=T, phi=T / 4.0)
    )
    return (signal * xr.DataArray(harmonic_mode, dims=dims)* dt**0.5).sum(dims)

def harmonic_phase(t=None, T=None, dt=None, signal=None, dims='time'):
    proj = harmonic_proj(t=t, T=T, dt=dt, signal=signal, dims=dims)
    phi = np.arctan2(np.imag(proj), np.real(proj)) * T / np.pi / 2
    phi.attrs['name'] = 'Phase'
    phi.attrs['units'] = 'days'
    return phi

def harmonic_amplitude(t=None, T=None, dt=None, signal=None, dims='time'):
    proj = harmonic_proj(t=t, T=T, dt=dt, signal=signal, dims=dims)
    return 2 * np.abs(proj)

def harm_fit(s_n,dims='time'):
    time_ordinal = np.array([pd.to_datetime(x).toordinal() for x in s_n[dims].values])
    time_ordinal -= time_ordinal[0]
    s_n.coords['time_ordinal']=([dims],time_ordinal)
    dt = time_ordinal[1]-time_ordinal[0]

    ah_pha = harmonic_phase(s_n.time_ordinal, 365,dt, s_n, dims=dims)
    ah_amp = harmonic_amplitude(s_n.time_ordinal, 365,dt, s_n, dims=dims)
    sh_pha = harmonic_phase(time_ordinal, 365 / 2.0,dt, s_n, dims=dims)
    sh_amp = harmonic_amplitude(time_ordinal, 365 / 2.0,dt, s_n, dims=dims)
    return xr.merge((ah_pha.rename('ah_pha'), ah_amp.rename('ah_amp'), sh_pha.rename('sh_pha'), sh_amp.rename('sh_amp')))

def reconstr_ts(harmo_JRA_obs,time,T,dims='time'):
    t = np.array([pd.to_datetime(x).toordinal() for x in time])
    t -= t[0]
    t=xr.DataArray(data=t,dims=dims)
    if T==365:
        amp,phi = harmo_JRA_obs.ah_amp,harmo_JRA_obs.ah_pha
    elif T==365/2:
        amp,phi = harmo_JRA_obs.sh_amp,harmo_JRA_obs.sh_pha
    
    JRA_rec = amp * harmonic_cycle(
        t, T=T, phi=phi)
    JRA_rec.coords[dims]=time
    return JRA_rec

def detrend_data(ds,dim,perform_HF=False,plot=False):

    ds_orig = ds
    date_1yr = '%4d-%02d-%02d'%((ds[dim][0].dt.year+1).values,ds[dim][0].dt.month.values,ds[dim][0].dt.day.values)
    date_1yr = np.datetime64(date_1yr,'ns')
    date_1yr = xr.DataArray(np.array([ds[dim][0].values, date_1yr], dtype='datetime64'),dims=dim)
    date_1yr.coords[dim]=date_1yr

    # remove annual and semiannual harmonic
    if perform_HF:
        ds_HF = harm_fit(ds,dims=dim)
        ds_AH = reconstr_ts(ds_HF,ds[dim].values,365,dims=dim)
        ds_SH = reconstr_ts(ds_HF,ds[dim].values,365/2,dims=dim)
        ds = ds-ds_AH-ds_SH

    p = ds.polyfit(dim=dim, deg=1)
    fit = xr.polyval(ds[dim], p.polyfit_coefficients)
    # fit = fit-2*fit[0]
    trend_1yr = xr.polyval(date_1yr, p.polyfit_coefficients)
    trend_1yr = trend_1yr.diff(dim).values

    # ## SIGNIFICANCE TEST

    # calculate degrees of freedom (dof)
    _,dof = decorrelation(ds.fillna(0),ds.fillna(0),'TIME',0);

    #find T critical value
    alpha =  1-0.05/2; # two-sided t-test
    t_crit = scipy.stats.t.ppf(q=alpha,df=dof)

    yresid = ds-fit
    yresid = (yresid**2).sum(dim); # sum of the squared residuals
    ytotal = (len(ds)-1) * ds.var(dim); # n * variance

    # adjusted coefficient of determination
    rsq_y = 1 - yresid/ytotal*(len(ds)-1)/(len(ds)-2);

    # calculate t-value
    t_val = np.sqrt(abs((rsq_y*(dof-2))/(1-rsq_y)));

    # significance test, t-test, 95% interval, H_0: R=0.0
    if abs(t_crit)<abs(t_val):
        trend_significant = True
    else:
        trend_significant = False

    slope=(p.polyfit_coefficients.sel(degree=1).values)
    intc=(p.polyfit_coefficients.sel(degree=0).values)

    ds_detrend = xr.Dataset(data_vars=dict(
            ds_orig=([dim],ds_orig.values),
            reg_slope=(p.polyfit_coefficients.sel(degree=1).values),
            reg_intc=(p.polyfit_coefficients.sel(degree=0).values),
            trend_1yr=(trend_1yr),
            ds_trend=([dim],(fit).data),
            ds_dtrnd=([dim],(ds-(fit-fit.mean(dim))).values),
            trend_sign=(trend_significant),
        ),
        coords={dim:ds[dim].values},
        )
    ds_detrend.ds_orig.attrs=ds_orig.attrs
    ds_detrend.ds_dtrnd.attrs=ds_orig.attrs
    ds_detrend.ds_trend.attrs=ds_orig.attrs
    ds_detrend.ds_trend.attrs['description']='trend fitted to ds_orig'
    ds_detrend.ds_trend.attrs['description']='(ds_trend-ds_trend.mean()) subtracted from ds_orig'

    if plot:
        fs=8
        font = {'weight' : 'normal',
                'size'   : fs}
        plt.rc('font', **font)
        
        fig,axs = plt.subplots(1,1,figsize=[5,1.5])
        ax=axs
        ds_detrend.ds_orig.plot.line(ax=ax,lw=0.5,label='orig')
        ds_detrend.ds_dtrnd.plot.line(ax=ax,lw=0.5,label='detrended')
        (fit-fit[0]+intc).plot(ax=ax,label='trend')
        ax.hlines(ds.mean(dim),ds[dim][0],ds[dim][-1])
        left,bottom=0.95,0.05
        if abs(trend_1yr)<1e-2:
            ax.text(left,bottom,'trend is %3.2f 10$^{-2}$ %s/yr, signifcance - %s'%(trend_1yr*1e2,ds_orig.units,trend_significant),ha='right',transform=ax.transAxes,fontsize=fs)
        else:
            ax.text(left,bottom,'trend is %3.2f %s/yr, signifcance - %s'%(trend_1yr,ds_orig.units,trend_significant),ha='right',transform=ax.transAxes,fontsize=fs)
        plt.legend(ncol=3,loc='upper right',fontsize=fs)
        ax.set_ylabel('%s [%s]'%(ds_orig.long_name,ds_orig.units))

    return fit, slope, intc, trend_significant

##################################################

## Standard error
def std_error(da,dim='TIME'):
    da_std  = da.std()
    _,dof = decorrelation(da.fillna(0),da.fillna(0),dim,0);
    return da_std/np.sqrt(dof);

def std_error_loop(ds):
    ds_vars = ds.data_vars
    var_count = 0
    for var in ds_vars:
        if ('_SE' not in var) and (ds[var].size>1):
            with xr.set_options(keep_attrs=True):
                ds[f'{var}_SE'] = (std_error(ds[var])).compute()
                
                ds[f'{var}_SE'].attrs = {'name':f'{var}_SE',
                        'long_name':f'Standard error of {var}',
                         'units':ds[var].units}
                var_count +=1    
    return ds

################

def xcorr_norm_optimized_access(x,y,dim):
    """
    Perform Cross-Correlation on x and y, ensuring efficient data access.
    """
    # Normalise variables using xarray methods
    xnorm = (x - x.mean(dim)) / (x.std(dim) * len(x[dim])) # use len(x[dim]) for consistency
    ynorm = (y - y.mean(dim)) / y.std(dim)

    # Pass the underlying NumPy arrays to the optimized Scipy function
    # This avoids potential xarray wrappers within the inner loop of signal.correlate
    corr = signal.correlate(xnorm.values, ynorm.values, mode="full")
    lags = signal.correlation_lags(len(xnorm.values), len(ynorm.values), mode="full")
    
    # Return as xarray DataArrays if you need them downstream (which you do, for np.trapz in decorrelation)
    # Recreate the xarray DataArray structure for the correlation output
    corr_da = xr.DataArray(corr, coords={'lags': lags}, dims=['lags'])
    lags_da = xr.DataArray(lags, coords={'lags': lags}, dims=['lags'])

    return corr_da, lags_da

# Add numba optimization to the decorrelation function where possible
@njit
def integrate_correlation(C, lags, Imin, Imax, len_y):
    # This part can be safely jitted
    dcl = np.trapz(C[Imin:Imax], lags[Imin:Imax])
    dof = len_y / dcl
    return dcl, dof

def decorrelation_optimized(x,y,dim,doplot,precision=2,print_text=False):
    # The xcorr_norm function likely cannot be jitted if it uses xarray objects internally
    C,lags = xcorr_norm(x,y,dim) 
    # ... (rest of the index finding logic) ...
    idx2 = int((len(C)+1)/2)
    # ... (find idx) ...
    
    if idx>0:
        Imin,Imax= idx2-idx-1,idx2+idx
        # Use the jitted function for the integration step
        dcl, dof = integrate_correlation(C.values, lags.values, Imin, Imax, len(y))
    # ... (rest of the logic for idx==0, printing, plotting) ...
    return dcl,dof


## Standard error (Refined)
def std_error_refined(da, dim='TIME'):
    da_std = da.std(dim=dim)
    # Perform fillna once here before passing to decorrelation
    da_filled = da.fillna(0)
    _, dof = decorrelation(da_filled, da_filled, dim, 0)
    return da_std / np.sqrt(dof)

def std_error_loop_optimized(ds):
    """
    Uses xarray/dask deferred computation for speed.
    """
    ds_vars = ds.data_vars
    new_vars_to_add = {}

    for var in ds_vars:
        if ('_SE' not in var.upper()) and (ds[var].size > 1):
            # The calculation is assigned, but NOT computed yet
            da_se = std_error_refined(ds[var]) # Use the refined function

            # Store the operation and metadata
            new_vars_to_add[f'{var}_SE'] = da_se.assign_attrs({
                'name':f'{var}_SE',
                'long_name':f'Standard error of {var}',
                 'units':ds[var].attrs.get('units', 'unknown')
            })

    # Execute all computations simultaneously and merge back
    if not new_vars_to_add:
        return ds

    # Create a new Dataset from the scheduled tasks and compute once
    ds_se = xr.Dataset(new_vars_to_add).compute() 
    
    # Merge the new standard error variables into the original dataset
    ds = ds.merge(ds_se,compat='no_conflicts')
    
    return ds


def generate_stats_table(ds, ds_pro, filename_tex, region_str):
    """
    Calculates statistics, prints to console, and saves a LaTeX table.

    Inputs:
        ds           : Xarray Dataset containing 'Qf' (full calc) and 'Qh' (full calc)
        ds_pro       : Xarray Dataset containing 'Qf' (profile calc) and 'Qh' (profile calc)
        filename_tex : String specifying the output .tex filename (e.g., 'stats_MB.tex')
        region_str   : String label for the region (e.g., 'Mid-Basin (MB)')
    """

    # Extract the relevant integrated transport DataArrays from the Datasets
    # Use .values to work with underlying numpy arrays or keep as DataArrays for alignment
    Qf = ds['Qf']
    Qf_pro = ds_pro['Qf']
    Qh = ds['Qh']
    Qh_pro = ds_pro['Qh']

    # Ensure alignment (xarray automatically aligns on coordinates, but this is good practice)
    Qf, Qf_pro = xr.align(Qf, Qf_pro)
    Qh, Qh_pro = xr.align(Qh, Qh_pro)

    # Calculate differences (for MBE and RMSE)
    Qf_diff = Qf - Qf_pro
    Qh_diff = Qh - Qh_pro

    # Define the scaling factor (10e-2 = 0.01)
    scale_factor = 1e-2
    
    # Define a helper function for scaling and calculating stats
    def calculate_scaled_stats(data_full, data_pro, data_diff, scale):
        stats = {}
        # Use skipna=True (equivalent to 'omitnan' in MATLAB)
        stats['mean_full'] = data_full.mean(skipna=True).item() / scale
        stats['mean_pro']  = data_pro.mean(skipna=True).item() / scale
        stats['std_full']  = data_full.std(skipna=True).item() / scale
        stats['std_pro']   = data_pro.std(skipna=True).item() / scale
        stats['mbe']       = data_diff.mean(skipna=True).item() / scale
        # RMSE manual calculation using numpy functions on the underlying data
        stats['rmse']      = np.sqrt(((data_full - data_pro)**2).mean(skipna=True).item()) / scale
        return stats

    # Calculate all stats in one go
    stats_Qf = calculate_scaled_stats(Qf, Qf_pro, Qf_diff, scale_factor)
    stats_Qh = calculate_scaled_stats(Qh, Qh_pro, Qh_diff, scale_factor)


    ### 2. Print stats to Python console (using f-strings for formatting)

    print(f"\nStatistics Summary for Region: {region_str}")
    print("####################################################")
    # Using format specifiers in f-strings: {var: >width.precisionf}
    print(f"{'Metric':<20} {'Qf (10e-2 Sv)':>15} {'Qh (10e-2 PW)':>15}")
    print("####################################################")

    print(f"{'Mean full':<20} {stats_Qf['mean_full']:>15.4f} {stats_Qh['mean_full']:>15.4f}")
    print(f"{'Mean profile':<20} {stats_Qf['mean_pro']:>15.4f} {stats_Qh['mean_pro']:>15.4f}")
    print(f"{'Mean Bias':<20} {stats_Qf['mbe']:>15.4f} {stats_Qh['mbe']:>15.4f}")
    print("----------------------------------------------------")
    print(f"{'Std Dev full':<20} {stats_Qf['std_full']:>15.4f} {stats_Qh['std_full']:>15.4f}")
    print(f"{'Std Dev profile':<20} {stats_Qf['std_pro']:>15.4f} {stats_Qh['std_pro']:>15.4f}")
    print(f"{'RMSE':<20} {stats_Qf['rmse']:>15.4f} {stats_Qh['rmse']:>15.4f}")
    print("####################################################\n")


    ### 3. Print as LaTeX table to the specified file

    # Python uses 'with open(...) as f:' context manager which safely handles file closing
    try:
        with open(filename_tex, 'w') as f: 
            # Write LaTeX table preamble
            f.write('\\begin{table}[h!]\n')
            f.write('\\centering\n')
            f.write(f'\\caption{{Summary of Qf and Qh Statistics for {region_str} (Units are in $10^{{-2}}$ Sv and $10^{{-2}}$ PW)}}\n')
            f.write(f'\\label{{tab:stats_summary_{region_str.replace(" ", "_").replace("-", "")}}}\n')
            f.write('\\begin{tabular}{|l|c|c|}\n')
            f.write('\\hline\n')

            # Write LaTeX table header row
            f.write('Metric & Qf (10e-2 Sv) & Qh (10e-2 PW) \\\\\n')
            f.write('\\hline\n')
            f.write('\\hline\n')

            # Write data rows (%.4f format specifiers)
            f.write(f"Mean full & {stats_Qf['mean_full']:.4f} & {stats_Qh['mean_full']:.4f} \\\\\n")
            f.write(f"Mean profile & {stats_Qf['mean_pro']:.4f} & {stats_Qh['mean_pro']:.4f} \\\\\n")
            f.write(f"Mean Bias & {stats_Qf['mbe']:.4f} & {stats_Qh['mbe']:.4f} \\\\\n")
            f.write('\\hline\n')
            f.write(f"Std Dev full & {stats_Qf['std_full']:.4f} & {stats_Qh['std_full']:.4f} \\\\\n")
            f.write(f"Std Dev profile & {stats_Qf['std_pro']:.4f} & {stats_Qh['std_pro']:.4f} \\\\\n")
            f.write(f"RMSE & {stats_Qf['rmse']:.4f} & {stats_Qh['rmse']:.4f} \\\\\n")
            f.write('\\hline\n')

            # Write LaTeX table postamble
            f.write('\\end{tabular}\n')
            f.write('\\end{table}\n')
        
        print(f"Successfully saved LaTeX table to: {filename_tex}")

    except IOError as e:
        print(f"Error writing to file {filename_tex}: {e}")

## Updated trends ###############################################
def calculate_dt_days(dataset_time_coord):
    """
    Safely calculates the time step duration in days from an xarray time coordinate.
    """
    if len(dataset_time_coord) < 2:
        raise ValueError("Time coordinate must have at least 2 steps.")

    # Get the precise difference as a pandas Timedelta
    dt_timedelta = pd.to_timedelta(np.diff(dataset_time_coord.values)[0])
    
    # Convert the timedelta to a numeric value representing total days
    dt_days = dt_timedelta.total_seconds() / (60 * 60 * 24)
    
    return dt_days

def calculate_and_format_trends(dataset):
    """
    Calculates Hamed and Rao modified MK trends for variables in an xarray dataset,
    excluding QS variables.
    Formats the results into structured pandas DataFrames with separate columns for
    Trend and P-value, and saves them as a specific LaTeX table format.

    Args:
        dataset (xr.Dataset): The input xarray dataset containing variables like Q_MB, Q_lp_MB etc.
    """
    results_list = []

    # Scale trend to 10 years:
    # Assuming 'TIME' or 'time' exists and is correct in the dataset
    time_coord_name = 'time' if 'time' in dataset.coords else 'TIME'

    dt_days = calculate_dt_days(dataset[time_coord_name])
    scale_factor_10yr = (10 * 365.25) / dt_days
    
    print(f"Detected time step duration (dt): {dt_days} days")
    print(f"10-year scaling factor: {scale_factor_10yr:.2f} time steps per 10 years")
    print("-" * 50)

    # Iterate through variables and perform the test
    for var_name in dataset.data_vars:
        # Skip variables that start with QS
        if var_name.startswith('QS'):
            continue

        data = dataset[var_name]
        
        if data.size > 1:
            try:
                series_data = data.to_pandas()
                result = mk.hamed_rao_modification_test(series_data)
                
                # Scale the slope to a 10-year trend value
                scaled_slope = result.slope * scale_factor_10yr
                
                results_list.append({
                    'Variable': var_name,
                    'Trend_Value': scaled_slope, # Renamed to avoid confusion with formatted string
                    'P_value_Numeric': result.p # Renamed to store numeric value for comparison
                })

            except Exception as e:
                print(f"Could not calculate trend for {var_name}: {e}")
                results_list.append({
                    'Variable': var_name,
                    'Trend_Value': np.nan,
                    'P_value_Numeric': np.nan
                })

    # Convert results list to a DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Filter out variables ending in _total or _total_lp for the main tables
    filtered_df = results_df[~results_df['Variable'].str.contains('_total|_tot_')]

    # Split into 'lp' (low pass) and 'main' dataframes
    df_lp = filtered_df[filtered_df['Variable'].str.contains('_lp')]
    df_main = filtered_df[~filtered_df['Variable'].str.contains('_lp')]

    def create_latex_table(df, filename_stem):
        # Extract the site (MB, EW, WW) and the variable name (Q, Qh, Qf)
        df['Site'] = df['Variable'].str.rsplit('_', n=1).str[-1]
        # .str[0] is needed to select the first element after the split
        df['VarName'] = df['Variable'].str.replace('_lp', '').str.rsplit('_', n=1).str[0]
        
        # Initialize new columns for the *formatted strings* for the table body
        df['Formatted_Trend'] = np.nan
        df['Formatted_Pvalue'] = np.nan
        
        # Apply specific formatting based on variable name and bold for significance
        
        def format_and_bold(row, is_p_value=False):
            value_numeric = row['P_value_Numeric'] if is_p_value else row['Trend_Value']
            var_name = row['VarName']
            
            if pd.isna(value_numeric):
                return '-'

            # Q formatting
            if var_name == 'Q':
                fmt_str = "{:3.2f}"
            # Qh and Qf formatting
            else: 
                if is_p_value:
                    fmt_str = "{:4.3f}"
                else:
                    value_numeric *= 1e2 # Apply 1e2 scale for display
                    fmt_str = "{:3.2f}"
            
            val_formatted = fmt_str.format(value_numeric)

            # Apply bolding based on P_value_Numeric column
            if row['P_value_Numeric'] <= 0.05:
                return f"\\textbf{{{val_formatted}}}"
            else:
                return val_formatted

        df['Formatted_Trend'] = df.apply(lambda row: format_and_bold(row, is_p_value=False), axis=1)
        df['Formatted_Pvalue'] = df.apply(lambda row: format_and_bold(row, is_p_value=True), axis=1)

        # Create pivoted tables using the NEW formatted string columns
        trends_table_str = df.pivot(index='VarName', columns='Site', values='Formatted_Trend')
        p_values_table_str = df.pivot(index='VarName', columns='Site', values='Formatted_Pvalue')

        # Combine into a MultiIndex column DataFrame, ensuring WW, MB, EW order
        site_order = ['WW', 'MB', 'EW']
        
        # Create empty combined table with desired structure
        final_table = pd.DataFrame(index=trends_table_str.index, columns=pd.MultiIndex.from_product([site_order, ['Trend', 'P-value']]))

        # Fill the combined table with data
        for site in site_order:
            if site in trends_table_str.columns:
                final_table[(site, 'Trend')] = trends_table_str[site]
                final_table[(site, 'P-value')] = p_values_table_str[site]

        # Reorder rows to the specified order (Q, Qh, Qf)
        var_order = ['Q', 'Qh', 'Qf']
        final_table = final_table.reindex(index=var_order)
        
        # Rename row index with units (using LaTeX math environment for units)
        final_table = final_table.rename(index={'Q': '$Q$ (Sv)', 
                                                'Qh': '$Q_h$ (10$^{-2}$ PW)', 
                                                'Qf': '$Q_f$ (10$^{-2}$ Sv)'})

        # --- Generate the specific LaTeX string manually to match the desired format ---
        latex_filename = f'{filename_stem}.tex'
        
        latex_string = "\\begin{table}\n"
        latex_string += "\t\\caption{10 year trends and p-value for volume ($ Q $), heat ($ Q_{h} $) and freshwater ($ Q_{f} $) transport after \\citep{Hamed1998}.}\n"
        latex_string += "\t\\begin{tabular}{|l|cc|cc|cc|}\n"
        latex_string += "\t\t\\hline\n"
        latex_string += "\t\t& \\multicolumn{2}{c|}{\\textbf{Western wedge}} & \\multicolumn{2}{c|}{\\textbf{Mid basin}} & \\multicolumn{2}{c|}{\\textbf{Eastern wedge}} \\\\\n"
        latex_string += "\t\t& Trend & P-value & Trend & P-value & Trend & P-value \\\\\n"
        latex_string += "\t\t\\hline\n"
        
        # Add the data rows
        for index, row in final_table.iterrows():
            # Use final_table structure which has MultiIndex columns
            row_str = f"\t\t{index} & {row[('WW', 'Trend')]} & {row[('WW', 'P-value')]} & {row[('MB', 'Trend')]} & {row[('MB', 'P-value')]} & {row[('EW', 'Trend')]} & {row[('EW', 'P-value')]} \\\\\n"
            # Ensure NaNs are handled correctly (pandas default is 'nan' string if conversion fails earlier)
            row_str = row_str.replace('nan', '-')
            latex_string += row_str
            
        latex_string += "\t\t\\hline\n"
        latex_string += "\t\\end{tabular}\n"
        latex_string += "\\end{table}"

        # Save the manually generated LaTeX string to a file
        with open(latex_filename, 'w') as f:
            f.write(latex_string)

        print(f"\nSuccessfully saved LaTeX table to {latex_filename}")
        
        return final_table

    # Generate and print the tables
    print("\n" + "="*50)
    print("Main Variables Trend Table (Excluding QS, Formatted for LaTeX)")
    print("="*50)
    main_table = create_latex_table(df_main.copy(), 'Q_trend_all')
    display(main_table)
    
    print("\n" + "="*50)
    print("Low Pass Variables Trend Table (Excluding QS, Formatted for LaTeX)")
    print("="*50)
    lp_table = create_latex_table(df_lp.copy(), 'Q_lp_all')
    display(lp_table)
    
    return main_table, lp_table

    ################################

def calculate_lombscargle_spectrum(q_moored: xr.DataArray, dim: str = 'TIME') -> xr.DataArray:
    """
    Calculates the Lomb-Scargle Periodogram for cycles between 5 years and the 
    Nyquist frequency of the data, with results in Cycles Per Day (cpd).
    """
    
    if dim not in q_moored.dims:
        raise ValueError(f"Dimension '{dim}' not found in the input DataArray.")

    # 1. Extract the time values (relative days from start time) and data values
    time_numeric = (q_moored[dim].values - q_moored[dim].values[0]) / np.timedelta64(1, 'D')
    data_values = q_moored.values

    # Handle potential NaNs in your data by filtering them out for Lomb-Scargle
    valid_indices = ~np.isnan(data_values)
    t = time_numeric[valid_indices]
    x = data_values[valid_indices]
    
    # Check if we have enough data points to proceed
    if len(t) < 10:
        raise ValueError("Not enough valid data points to perform spectral analysis.")

    # 2. Define the frequencies to scan (in Cycles Per Day (cpd))

    # Dynamically calculate the average sampling interval (dt_avg) in DAYS
    dt_avg_days = np.mean(np.diff(t))
    
    # Nyquist frequency is 1 / (2 * dt_avg_days)
    # The highest frequency you can reliably measure is half the average sampling rate
    max_freq = 1.0 / (2.0 * dt_avg_days)
    
    # Minimum frequency is based on your requirement (5 years = ~1826 days)
    min_freq = 1.0 / (5 * 365.25) 
    
    num_freqs = 2000 # Increased resolution for wide range

    # Ensure min_freq is not higher than max_freq
    freqs = np.linspace(min_freq, max_freq, num_freqs) 

    # 3. Calculate the Lomb-Scargle Periodogram
    power = signal.lombscargle(t, x, freqs, normalize=True) 

    # 4. Convert the results back into an Xarray DataArray
    spectrum_da = xr.DataArray(
        power,
        coords={"frequency_cpd": freqs},
        dims=["frequency_cpd"]
    )
    
    return spectrum_da

######################

def process_all_spectra(ds_input: xr.Dataset, dim: str = 'TIME') -> xr.Dataset:
    """
    Calculates the Lomb-Scargle spectrum for specific Q/Qh/Qf variables 
    across different regions in the input Dataset.

    Parameters
    ----------
    ds_input : xr.Dataset
        The input dataset (e.g., RT_Q_Qh_Qf)
    dim : str, optional
        The name of the time dimension, by default 'TIME'.

    Returns
    -------
    xr.Dataset
        A new Dataset containing the frequency spectra for all valid variables.
    """
    
    base_vars = ['Q', 'Qh', 'Qf']
    regions = ['_MB', '_WW', '_EW', '_total']
    
    # Generate all potential variable names we are looking for
    target_vars = [base + region for base in base_vars for region in regions]

    # Dictionary to store the results
    spectra_results = {}
    
    print(f"Starting spectral analysis for {len(target_vars)} potential variables...")

    for var_name in target_vars:
        if var_name in ds_input.data_vars:
            print(f"  Processing variable: {var_name}")
            # Select the single DataArray and run the calculation
            ts_dataarray = ds_input[var_name].reset_coords()
            ts_dataarray = ts_dataarray[var_name]
            # Use the helper function to get the spectrum
            try:
                spectrum_da = calculate_lombscargle_spectrum(ts_dataarray, dim='TIME')
                spectra_results[var_name + '_spectrum'] = spectrum_da
            except ValueError as e:
                print(f"  Skipping {var_name} due to error: {e}")
        else:
            print(f"  Variable {var_name} not found in input dataset. Skipping.")

    if not spectra_results:
        print("No spectra were successfully calculated.")
        return None

    # Combine all individual spectrum DataArrays into a single new Dataset
    output_dataset = xr.Dataset(spectra_results)
    
    print(f"\nSuccessfully generated spectra for {len(output_dataset.data_vars)} variables.")
    return output_dataset