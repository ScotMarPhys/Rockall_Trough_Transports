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
import src.features.RT_data as rtd
import src.features.matfile_functions as matlab_fct

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

def repeat_upper_values(var):
    mask = var.notnull()
    mask = mask + var.shift(PRES=-10).notnull()
    var = var.interpolate_na(
        dim='PRES',
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



def calc_SA_CT_sigma0(ds):
    
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
        sw.eos80.pres,
        -abs(ds.psal.depth),ds.lat,
        dask='parallelized', output_dtypes=[float, ]
        )


    ds['SA'] = xr.apply_ufunc(gsw.SA_from_SP,
                  ds.psal,ds.PRES,ds.lon,ds.lat,
                  dask = 'parallelized',output_dtypes=[float,])
    ds.SA.attrs = SA_attrs


    ds['CT'] = xr.apply_ufunc(gsw.CT_from_t,
                  ds.SA,ds.temp,ds.PRES,
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
        data, lowcut, highcut, fs,
        input_core_dims=[[dim],[],[],[]],
        output_core_dims=[[dim]],
        dask='parallelized')
    return y


def lazy_butter_lp_filter(data, lowcut, fs,dim='time_counter'):
    y = xr.apply_ufunc(
        __butter_lowpass_filter,
        data, lowcut, fs,
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
        elif '_se' in var:
            print(f'{var} already in Dataset, will be overwritten')
        elif ds[var].size==1:
            print(f'{var} has length {ds[var].size}, no se calculated')
    
    return ds


