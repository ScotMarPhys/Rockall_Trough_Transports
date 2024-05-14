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

def plot_figure_gap(v_merge,y_pred,py_str,y_pred_lp,py_lp_str,period=[None,None]):
        fig,axs = plt.subplots(3,1,figsize=[15,8])
        ax = axs[0]
        v_merge[py_str].sel(TIME=slice(*period)).plot(label=py_str,lw=1,ax=ax,color='C1')
        v_merge[y_pred].sel(TIME=slice(*period)).plot(label=f'pred {py_str}',lw=0.58,ax=ax,color='k')

        ax = axs[1]
        v_merge[py_lp_str].sel(TIME=slice(*period)).plot(
            label=f'{py_str} {day_cutoff:.0f}d-lp',lw=1,ax=ax,color='C1')
        v_merge[y_pred_lp].sel(TIME=slice(*period)).plot(
            label=f'pred {py_str} {day_cutoff:.0f}d-lp',lw=0.58,ax=ax,color='k')
        
        ax = axs[2]
        (v_merge[py_str]-v_merge[y_pred]).sel(TIME=slice(*period)).plot.line(
            'x',label=f'residuals',lw=0.58,ax=ax,color='C0')
        (v_merge[py_lp_str]-v_merge[y_pred_lp]).sel(TIME=slice(*period)).plot.line(
            '+',label=f'residuals lp',lw=0.58,ax=ax,color='C2')

        for ax in axs.flat:
            ax.legend()
            ax.set_title('')
            ax.grid()

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

def gsw_geo_strf_dyn_height(SA,CT,P,P_ref):
    y = xr.apply_ufunc(
        gsw.geo_strf_dyn_height,
        SA,CT,P,P_ref,
        input_core_dims=[['depth','TIME'],['depth','TIME'],['depth'],[]],
        output_core_dims=[['depth','TIME']],
        dask='parallelized')
    return y

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
def decorrelation(x,y,dim,doplot,precision=2):
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
def detrend_data(ds,dim):
   
    # # remove annual and semiannual harmonic
    # ds_HF = harm_fit(ds,dims=dims)
    # ds_AH = reconstr_ts(ds_HF,ds[dims].values,365,dims=dims)
    # ds_SH = reconstr_ts(ds_HF,ds[dims].values,365/2,dims=dims)
    # ds_no_cyc = ds-ds_AH-ds_SH
    
    ds = ds - ds.mean(dim)
    
    p = ds.polyfit(dim=dim, deg=1)
    fit = xr.polyval(ds[dim], p.polyfit_coefficients)


    ## SIGNIFICANCE TEST

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
    t_val = np.sqrt((rsq_y*(dof-2))/(1-rsq_y));

    # significance test, t-test, 95% interval, H_0: R=0.0
    if abs(t_crit)<abs(t_val):
        trend_significant = True
    else:
        trend_significant = False
    
    slope=(p.polyfit_coefficients.sel(degree=1).values)
    intc=(p.polyfit_coefficients.sel(degree=0).values)
    
    # print(f'Trend of {slope} is significant: {}')
    #         ds_trnd=([dim],fit),
    #         INT_dtnd=([dim],ds-fit+ds.mean(dim)),
    #         INT_trnd_signf=(trend_significant),
        
#     top_int,bot_int = regression_line_ci(0.05,
#                                          p.polyfit_coefficients.sel(degree=1).values,
#                                          p.polyfit_coefficients.sel(degree=0).values,
#                                          ds,dims)
    
    # ds_detrend = xr.Dataset(data_vars=dict(
    #         ds_orig=([dim],ds),
    #         ds_reg_slope=(p.polyfit_coefficients.sel(degree=1).values),
    #         ds_reg_intc=(p.polyfit_coefficients.sel(degree=0).values),
    #         ds_trnd=([dim],fit),
    #         INT_dtnd=([dim],ds-fit+ds.mean(dim)),
    #         INT_trnd_signf=(trend_significant),
    #     ),
    #     coords=dict(TIME=ds[dim].values),
    #     )

    return fit, slope, intc, trend_significant

##################################################

## Standard error
def std_error(da,dim='TIME'):
    da_std  = da.std()
    _,dof = decorrelation(da.fillna(0),da.fillna(0),dim,0);
    return da_std/np.sqrt(dof);

###################################################

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

#######################################
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

#######################################

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
    ' derived from dynamic height difference between the moorings RTEB1 and RTWB1+2'

    if sens_analysis:
        ## Some sensitivity analysis - extra

        # Calculate temporal mean over entire timeseries and broadcast over TIME dimension
        (ds_RT_ave,_) = xr.broadcast(ds_RT.mean('TIME'),ds_RT.TIME)

        # % Do it again isolating CT changes/ keeping SA fixed
        SF_RTWB_SAfix = gsw_geo_strf_dyn_height(ds_RT.SG_WEST,ds_RT_ave.TG_WEST,ds_RT.PRES,P_ref)
        SF_RTEB_SAfix = gsw_geo_strf_dyn_height(ds_RT.SG_EAST,ds_RT_ave.TG_EAST,ds_RT.PRES,P_ref)
        q_MB_SAfix = ds_RT.dz*((SF_RTEB_SAfix - SF_RTWB_SAfix)/f)
        Q_MB_SAfix = q_MB_SAfix.sum('depth',min_count=1)/1e6
        Q_MB_SAfix.attrs['name']= 'RT_Q_MB_SAfix'
        Q_MB_SAfix.attrs['long_name']= 'RT MB Volume Transport CT fixed'
        Q_MB_SAfix.attrs['units']='Sv'
        Q_MB_SAfix.attrs['description']='As Q_MB but holding temperature fixed at temporal mean at all moorings'

        # % Do it again isolating SA changes / keeping CT fixed
        SF_RTWB_CTfix = gsw_geo_strf_dyn_height(ds_RT_ave.SG_WEST,ds_RT.TG_WEST,ds_RT.PRES,P_ref)
        SF_RTEB_CTfix = gsw_geo_strf_dyn_height(ds_RT_ave.SG_EAST,ds_RT.TG_EAST,ds_RT.PRES,P_ref)
        q_MB_CTfix = ds_RT.dz*((SF_RTEB_CTfix - SF_RTWB_CTfix)/f)
        Q_MB_CTfix = q_MB_CTfix.sum('depth',min_count=1)/1e6
        Q_MB_CTfix.attrs['name']= 'RT_Q_MB_CTfix'
        Q_MB_CTfix.attrs['long_name']= 'RT MB Volume Transport SA fixed'
        Q_MB_CTfix.attrs['units']='Sv'
        Q_MB_CTfix.attrs['description']= 'As Q_MB but holding salinity fixed at temporal mean at all moorings'

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
                        Q_MB_EB_CTvar.rename(Q_MB_EB_CTvar.attrs['name'])])
    else:
        RT_Q_MB = Q_MB
        
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
    
    return RT_Q_MB, q_MB

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

    ds_RT_MB_grid = xr.merge([q_MB_grid.rename('q'),
                          TG_MB_grid.rename('CT').drop('lat_MB').interp(lon_MB=q_MB_grid.lon_MB),
                          SG_MB_grid.rename('SA').drop('lat_MB').interp(lon_MB=q_MB_grid.lon_MB)])
    return ds_RT_MB_grid

#########################

def calc_WW_transport(ds_RT,ds_RT_loc,RT_hor_grid,bathy_WW):
    v_WW = ds_RT.v_RTWB.rename('v_WW').compute()
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
    q_WW = dx_WW*ds_RT.dz*(v_WW/1e2)

    # Integrate for transport timeseries (Sv)
    Q_WW = q_WW.sum(['depth','lon_WW'],min_count=1)/1e6 
    Q_WW.coords['mask_WW'] = ds_RT.v_RTWB.isel(depth=50).notnull()
    Q_WW.attrs['name']= 'RT_Q_WW'
    Q_WW.attrs['long_name']= 'RT WW Volume Transport'
    Q_WW.attrs['units']='Sv'
    Q_WW.attrs['description']='Volume transport at western wedge of Rockall Trough'\
    ' derived from moored velocity measurements at RTWB1+2'

    return Q_WW, q_WW
#########################

def calc_fluxes(ds):
    SA_ref = 35.34         # Reference Absolute Salinity (g/kg), for freshwater flux calc
    CT_ref = 7.07          # Reference Conservative Temperature (C), for heat flux calc
    rhoCp = 4.1e6         # Constant: Reference density times specific heat capacity (J m^-3 C^-1)   
    rho0 = 1027.4            # Reference density      
    
    qh = rhoCp*ds.q*(ds.CT - CT_ref)
    qf = -1*ds.q*(ds.SA - SA_ref)/SA_ref
    qS = ds.q*ds.SA/rho0
    
    
    qh_attrs={'name':'qh',
            'long_name':'heat transport per grid cell',
            'units':'PW',
            'description':f'Heat transport per grid cell referenced '\
            f'to temperature of {CT_ref}degC'}
    qf_attrs = {'name': 'qf',
                'long_name': 'Freshwater transport per grid cell',
                'units':'Sv',
                'description':f'Freshwater transport per grid cell referenced '\
                f'to salinity of {CT_ref} g/kg'}
    qS_attrs = {'name': 'qS',
                'long_name': 'Salt transport per grid cell',
                'units':'Sv',
                'description':f'Salt transport per grid cell referenced '\
                f'to density time specific heat capacity of of {rho0}kg m^-3'}
    
    qh.attrs =qh_attrs
    qf.attrs =qf_attrs
    qS.attrs =qS_attrs
    return qh, qf, qS