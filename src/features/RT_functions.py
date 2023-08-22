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
from scipy.signal import butter, filtfilt
from xhistogram.xarray import histogram as xhist

def ds_rt_swap_vert_dim(ds_RT):
    ds_RT_swap = ds_RT.swap_dims({'PRES':'depth'})
    ds_RT_swap['depth']=-1*ds_RT_swap.depth
    ds_RT_swap = ds_RT_swap.interp(depth=ds_RT.PRES.values)
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
    y.attrs['long_name'] = f'{1/lowcut} days low pass filtered {y.long_name}'
    y.attrs['description'] = f'{1/lowcut} days low pass filtered {y.description}'
    return y

##################################################
def xcorr_norm(x,y,dim):
        """
        Perform Cross-Correlation on x and y
        x    : 1st signal
        y    : 2nd signal

        returns
        corr : coefficients of correlation
        """
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