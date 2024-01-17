import gsw
import scipy
import numpy as np
import xarray as xr
import xeofs as xe
from matplotlib import pyplot as plt

def add_nan_glider_sections(ds_glider):
    t1=np.datetime64('2020-09-01')
    t2=np.datetime64('2021-03-01')
    t3=np.datetime64('2021-09-01')
    t4=np.datetime64('2022-03-01')
    t5=np.datetime64('2022-06-01')
    t6=np.datetime64('2022-10-01')
    dummy1=ds_glider.isel(time=0)*np.nan
    dummy1['time']=t1
    dummy2=ds_glider.isel(time=0)*np.nan
    dummy2['time']=t2
    dummy3=ds_glider.isel(time=0)*np.nan
    dummy3['time']=t3
    dummy4=ds_glider.isel(time=0)*np.nan
    dummy4['time']=t4
    dummy5=ds_glider.isel(time=0)*np.nan
    dummy5['time']=t5
    dummy6=ds_glider.isel(time=0)*np.nan
    dummy6['time']=t6
    ds_glider_nan = xr.concat([ds_glider.sel(time=slice(None,t1)),dummy1,
                              ds_glider.sel(time=slice(t1,t2)),dummy2,
                              ds_glider.sel(time=slice(t2,t3)),dummy3,
                              ds_glider.sel(time=slice(t3,t4)),dummy4,
                              ds_glider.sel(time=slice(t4,t5)),dummy5,
                              ds_glider.sel(time=slice(t5,t6)),dummy6,
                              ds_glider.sel(time=slice(t6,None)),
                              ],
                             dim='time')
    return ds_glider_nan

def normalize(x=None, y=None):
    return y / np.linalg.norm(y) / (x.max() - x.min()) ** 0.5


def normal_mode(x, mode=1):
    """Create normal sine modes."""
    return normalize(
        x=x,
        y=np.sin(
            mode * np.pi * (x - x.min()) / (x.max() - x.min())
        )
    )

def harmonic_cycle(t, T=1, phi=0):
    """Create harmonic cycles."""
    return np.sin(2 * np.pi / T * (t + phi))

def plot_EOF(model,dim,EOF=True,PC=False):
    expvar = model.explained_variance()
    expvar_ratio = model.explained_variance_ratio()
    scores = model.scores()
    components = model.components()
    
    if PC:
        axhdl = scores.plot.line(x="time", col="mode", lw=1, ylim=(-0.2, 0.2))
        for i,ax in enumerate(axhdl.axes.flat):
            ax.axhline(0,color='k',ls='--')
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    if EOF:
        fs=14
        font = {'weight' : 'normal',
                'size'   : fs}
        plt.rc('font', **font)
        fig,axs = plt.subplots(1,components.mode.size,figsize=[12,4],sharey=True)
        vmin,vmax,levs=-0.02,0.02,21
        for i,ax in enumerate(axs):
            im_hdl = components.isel(mode=i).plot(x=dim,ax=ax,add_colorbar=False,
                        vmin=vmin,vmax=vmax,levels=levs,cmap='RdBu_r',)

            ax.text(0.95, 0.05,f'Expl. Var.\n {(expvar_ratio * 100).round(0).values[i]:.0f}%',
                    transform=ax.transAxes, fontsize=fs,
                     verticalalignment='bottom',horizontalalignment='right')
            components.isel(mode=i).plot.contour(ax=ax,x=dim,colors='w',linewidths=.5,
                                                               vmin=vmin,vmax=vmax,levels=levs)
            ax.grid()
            if i>0:
                ax.set_ylabel('')
        plt.tight_layout()
        fig.subplots_adjust(right=0.90)
        cbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.7])
        cb =fig.colorbar(im_hdl, cax=cbar_ax)
    return fig

def EOF_func(v_anomaly,n_modes=4,plot_out=True,dim='x'):
    kwargs = dict(n_modes = n_modes, use_coslat=False,check_nans=True)
    model = xe.models.EOF( **kwargs)
    # model = xe.models.ComplexEOF(padding="none", **kwargs)

    model.fit(v_anomaly, dim="time")
    
    if plot_out == True:
        fig = plot_EOF(model,dim=dim,EOF=True,PC=True)
    
    return model

def lin_reg(X,y):
    XT = X.transpose() # dims (mode,depth*lon)
    XT_X = np.matmul(XT,X) # dims (mode,mode)
    XT_y = np.matmul(XT,y) # dims (mode,time)
    alpha=np.matmul(np.linalg.inv(XT_X),XT_y) # dims (mode,time)
    # print(f'alpha dims: {alpha.shape}')
    return alpha

def EOF_alpha(ds_X,ds_y):
    # reshape to (depth*lon,time)
    y = np.matrix(ds_y.stack(loc=['depth','lon']).fillna(0).to_numpy()).transpose()
    # print(f'y dims: {y.shape}')

    ds_X['depth']=np.abs(ds_X.depth)
    ds_X = ds_X.reindex(depth=list(reversed(ds_X.depth)))
    # reshape dimension (depth*lon,mode)
    X = np.matrix(ds_X.stack(loc=['depth','lon']).fillna(0).to_numpy()).transpose()
    # print(f'X dims: {X.shape}')
    
    return xr.DataArray(data=lin_reg(X,y),coords=dict(mode=(ds_X.mode),time=ds_y.time))

def rec_v_sec(ds_X,ds_y,glider_EOF,glider_vcur):
    v_rec_sec = xr.DataArray()
    for nmod in ds_X.mode.values:
        alpha = EOF_alpha(ds_X.sel(mode=slice(None,nmod)),ds_y)

        # reconstruction = alpha*EOF_EV + mean_glider_section
        v_rec = glider_EOF.components()*alpha+glider_vcur.mean('time')
        v_rec = v_rec.mean('mode')
        v_rec['mode']=nmod
        if nmod == 1:    
            v_rec_sec = v_rec
        else:
            v_rec_sec = xr.concat([v_rec_sec,v_rec],dim='mode')
    v_rec_sec['lat']=('lon',v_rec_sec.lat.isel(depth=0).values)
    return v_rec_sec

def calc_transport(da_v):
    dx = gsw.distance(da_v.lon[:2],da_v.lat[:2])
    dz = abs(da_v.depth[:2].diff('depth'))
    T = ((da_v.rename('Q')*dx*dz.values).sum(['lon','depth'])*1e-6)
    T.attrs = {'long_name':'Meridional Transport','units':'Sv'}
    return T

### visualisation functions

def plot_mean_section(ds_glider,ds_q_RT,v_rec,mode_no=1,mean=False):
    
    fig,axs = plt.subplots(1,4,figsize=[15,3])
    vmin,vmax,levs=-0.2,0.2,41
    ax=axs[0]
    ds_glider.vcur.mean(['time']).plot(x='lon',ax=ax,vmin=vmin,vmax=vmax,levels=levs,cmap='RdBu_r')
    ax.set_title('Glider')
    ax=axs[1]
    ds_q_RT.v_EW.mean(['TIME']).plot(x='lon_EW',ax=ax,vmin=vmin,vmax=vmax,levels=levs,cmap='RdBu_r')
    ax.set_title('RT EW full')
    v_EOF = v_rec.sel(mode=mode_no)
    if mean:
        v_EOF = v_EOF.mean('mode')        
    ax=axs[2]
    v_EOF.mean('time').plot(x='lon',ax=ax,vmin=vmin,vmax=vmax,levels=levs,cmap='RdBu_r')
    ax.set_title(f'EOF {mode_no} full')
    ax=axs[3]
    v_EOF.interp(time=ds_glider.time.values
                    ).mean(['time']).plot(x='lon',ax=ax,vmin=vmin,vmax=vmax,levels=levs,cmap='RdBu_r')
    ax.set_title(f'EOF {mode_no} resampled')
    plt.tight_layout()

def plot_seasonal_cycle(ds_glider,ds_q_RT,v_rec,ax=0,mode_no=1,mean=False):
    color='C0'
    m=ds_glider.vcur.groupby('time.month').mean(['time','lon','depth'])
    d=ds_glider.vcur.groupby('time.month').std('time').mean(['lon','depth'])
    ax.errorbar(m.month, m, yerr=d, fmt='-', capsize=3, capthick=1, color=color,label='Glider')

    color='C1'
    m = ds_q_RT.v_EW.groupby('TIME.month').mean(['TIME','lon_EW','depth']) 
    d=ds_q_RT.v_EW.groupby('TIME.month').std('TIME').mean(['lon_EW','depth'])
    ax.errorbar(m.month, m, yerr=d, fmt='-', capsize=3, capthick=1, color=color,label='RT EW full')

    v_EOF = v_rec.sel(mode=mode_no)
    if mean:
        v_EOF = v_EOF.mean('mode') 

    color='C2'
    m = v_EOF.groupby('time.month').mean(['time','lon','depth']) 
    d=v_EOF.groupby('time.month').std('time').mean(['lon','depth'])
    ax.errorbar(m.month, m, yerr=d, fmt='-', capsize=3, capthick=1, color=color,label=f'EOF {mode_no} full')

    color='C3'
    m = v_EOF.interp(time=ds_glider.time.values
                    ).groupby('time.month').mean(['time','lon','depth'])
    d=v_EOF.interp(time=ds_glider.time.values
                    ).groupby('time.month').std('time').mean(['lon','depth'])
    ax.errorbar(m.month, m, yerr=d, fmt='-', capsize=3, capthick=1, color=color,label=f'EOF {mode_no} resampled')
    ax.grid()
    ax.legend(bbox_to_anchor=(0,1.02,1,0.2),loc='lower left',mode='expand',ncol=2)

def plot_longterm(ds_glider,ds_q_RT,v_rec,ax=0,mode_no=1,mean=False):

    ds_q_RT.v_EW.sel(TIME=slice(ds_glider.time.min().values,ds_glider.time.max().values)
                    ).mean(['lon_EW','depth']).plot.line('-',label='RT EW orig',color='C1',ax=ax)
    ds_glider.vcur.mean(['lon','depth']).plot.line('-',label='glider',color='C0',ax=ax)
    v_EOF = v_rec.sel(mode=mode_no)
    if mean:
        v_EOF = v_EOF.mean('mode') 
    v_EOF.sel(time=slice(ds_glider.time.min().values,ds_glider.time.max().values)
              ).mean(['lon','depth']).plot.line('-',label=f'EOF {mode_no} full',color='C2',ax=ax)
    ax.legend()
    ax.grid()

def plot_transport(Q_glider,Q_rec,Q_moor,ax=0,mode_no=1,mean=False):

    Q_moor.sel(TIME=slice(ds_glider.time.min().values,ds_glider.time.max().values)
                    ).plot.line('-',label='RT EW orig',color='C1',ax=ax)
    Q_glider.plot.line('-',label='glider',color='C0',ax=ax)
    Q_rec = Q_rec.sel(mode=mode_no)
    if mean:
        Q_rec = Q_rec.mean('mode') 
    Q_rec.sel(time=slice(ds_glider.time.min().values,ds_glider.time.max().values)
              ).plot.line('-',label=f'EOF {mode_no} full',color='C2',ax=ax)
    ax.legend()
    ax.grid()

def plot_error(da_Q_obs,da_Q_rec,mode,axs):
    
    if mode==0:
        Q_rec = da_Q_rec
        result = scipy.stats.linregress(Q_rec,da_Q_obs)
        RMSE = np.sqrt(((da_Q_obs - Q_rec)**2).mean('time'))
        axs.plot(da_Q_obs,Q_rec,'.',
             label=f'Fraser et al. (2022), \nR={result.rvalue:3.2f}, \nRMSE={RMSE:3.2f} Sv, \nSTDE={result.stderr:3.2f} ')
    else:
        for i in range(mode):
            Q_rec = da_Q_rec.isel(mode=i)
            result = scipy.stats.linregress(Q_rec,da_Q_obs)
            RMSE = np.sqrt(((da_Q_obs - Q_rec)**2).mean('time'))
            axs.plot(da_Q_obs,Q_rec,'.',
                     label=f'{Q_rec.mode.values} EOFs, \nR={result.rvalue:3.2f}, \nRMSE={RMSE:3.2f} Sv, \nSTDE={result.stderr:3.2f} ')
    axs.plot(np.arange(-7,11),np.arange(-7,11),color='k',lw=0.8,ls='--')
    axs.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2))
    axs.set_xlabel('Observed transport')
    axs.set_ylabel('Reconstructed transport')
    axs.set_ylim([-3,11])
    axs.set_xlim([-6,10])
    axs.set_aspect('equal', adjustable='box')
    axs.grid()
    axs.axvline(0,color='k',lw=0.8,ls='--')
    axs.axhline(0,color='k',lw=0.8,ls='--')

def plot_transport(Q_glider,Q_rec,Q_moor,ax=0,mode_no=1,mean=False):

    Q_moor.sel(TIME=slice(Q_glider.time.min().values,Q_glider.time.max().values)
                    ).plot.line('-',label='RT EW orig',color='C1',ax=ax)
    Q_glider.plot.line('-',label='glider',color='C0',ax=ax)
    Q_rec = Q_rec.sel(mode=mode_no)
    if mean:
        Q_rec = Q_rec.mean('mode') 
    Q_rec.sel(time=slice(Q_glider.time.min().values,Q_glider.time.max().values)
              ).plot.line('-',label=f'EOF {mode_no} full',color='C2',ax=ax)
    ax.legend()
    ax.grid()

def plot_seasonal_cycle_Q(Q_glider,Q_rec,Q_moor,ax=0,mode_no=1,mean=False):
    color='C0'
    m=Q_glider.groupby('time.month').mean('time')
    d=Q_glider.groupby('time.month').std('time')
    ax.errorbar(m.month, m, yerr=d, fmt='-', capsize=3, capthick=1, 
                color=color,label='Glider')

    color='C1'
    m = Q_moor.groupby('TIME.month').mean('TIME') 
    d=Q_moor.groupby('TIME.month').std('TIME')
    ax.errorbar(m.month, m, yerr=d, fmt='-', capsize=3, capthick=1, 
                color=color,label='RT EW full')

    v_EOF = Q_rec.sel(mode=mode_no)
    if mean:
        v_EOF = v_EOF.mean('mode') 

    color='C2'
    m = v_EOF.groupby('time.month').mean('time') 
    d=v_EOF.groupby('time.month').std('time')
    ax.errorbar(m.month, m, yerr=d, fmt='-', capsize=3, capthick=1, 
                color=color,label=f'EOF {mode_no} full')

    color='C3'
    dummy = v_EOF.interp(time=Q_glider.time.values)
    m = dummy.groupby('time.month').mean('time')
    d= dummy.groupby('time.month').std('time')
    ax.errorbar(m.month, m, yerr=d, fmt='-', capsize=3, capthick=1, 
                color=color,label=f'EOF {mode_no} resampled')
    
    color='C4'
    m = Q_glider.sel(time=slice(None,v_EOF.time.max())
                    ).groupby('time.month').mean('time')
    d = Q_glider.sel(time=slice(None,v_EOF.time.max())
                    ).groupby('time.month').std('time')
    ax.errorbar(m.month, m, yerr=d, fmt='-', capsize=3, capthick=1, 
                color=color,label=f'Glider 2020- Oct 2022')
    
    ax.grid()
    ax.legend(bbox_to_anchor=(0,1.02,1,0.2),loc='lower left',mode='expand',ncol=2)
    ax.set_xlabel('Month of year')
    ax.set_ylabel(f'{Q_glider.long_name} [{Q_glider.units}]')