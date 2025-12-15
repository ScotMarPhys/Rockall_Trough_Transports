import sys; sys.path.append(r'../') # Add this line at the beginner of each notebook to be able to load local functions 
import src.features.RT_functions as rtf
import src.features.RT_data as rtd
import src.set_paths as sps
import src.RT_parameters as rtp
import src.features.RT_EOF_functions as rt_eof

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
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.transforms as mtransforms

def create_formatter(direction, ax):
    """
    Creates a FuncFormatter that adjusts decimal precision based on tick spacing.
    """
    def formatter(x, pos):
        # Get the interval between major ticks
        major_locator = ax.xaxis.get_major_locator() if direction in ('lon', 'x') else ax.yaxis.get_major_locator()
        try:
            # `tick_values` might not be available or stable on all locators,
            # so we'll get the ticks from the ax object.
            ticks = major_locator.tick_values(ax.get_xlim()[0], ax.get_xlim()[1]) if direction in ('lon', 'x') else major_locator.tick_values(ax.get_ylim()[0], ax.get_ylim()[1])
            if len(ticks) > 1:
                interval = abs(ticks[1] - ticks[0])
            else:
                interval = 1 # Default interval if not enough ticks
        except Exception:
            interval = 1
        
        # Determine formatting based on the tick interval
        if interval < 1:
            fmt = ".1f" # Use one decimal place for fine intervals
        else:
            fmt = ".0f" # Use no decimal places for coarse intervals

        if direction == 'lon':
            if x > 0:
                return f"{x:{fmt}}{r'$\degree$'}E"
            elif x < 0:
                return f"{-x:{fmt}}{r'$\degree$'}W"
            else:
                return f"{x:{fmt}}{r'$\degree$'}"
        elif direction == 'lat':
            if x > 0:
                return f"{x:{fmt}}{r'$\degree$'}N"
            elif x < 0:
                return f"{-x:{fmt}}{r'$\degree$'}S"
            else:
                return f"{x:{fmt}}{r'$\degree$'}"
    return ticker.FuncFormatter(formatter)

def axis_lat_lon_formatter(ax,form='xlon'):
    if form=='xlon':
        lon_formatter = create_formatter('lon', ax)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lon_formatter))
        ax.set_xlabel('')
    elif form=='xlat':
        lon_formatter = create_formatter('lat', ax)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lat_formatter))
        ax.set_xlabel('')
    elif form=='ylon':
        lon_formatter = create_formatter('lon', ax)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lon_formatter))
        ax.set_ylabel('')
    elif form=='ylat':
        lon_formatter = create_formatter('lat', ax)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lat_formatter))
        ax.set_ylabel('')

def date_str_func(ds,dim='TIME',formatter="%Y/%m",save_pic=False):
    if save_pic:
        date_str = f'{ds[dim].min().dt.strftime(formatter).values}_{ds[dim].max().dt.strftime(formatter).values}'
    else:
        date_str = f'{ds[dim].min().dt.strftime(formatter).values}-{ds[dim].max().dt.strftime(formatter).values}'
    return date_str

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
            
def plot_extr_patch(ds1,ax,plot_mean=True,dim='TIME'):
    th_p = ds1.mean()+ds1.std()
    th_m = ds1.mean()-ds1.std()
    ylim = ax.get_ylim()
    
    if plot_mean:
        ax.hlines(ds1.mean(),ds1[dim].min(),ds1[dim].max(),color='r',lw=0.5)
        ax.hlines(ds1.mean()+ds1.std(),ds1[dim].min(),ds1[dim].max(),color='r',lw=0.5,ls='--')
        ax.hlines(ds1.mean()-ds1.std(),ds1[dim].min(),ds1[dim].max(),color='r',lw=0.5,ls='--')
    
    if ds1.mean()>0:
        ax.vlines(ds1[dim].where(ds1>th_p),-100,100,color='mistyrose', zorder=0)
        ax.vlines(ds1[dim].where(ds1<th_m),-100,100,color='aliceblue', zorder=0)
    else:
        ax.vlines(ds1[dim].where(ds1>th_p),-100,100,color='aliceblue', zorder=0)
        ax.vlines(ds1[dim].where(ds1<th_m),-100,100,color='mistyrose', zorder=0)
            
    ax.set_ylim(ylim)

def plot_moorings_paper(ds_RT,ds_RT_stacked,cruise_label=True):
    
    ds_cruises = rtd.load_cruise_list()
    
    fs=14
    font = {'weight' : 'normal',
            'size'   : fs}
    plt.rc('font', **font)

    sig_lev = np.array([27.2,27.4,27.6,27.7])
    vel_lev = np.arange(-.5,.55,.1)
    tem_levs = np.arange(0,15,1)
    sal_levs = np.arange(35,35.7,.1)
    
    fig,axs = plt.subplots(3,2,figsize=[12,10],sharex=True,sharey=True)

    # EB1
    sigma = gsw.sigma0(ds_RT.SG_EAST, ds_RT.TG_EAST)
    ax = axs[0,0]
    imV = (ds_RT.V_EAST*1e-2).plot(ax=ax,x='TIME',yincrease=False,
                       levels=vel_lev,cmap=cm.cm.balance,
                       add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    pres = -1*gsw.z_from_p(ds_RT_stacked.PS_EAST_1_UV, 57.1)
    pres.plot.line('k',ax=ax,x='TIME',hue='ZS_EAST_1_UV',add_legend=False,lw=0.5)
    ax.set_ylabel('Depth (m)')
    
    ax = axs[1,0]
    imT = ds_RT.TG_EAST.plot(ax=ax,x='TIME',yincrease=False,
                       levels=tem_levs,cmap=cm.cm.thermal,
                       add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    pres = -1*gsw.z_from_p(ds_RT_stacked.PS_EAST_TS, 57.1)
    pres.plot.line('k',ax=ax,x='TIME',hue='ZS_EAST_TS',add_legend=False,lw=0.5)
    ax.set_ylabel('Depth (m)')
    
    ax = axs[2,0]
    imS= ds_RT.SG_EAST.plot(ax=ax,x='TIME',yincrease=False,
                        levels=sal_levs, cmap=cm.cm.haline,
                        add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    pres = -1*gsw.z_from_p(ds_RT_stacked.PS_EAST_TS, 57.1)
    pres.plot.line('k',ax=ax,x='TIME',hue='ZS_EAST_TS',add_legend=False,lw=0.5)
    ax.set_ylabel('Depth (m)')
    
    # WB1
    sigma = gsw.sigma0(ds_RT.SG_WEST, ds_RT.TG_WEST)

    # Create merged WB1/2 CM
    ds_RT = rtf.merge_RT_WB1_2(ds_RT)

    ax = axs[0,1]
    imV = (ds_RT.v_RTWB*1e-2).plot(ax=ax,x='TIME',yincrease=False,
                       levels=vel_lev,cmap=cm.cm.balance,
                       add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    pres = -1*gsw.z_from_p(ds_RT_stacked.PS_WEST_1_UV, 57.1)
    pres.plot.line('k',ax=ax,x='TIME',hue='ZS_WEST_1_UV',add_legend=False,lw=0.5)
    pres = -1*gsw.z_from_p(ds_RT_stacked.PS_WEST_2_UV, 57.1)
    pres.plot.line('k',ax=ax,x='TIME',hue='ZS_WEST_2_UV',add_legend=False,lw=0.5)

    ax = axs[1,1]
    ds_RT.TG_WEST.plot(ax=ax,x='TIME',yincrease=False,
                       levels=tem_levs,cmap=cm.cm.thermal,
                       add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    pres = -1*gsw.z_from_p(ds_RT_stacked.PS_WEST_TS, 57.1)
    pres.plot.line('k',ax=ax,x='TIME',hue='ZS_WEST_TS',add_legend=False,lw=0.5)

    ax = axs[2,1]
    ds_RT.SG_WEST.plot(ax=ax,x='TIME',yincrease=False,
                       levels=sal_levs,cmap=cm.cm.haline,
                       add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    pres = -1*gsw.z_from_p(ds_RT_stacked.PS_WEST_TS, 57.1)
    pres.plot.line('k',ax=ax,x='TIME',hue='ZS_WEST_TS',add_legend=False,lw=0.5)


    for ax in axs[0:,1]:
        ax.set_ylabel('')
    for ax in axs.flat:
        ax.tick_params(axis='x', labelrotation=45) 

    for i, label in enumerate(('(a)', '(b)','(c)','(d)','(e)','(f)')):
        ax =  axs.flat[i]
        ax.text(0, 1.05, label, transform=ax.transAxes,
          fontsize=fs, ha='left',va='bottom')
        ax.vlines(ds_cruises.TIME,0, 1,transform=ax.get_xaxis_transform(),color='k',linestyle='--')
        ax.grid()
        if i<6:
            ax.set_xlabel('')

    if cruise_label:
        for i,text in enumerate(ds_cruises):
            if text.TIME.dt.year<=ds_RT.TIME.max().dt.year:
                if i==0:
                    t=pd.to_datetime(ds_cruises[i].TIME.values)+datetime.timedelta(days=30)
                    axs[0,0].annotate(text.values, xy=(t,-10),
                                  ha ='right', va='bottom', rotation=-60)
                    axs[0,1].annotate(text.values, xy=(t,-10),
                                      ha ='right', va='bottom', rotation=-60)
                else:
                    axs[0,0].annotate(text.values, xy=(ds_cruises[i].TIME,-10),
                                      ha ='right', va='bottom', rotation=-60)
                    axs[0,1].annotate(text.values, xy=(ds_cruises[i].TIME,-10),
                                      ha ='right', va='bottom', rotation=-60)

    plt.tight_layout()
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.70, 0.02, 0.245])
    cb =fig.colorbar(imV, cax=cbar_ax)
    cb.set_label('Meridional\nvelocity (m s$^{-1}$)')

    cbar_ax = fig.add_axes([0.92, 0.39, 0.02, 0.245])
    cb =fig.colorbar(imT, cax=cbar_ax)
    cb.set_label(f'Conservative\ntemperature {r'($^{\circ}$C)'}')

    cbar_ax = fig.add_axes([0.92, 0.08, 0.02, 0.245])
    cb =fig.colorbar(imS, cax=cbar_ax)
    cb.set_label('Absolute\nsalinity (g kg$^{-1}$)')

    return fig

##########################################################
def plot_moorings_paper_A(ds_RT,ds_RT_stacked,cruise_label=True):
    
    ds_cruises = rtd.load_cruise_list()
    
    fs=14
    font = {'weight' : 'normal',
            'size'   : fs}
    plt.rc('font', **font)

    sig_lev = np.array([27.2,27.4,27.6,27.7])
    vel_lev = np.arange(-.5,.51,.125)
    tem_levs = np.arange(-1.5,1.52,.25)
    sal_levs = np.arange(-0.15,0.152,0.025)
    
    fig,axs = plt.subplots(2,2,figsize=[12,7],sharex=True,sharey=True)

#     # EB1
    sigma = gsw.sigma0(ds_RT.SG_EAST, ds_RT.TG_EAST)

    ax = axs[0,0]
    da = ds_RT.TG_EAST
    imT = (da-da.mean('TIME')).plot(ax=ax,x='TIME',y='depth',yincrease=False,
                       levels=tem_levs,cmap=cm.cm.balance,
                       add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',y='depth',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    ds_RT_stacked.PS_EAST_TS.plot.line('k',ax=ax,x='TIME',hue='ZS_EAST_TS',add_legend=False,lw=0.5)

    ax = axs[1,0]
    da= ds_RT.SG_EAST
    imS= (da-da.mean('TIME')).plot(ax=ax,x='TIME',y='depth',yincrease=False,
                        levels=sal_levs, cmap=cm.cm.balance,
                        add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',y='depth',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    ds_RT_stacked.PS_EAST_TS.plot.line('k',ax=ax,x='TIME',hue='ZS_EAST_TS',add_legend=False,lw=0.5)

    # WB1
    sigma = gsw.sigma0(ds_RT.SG_WEST, ds_RT.TG_WEST)

    # Create merged WB1/2 CM
    ds_RT = rtf.merge_RT_WB1_2(ds_RT)

    ax = axs[0,1]
    da = ds_RT.TG_WEST
    (da-da.mean('TIME')).plot(ax=ax,x='TIME',y='depth',yincrease=False,
                       levels=tem_levs,cmap=cm.cm.balance,
                       add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',y='depth',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    ds_RT_stacked.PS_WEST_TS.plot.line('k',ax=ax,x='TIME',hue='ZS_WEST_TS',add_legend=False,lw=0.5)

    ax = axs[1,1]
    da = ds_RT.SG_WEST
    (da-da.mean('TIME')).plot(ax=ax,x='TIME',y='depth',yincrease=False,
                       levels=sal_levs,cmap=cm.cm.balance,
                       add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',y='depth',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    ds_RT_stacked.PS_WEST_TS.plot.line('k',ax=ax,x='TIME',hue='ZS_WEST_TS',add_legend=False,lw=0.5)


    for ax in axs[0:,1]:
        ax.set_ylabel('')
    for ax in axs[0:,0]:
        ax.set_ylabel('Depth (m)')
    for ax in axs.flat:
        ax.tick_params(axis='x', labelrotation=45)

    for i, label in enumerate(('(a)', '(b)','(c)','(d)')):
        ax =  axs.flat[i]
        ax.text(.0, 1.05, label, transform=ax.transAxes,
          fontsize=fs, ha='left',va='bottom')
        ax.vlines(ds_cruises.TIME,0, 1,transform=ax.get_xaxis_transform(),color='k',linestyle='--')
        ax.grid()
        if i<6:
            ax.set_xlabel('')
    
    if cruise_label:
        for i,text in enumerate(ds_cruises[:-1]):
            if i==0:
                t=pd.to_datetime(ds_cruises[i].TIME.values)+datetime.timedelta(days=30)
                axs[0,0].annotate(text.values, xy=(t,-10),
                              ha ='right', va='bottom', rotation=-60)
                axs[0,1].annotate(text.values, xy=(t,-10),
                                  ha ='right', va='bottom', rotation=-60)
            else:
                axs[0,0].annotate(text.values, xy=(ds_cruises[i].TIME,-10),
                                  ha ='right', va='bottom', rotation=-60)
                axs[0,1].annotate(text.values, xy=(ds_cruises[i].TIME,-10),
                                  ha ='right', va='bottom', rotation=-60)

    plt.tight_layout()
    fig.subplots_adjust(right=0.90)
    # cbar_ax = fig.add_axes([0.92, 0.69, 0.02, 0.25])
    # cb =fig.colorbar(imV, cax=cbar_ax)
    # cb.set_label('Meridional velocity Anomaly [m/s]')

    cbar_ax = fig.add_axes([0.92, 0.56, 0.02, 0.35])
    cb =fig.colorbar(imT, cax=cbar_ax)
    cb.set_label('Conservative temperature\n anomaly (°C)')

    cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.35])
    cb =fig.colorbar(imS, cax=cbar_ax)
    cb.set_label('Absolute salinity\n anomaly (g kg$^{-1}$)')

    return fig,axs

##########################################################
def plot_RT_mean_sections_from_mooring(ds_q_RT,ds_RT_loc):
    plt.rcParams.update({'font.size': 14})

    xticks = np.arange(-13.,-8,1)
    xticklabels = [r'13$\degree$W',r'12$\degree$W',r'11$\degree$W',
                   r'10$\degree$W',r'9$\degree$W']

    sigma_contours = [27.2,27.5, 27.7]
    manual_locations = [(-12,100),(-12,900),(-12,1250)]

    # Set up figure
    fig,axs = plt.subplots(3,1,figsize=[10,18])

    ds_q_RT.v.mean('time',keep_attrs=True).plot(
        ax=axs[0],y='depth',x='lon',yincrease=False,cmap=cm.cm.balance,
    cbar_kwargs={'label':'Meridional velocity (m/s)'})
    ds_q_RT.SA.mean('time',keep_attrs=True).plot(
        ax=axs[1],y='depth',x='lon',yincrease=False,cmap=cm.cm.haline)
    ds_q_RT.CT.mean('time',keep_attrs=True).plot(
        ax=axs[2],y='depth',x='lon',yincrease=False,cmap=cm.cm.thermal,vmin=0)

    for ax in axs:
        ax.fill_between(ds_q_RT.lon, -ds_q_RT.bathy,2300,color='grey')
        (-ds_q_RT.bathy).plot.line('k',ax=ax)
        CS = ds_q_RT.sigma0.mean('time',keep_attrs=True).where(
            ds_q_RT.depth<=-ds_q_RT.bathy).plot.contour(
            ax=ax,x='lon',levels=sigma_contours,yincrease=False,colors='k')
        ax.clabel(CS,manual=manual_locations)

        ax.vlines(ds_RT_loc.lon_RTWB,-ds_q_RT.bathy.sel(lon=ds_RT_loc.lon_RTWB,method='nearest'),
                  0,colors='k',ls='--')
        ax.vlines(ds_RT_loc.lon_RTES,-ds_q_RT.bathy.sel(lon=ds_RT_loc.lon_RTES,method='nearest'),
                  0,colors='k',ls='--')
        ax.vlines(ds_RT_loc.lon_RTWS,-ds_q_RT.bathy.sel(lon=ds_RT_loc.lon_RTWS,method='nearest'),
                  0,colors='k',ls='--')

        ax.vlines(ds_RT_loc.lon_RTADCP,-ds_q_RT.bathy.sel(lon=ds_RT_loc.lon_RTADCP,method='nearest'),
                  0,colors='k')
        ax.vlines(ds_RT_loc.lon_RTEB,-ds_q_RT.bathy.sel(lon=ds_RT_loc.lon_RTEB,method='nearest'),
                  50,colors='k')
        ax.vlines(ds_RT_loc.lon_RTWB1,-ds_q_RT.bathy.sel(lon=ds_RT_loc.lon_RTWB1,method='nearest'),
                  50,colors='k')
        ax.vlines(ds_RT_loc.lon_RTWB2,-ds_q_RT.bathy.sel(lon=ds_RT_loc.lon_RTWB2,method='nearest'),
                  1000,colors='k')

        d= 50.
        ax.plot([ds_RT_loc.lon_RTES,ds_RT_loc.lon_RTWS],[d,d],'or')
        ax.plot([ds_RT_loc.lon_RTWB],[d],'dy')
        ax.plot([ds_RT_loc.lon_RTADCP],[d],'sb')
        ax.plot([ds_RT_loc.lon_RTEB,ds_RT_loc.lon_RTWB1],[d,d],'^g')
        ax.plot([ds_RT_loc.lon_RTWB2],[1000],'^g')

        ax.set_xlim([-13.1,-9.])
        ax.set_xlabel('')
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels,fontsize=14)
        ax.grid()

        ax.set_ylabel('Depth [m]')
        ax.set_ylim([2300,0])
        
    return fig
##########################################################################################

def plot_EOF(model_EOF,v_mean,ds_RT_loc,ds_q_RT,dim,TIME_dim='TIME'):
    
    fs = 14
    font = {'weight' : 'normal',
        'size'   : fs}
    plt.rc('font', **font)
    #axis_lat_lon_formatter(ax,form='xlon')
    xticks_EW = np.arange(-9.5,ds_RT_loc.lon_RTES,.1)
    xticklabels_EW = ['9.5°W','9.4°W','9.3°W','9.2°W']

    expvar = model_EOF.explained_variance()
    expvar_ratio = model_EOF.explained_variance_ratio()
    components = model_EOF.components()

    fig,axs = plt.subplots(1,components.mode.size+1,
                           figsize=[12,4],sharey=True,sharex=True)

    ax = axs[0]
    vmin,vmax,levs=-0.2,0.2,21
    im_hdl_1 = v_mean.plot(x=dim,ax=ax,yincrease=False,add_colorbar=False,
                vmin=vmin,vmax=vmax,levels=levs,cmap='RdBu_r')
    ax.set_ylabel('Depth (m)', fontsize=fs)   


    for i,ax in enumerate(axs[1:]):
        vmin,vmax,levs=-0.02,0.02,21
        im_hdl = components.isel(mode=i).plot(x=dim,ax=ax,add_colorbar=False,
                    vmin=vmin,vmax=vmax,levels=levs,cmap='PiYG',yincrease=False)

        ax.text(0.99, 0.05,f'EOF {i+1} \nExpl. Var.\n {(expvar_ratio * 100).round(0).values[i]:.0f}%',
                transform=ax.transAxes, fontsize=fs,
                 verticalalignment='bottom',horizontalalignment='right')
        components.isel(mode=i).plot.contour(ax=ax,x=dim,colors='w',linewidths=.5,yincrease=False,
                                                           vmin=vmin,vmax=vmax,levels=levs)
        
        ax.set_title('')
        ax.set_ylabel('')      
   
    for ax in axs.flat:
        ax.grid()
        ax.set_xlabel('')
        ax.set_xticks(xticks_EW)
        ax.set_xlim([ds_RT_loc.lon_RTEB-0.01,ds_RT_loc.lon_RTES+0.01])
        axis_lat_lon_formatter(ax,form='xlon')
        ax.tick_params(axis='x', labelrotation=25)
        
        ax.fill_between(ds_q_RT.lon, -ds_q_RT.bathy,2000,color='grey')
        ax.fill_between([-9.2,-9.1], [194.7,194.7],2000,color='grey')

        ax.vlines(ds_RT_loc.lon_RTADCP,700,50,color='k',ls=':',lw=3)
        ax.plot(ds_RT_loc.lon_RTADCP,50, marker='s', 
                markerfacecolor='b', 
                markeredgecolor='w',     # White border
                markersize=10, 
                mew=2.5)
        
        ax.vlines(ds_RT_loc.lon_RTEB,1760,50,color='k',ls=':',lw=3)
        ax.plot(ds_RT_loc.lon_RTEB,50,marker='o', 
                markerfacecolor='lime', 
                markeredgecolor='k',     # White border
                markersize=12,
                mew=2.5)
        
        ax.vlines(ds_RT_loc.lon_RTES,200,50,color='k',ls=':',lw=3)
        ax.plot(ds_RT_loc.lon_RTES,50,marker='^', 
                markerfacecolor='r', 
                markeredgecolor='k',     # White border
                markersize=12,
                mew=2)
        ax.set_ylim([2000,0])
    
    for i, label in enumerate(('(a)','(b)','(c)','(d)')):
        ax =  axs.flat[i]
        ax.text(0.05, 1.02, label, transform=ax.transAxes,
          fontsize=fs, ha='left',va='bottom')
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.30)
    cbar_ax_shared = fig.add_axes([0.37, 0.08, 0.55, 0.04]) 
    fig.colorbar(im_hdl, cax=cbar_ax_shared, 
                 orientation='horizontal', label='Amplitude')
    
    # [left, bottom, width, height]
    cbar_ax_1 = fig.add_axes([0.05, 0.08, 0.25, 0.04]) 
    cb1 = fig.colorbar(im_hdl_1, cax=cbar_ax_1, 
                 orientation='horizontal', 
                 label='Meridional velocity (m s$^{-1}$)')

    major_tick_locations =  [-.2, -.1, 0, .1, .2] 
    cb1.ax.xaxis.set_major_locator(ticker.FixedLocator(major_tick_locations))
    cb1.ax.minorticks_on()
    
    return fig

###########################################################################################
def plot_EOF_HEOF(model_EOF,model_HEOF,ds_RT_loc,dim,TIME_dim='TIME'):
    fs = 14
    font = {'weight' : 'normal',
        'size'   : fs}
    plt.rc('font', **font)
    #axis_lat_lon_formatter(ax,form='xlon')
    xticks_EW = np.arange(-9.5,ds_RT_loc.lon_RTES,.1)
    xticklabels_EW = ['9.5°W','9.4°W','9.3°W','9.2°W']

    expvar = model_EOF.explained_variance()
    expvar_ratio = model_EOF.explained_variance_ratio()
    components = model_EOF.components()
    
    expvar_HEOF = model_HEOF.explained_variance()
    expvar_ratio_HEOF = model_HEOF.explained_variance_ratio()
    amp = model_HEOF.components_amplitude()
    phase = model_HEOF.components_phase()

    fig,axs = plt.subplots(3,components.mode.size,
                           figsize=[15,4*3],sharey=True,sharex=True)
    vmin,vmax,levs=-0.02,0.02,21
   
    for i,ax in enumerate(axs[0,:]):
        im_hdl = components.isel(mode=i).plot(x=dim,ax=ax,add_colorbar=False,
                    vmin=vmin,vmax=vmax,levels=levs,cmap='RdBu_r',yincrease=False)

        ax.text(0.95, 0.05,f'EOF \nExpl. Var.\n {(expvar_ratio * 100).round(0).values[i]:.0f}%',
                transform=ax.transAxes, fontsize=fs,
                 verticalalignment='bottom',horizontalalignment='right')
        components.isel(mode=i).plot.contour(ax=ax,x=dim,colors='w',linewidths=.5,yincrease=False,
                                                           vmin=vmin,vmax=vmax,levels=levs)
        
        ax.set_title(f'mode = {i+1}', fontsize=fs)
        if i>0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Depth [m]', fontsize=fs)         
            
    vmin,vmax,levs=0,0.025,26
    for i,ax in enumerate(axs[1,:]):
        im_amp = amp.isel(mode=i).plot(x=dim,ax=ax,add_colorbar=False,
                                       vmin=vmin,vmax=vmax,levels=levs,yincrease=False)
        ax.text(0.95, 0.05,f'HEOF\n Expl. Var.\n {(expvar_ratio_HEOF * 100).round(0).values[i]:.0f}%',
                            transform=ax.transAxes, fontsize=fs,
                            verticalalignment='bottom',horizontalalignment='right')
        amp.isel(mode=i).plot.contour(ax=ax,x=dim,colors='w',linewidths=.5,yincrease=False,
                                                           vmin=vmin,vmax=vmax,levels=levs)
        ax.set_title('')
        if i>0:
            ax.set_ylabel('') 
        else:
            ax.set_ylabel('Depth [m]')
            
    vmin,vmax,levs=-3.6,3.6,37
    for i,ax in enumerate(axs[2,:]):
        im_pha = phase.isel(mode=i).plot(x=dim,ax=ax,add_colorbar=False,cmap="twilight",
                                       vmin=vmin,vmax=vmax,levels=levs,yincrease=False)
        ax.text(0.95, 0.05,f'HEOF\n Expl. Var.\n {(expvar_ratio_HEOF * 100).round(0).values[i]:.0f}%',
                            transform=ax.transAxes, fontsize=fs,
                            verticalalignment='bottom',horizontalalignment='right')
        phase.isel(mode=i).plot.contour(ax=ax,x=dim,colors='w',linewidths=.5,yincrease=False,
                                                           vmin=vmin,vmax=vmax,levels=levs) 
        ax.set_title('')
        if i>0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Depth [m]')

   
    for ax in axs.flat:
            ax.grid()
            ax.set_xlabel('')
            ax.set_xlim([ds_RT_loc.lon_RTEB-0.01,ds_RT_loc.lon_RTES+0.01])
            ax.set_xticks(xticks_EW)
            ax.set_xticklabels(xticklabels_EW,fontsize=fs)
    
    for i, label in enumerate(('a)', 'b)','c)','d)','e)','f)','g)','h)','i)')):
        ax =  axs.flat[i]
        ax.text(0.05, 1.02, label, transform=ax.transAxes,
          fontsize=fs, ha='left',va='bottom')
    
    plt.tight_layout()
    fig.subplots_adjust(right=0.90)
    cbar_ax = fig.add_axes([0.92, 0.7, 0.02, 0.25])
    cb =fig.colorbar(im_hdl, cax=cbar_ax)
    cb.ax.set_ylabel('amplitude')
    
    cbar_ax = fig.add_axes([0.92, 0.37, 0.02, 0.25])
    cb =fig.colorbar(im_amp, cax=cbar_ax)
    cb.ax.set_ylabel('amplitude')
    
    cbar_ax = fig.add_axes([0.92, 0.07, 0.02, 0.25])
    cb =fig.colorbar(im_pha, cax=cbar_ax)
    cb.ax.set_ylabel('phase')
    return fig

#########################
def quick_plot_spectrum(spectrum_test):
    vl_val_days = np.array([16,30,60,90,365,])
    vl_labels = ['16 days', '30 days', '60 days', '90 days', '1 year']
    ymin,ymax = 0,0.2
    text_pad_points = 3 
    
    fig,axs=plt.subplots(1,1)
    
    ax=axs
    spectrum_test.sel(frequency_cpd = slice(None,0.1)).plot(ax=ax)
    
    ax.vlines(
        x=1/vl_val_days,    # The X-coordinates of the lines
        ymin=ymin,       # The bottom of all the lines
        ymax=ymax,       # The top of all the lines
        colors='red',        # Color of the lines
        linestyles='--',     # Style of the lines (e.g., dashed)
        lw=0.8,       # Label for the legend 
            )
    for x_pos, label_txt in zip(vl_val_days, vl_labels):
        offset_transform = mtransforms.offset_copy(
            ax.transData, 
            fig=fig, 
            x=text_pad_points, # Offset in X direction by 5 points
            y=0, 
            units='points'
        )
        ax.text(
            x=1/(x_pos), 
            y=ymax * 0.95,           # Place text near the top of the line (95% of ymax)
            s=label_txt,             # The text string
            rotation='vertical',     # Make the text vertical
            horizontalalignment='left', # Align the right edge of the text with x_pos
            verticalalignment='top',
            color='red',
            fontsize=10,
            transform=offset_transform
        )
    
    ax.set_ylim([ymin,ymax])
    ax.set_xlabel('Frequency [cycles per day]')
    ax.set_ylabel('Power Spectral\nDensity')
    ax.grid(True)

################

def plot_spectra_by_region_vertical(spectra_ds: xr.Dataset, var_base: str,fs=14):
    """
    Plots the Lomb-Scargle spectra for Total, EW, MB, and WW regions 
    in a 4x1 vertical subplot grid for a specific variable base (Q, Qh, or Qf).
    """
    font = {'weight' : 'normal',
        'size'   : fs}
    plt.rc('font', **font)

    # colors for regions
    c_EW = 'C1'
    c_MB = 'C2'
    c_WW = 'C0'
    c_total = 'k'
    
    # Configuration for vertical lines
    vl_val_days = np.array([16, 30, 60, 90, 365.25])
    vl_labels = ['16 days', '30 days', '60 days', '90 days', '1 year']
    x_positions_cpd = 1 / vl_val_days
    ymin, ymax = 0, 0.2  # Adjust Y-limits as needed for your data range
    xmin,xmax= None,0.08
    text_pad_points = 3 

    regions = ['_total', '_EW', '_MB', '_WW']
    region_titles = ['(a) Total', '(b) Eastern wedge',
                     '(c) Mid basin', '(d) Western wedge']
    c_region = [c_total,c_EW,c_MB,c_WW]

    # Create a 4x1 vertical subplot grid, sharing the X and Y axes
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True, sharey=True)

    for i, region in enumerate(regions):
        ax = axs[i]
        var_name = f"{var_base}{region}_spectrum"
        
        if var_name in spectra_ds.data_vars:
            # 1. Plot the actual spectrum data
            spectra_ds[var_name].sel(frequency_cpd=slice(None, 0.1)).plot.line(
                ax=ax, color=c_region[i], linewidth=1.5)
            # Add the title to the specific axis
            ax.set_title(f"{region_titles[i]} ({var_base})", loc='left',
                         fontsize=fs)
        else:
            ax.set_title(f"{region_titles[i]} (Data Missing)", loc='left')
            print(f"Warning: {var_name} not found in input dataset.")

        # 2. Add Vertical Lines and Labels
        ax.vlines(
            x=x_positions_cpd,
            ymin=ymin,
            ymax=ymax,
            colors='red',
            linestyles='--',
            lw=0.8,
        )
        
        for x_pos in x_positions_cpd:
            offset_transform = mtransforms.offset_copy(
                ax.transData, 
                fig=fig, 
                x=text_pad_points, 
                y=0, 
                units='points'
            )
            
            # Find the corresponding label for the current x_pos (frequency)
            label_index = np.where(np.isclose(x_positions_cpd, x_pos))[0][0]
            label_txt = vl_labels[label_index]

            ax.text(
                x=x_pos, 
                y=ymax * 0.95,
                s=label_txt,
                rotation='vertical',
                horizontalalignment='left',
                verticalalignment='top',
                color='red',
                fontsize=fs-2,
                transform=offset_transform
            )
        
        # 3. Final Axis Configuration
        ax.set_ylim([ymin, ymax])
        ax.set_xlim([xmin,xmax])
        ax.set_ylabel('Power Spectral\nDensity')
        ax.grid(True)
        
        # Remove X-axis label for all but the bottom-most plot
        if i < len(regions) - 1:
            ax.set_xlabel('')
        else:
            ax.set_xlabel('Frequency (cycles per day)')


    plt.tight_layout()
    # Add a main super title if needed
    plt.show()