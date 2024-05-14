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

def plot_moorings_paper(ds_RT,ds_RT_stacked):
    
    ds_cruises = rtd.load_cruise_list()
    
    fs=14
    font = {'weight' : 'normal',
            'size'   : fs}
    plt.rc('font', **font)

    sig_lev = np.array([27.2,27.4,27.6,27.7])
    vel_lev = np.arange(-.5,.55,.1)
    tem_levs = np.arange(0,15,1)
    sal_levs = np.arange(35,35.7,.1)
    fig,axs = plt.subplots(3,2,figsize=[19,12],sharex=True,sharey=True)

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
    ds_RT_stacked.PS_EAST_1_UV.plot.line('k',ax=ax,x='TIME',hue='ZS_EAST_1_UV',add_legend=False,lw=0.5)

    ax = axs[1,0]
    imT = ds_RT.TG_EAST.plot(ax=ax,x='TIME',yincrease=False,
                       levels=tem_levs,cmap=cm.cm.thermal,
                       add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    ds_RT_stacked.PS_EAST_TS.plot.line('k',ax=ax,x='TIME',hue='ZS_EAST_TS',add_legend=False,lw=0.5)

    ax = axs[2,0]
    imS= ds_RT.SG_EAST.plot(ax=ax,x='TIME',yincrease=False,
                        levels=sal_levs, cmap=cm.cm.haline,
                        add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    ds_RT_stacked.PS_EAST_TS.plot.line('k',ax=ax,x='TIME',hue='ZS_EAST_TS',add_legend=False,lw=0.5)

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
    ds_RT_stacked.PS_WEST_1_UV.plot.line('k',ax=ax,x='TIME',hue='ZS_WEST_1_UV',add_legend=False,lw=0.5)
    ds_RT_stacked.PS_WEST_2_UV.plot.line('k',ax=ax,x='TIME',hue='ZS_WEST_2_UV',add_legend=False,lw=0.5)

    ax = axs[1,1]
    ds_RT.TG_WEST.plot(ax=ax,x='TIME',yincrease=False,
                       levels=tem_levs,cmap=cm.cm.thermal,
                       add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    ds_RT_stacked.PS_WEST_TS.plot.line('k',ax=ax,x='TIME',hue='ZS_WEST_TS',add_legend=False,lw=0.5)

    ax = axs[2,1]
    ds_RT.SG_WEST.plot(ax=ax,x='TIME',yincrease=False,
                       levels=sal_levs,cmap=cm.cm.haline,
                       add_colorbar=False)
    p=sigma.plot.contour(ax=ax,x='TIME',
                        levels=sig_lev,colors='grey',
                        yincrease=False,linewidths=1)
    # plt.clabel(p,levels=sig_lev[::2],fmt='%3.1f',fontsize=fs)
    ds_RT_stacked.PS_WEST_TS.plot.line('k',ax=ax,x='TIME',hue='ZS_WEST_TS',add_legend=False,lw=0.5)


    for ax in axs[0:,1]:
        ax.set_ylabel('')

    for i, label in enumerate(('a)', 'b)','c)','d)','e)','f)')):
        ax =  axs.flat[i]
        ax.text(-.05, 1., label, transform=ax.transAxes,
          fontsize=fs, ha='left',va='bottom')
        ax.vlines(ds_cruises.TIME,0, 1,transform=ax.get_xaxis_transform(),color='k',linestyle='--')
        ax.grid()
        if i<6:
            ax.set_xlabel('')

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
    cbar_ax = fig.add_axes([0.92, 0.69, 0.02, 0.25])
    cb =fig.colorbar(imV, cax=cbar_ax)
    cb.set_label('Meridional velocity [m/s]')

    cbar_ax = fig.add_axes([0.92, 0.38, 0.02, 0.25])
    cb =fig.colorbar(imT, cax=cbar_ax)
    cb.set_label('Conservative temperature [°C]')

    cbar_ax = fig.add_axes([0.92, 0.06, 0.02, 0.25])
    cb =fig.colorbar(imS, cax=cbar_ax)
    cb.set_label('Absolute salinity [g/kg]')

    return fig


##########################################################
def plot_RT_mean_sections_from_mooring(ds_q_RT,ds_RT_loc):
    plt.rcParams.update({'font.size': 14})

    xticks = np.arange(-13.,-8,1)
    xticklabels = ['13°W','12°W','11°W','10°W','9°W']

    sigma_contours = [27.2,27.5, 27.7]
    manual_locations = [(-12,100),(-12,900),(-12,1250)]

    # Set up figure
    fig,axs = plt.subplots(3,1,figsize=[10,13])

    ds_q_RT.v.mean('time',keep_attrs=True).plot(
        ax=axs[0],y='depth',x='lon',yincrease=False,cmap=cm.cm.balance)
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