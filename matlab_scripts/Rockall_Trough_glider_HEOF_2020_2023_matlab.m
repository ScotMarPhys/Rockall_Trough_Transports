clear
close all

%% load data
load('C:\Users\sa07kb\OneDrive - SAMS\data\data_seaglider\glider_sections_gridded.mat')



%Reshape data matrix [dimt, dimx*dimz]
[dimx,dimz,dimt] = size(v_grid);
tot_u = reshape(v_grid,[dimx*dimz,dimt]);
tot_u = tot_u';

%Remove mean
meansect_u = nanmean(tot_u, 1);
tot_u = tot_u - repmat(meansect_u, dimt, 1);
tot_u(isnan(tot_u))=-9999;
%Calculate the Hilbert transform of the zonal velocities
DC = hilbert(tot_u);

%% Computation of the Hilbert EOF

%Compute the covariance matrix
cov_dc = cov(DC);

%Resolve the eigen value problem (eigen vectors are automatically normalized
%to 1)
[EV, D] = eig(cov_dc);

%Project to get the expansion coefficients
EC = DC * EV;

%% Plot HEOF
%% figure settings (fontsize, etc)
fs=14;

f1=figure(1);clf
%Blue->Red custom colormap
polarmap
[LAT, DEPTH] = meshgrid(lon_grid, z_grid);

for k=1:4
    subplot(4,2,2*(k-1)+1)
    [LAT, DEPTH] = meshgrid(x2_grid,z_grid);
    contourf(LAT, DEPTH, permute(real(reshape(EV(:,end-k+1), dimx, dimz)),[2,1]), 80,'LineStyle','none');
    title(['Real HEOF ',int2str(k),', expl. var. ',...
    int2str(floor(D(end-k+1,end-k+1) / sum(diag(D)) * 100)),'%'],...
    'fontweight','normal','fontsize',fs);
    set(gca,'YLim', [-1000 0], 'TickDir','out',...
	 'CLim', [-0.04 0.04],'fontsize',fs)
%     deplabel = get(gca,'YTickLabel');
%     deplabel(1,:) = char('60m');
%     set(gca,'YTickLabel',deplabel);
%     t=text(lat_label,xlab_z*ones(1,n_lat),xlabel_str);
%     set(t,'fontsize',fs,'VerticalAlignment','top','HorizontalAlignment','center')
    grid on
    
    subplot(4,2,2*(k-1)+2)
    contourf(LAT, DEPTH, permute(imag(reshape(EV(:,end-k+1), dimx, dimz)),[2,1]), 80,'LineStyle','none');
    title(['Imaginary HEOF ' int2str(k)],'fontweight','normal','fontsize',fs);
    set(gca,'YLim', [-1000 0], 'TickDir','out','YTick',60:40:270,...
	 'CLim', [-0.04 0.04],'fontsize',fs)
%     deplabel = get(gca,'YTickLabel');
%     deplabel(1,:) = char('60m');
%     set(gca,'YTickLabel',deplabel);
%     t=text(lat_label,xlab_z*ones(1,n_lat),xlabel_str);
%     set(t,'fontsize',fs,'VerticalAlignment','top','HorizontalAlignment','center')
    grid on
end
cb=colorbar('horiz','position',[0.1 .03 0.85 0.02]);
%set(get(cb,'ylabel'),'string','zonal velocity ms^{-1}','fontsize',fs,'interpreter','tex')
set(cb,'ytick',-.04:.01:0.04,'fontsize',fs)
% 
% %%% print figure
% set(f1,'PaperPositionMode','manual','PaperUnits','centimeters','PaperPosition',[0 0 19 21],'PaperSize',[19 21])
% print(f1,'-depsc','-painters',[out_dir,'NEUC_23w_u_mean_all_',outlatstr,'ev1_4'])

