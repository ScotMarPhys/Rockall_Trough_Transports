function [SA, CT] = calculateSACT(pt, SP, depth, lon, lat)
% calculateSACT Calculates Absolute Salinity (SA) and Conservative Temperature (CT).
% 
% Inputs:
%   pt    : potential temperature (ITS-90, deg C), 3D array (n_z x n_x x n_y).
%   SP    : Practical Salinity (PSS-78, unitless), 3D array (n_z x n_x x n_y).
%   depth : depth (m), 1D array (1 x n_z).
%   lon   : longitude (degrees), 1D array (1 x n_x).
%   lat   : latitude (degrees), 1D array (1 x n_y).
%
% Outputs:
%   SA    : Absolute Salinity (g/kg), 3D array (n_z x n_x x n_y).
%   CT    : Conservative Temperature (ITS-90, deg C), 3D array (n_z x n_x x n_y).
%
% This function requires the TEOS-10 GSW Oceanographic Toolbox.

% Get dimensions from the primary 3D inputs
[n_x, n_z, n_y] = size(pt);
depth = -1.*(abs(depth));
% --- Step 1: Reshape inputs for singleton expansion ---
depth_reshaped = reshape(depth, 1,n_z,  1);
lon_reshaped = reshape(lon, n_x, 1,  1);
lat_reshaped = reshape(lat, n_x, 1, 1);

depth_reshaped = repmat(depth_reshaped, n_x,1, n_y);
lon_reshaped = repmat(lon_reshaped, 1, n_z, n_y);
lat_reshaped = repmat(lat_reshaped, 1, n_z, n_y);

% --- Step 2: Calculate sea pressure (p) from depth ---
% The gsw_p_from_z function handles broadcasting for lat. We use the reshaped
% depth and broadcast lat to match the dimensions of depth_reshaped.
p = gsw_p_from_z(depth_reshaped, mean(squeeze(lat))*ones(size(depth_reshaped)));

% --- Step 3: Calculate Absolute Salinity (SA) from Practical Salinity (SP) ---
% The gsw_SA_from_SP function requires SP, p, lon, and lat to be the same size
% through broadcasting.
SA = gsw_SA_from_SP(SP, p, lon_reshaped, lat_reshaped);

% --- Step 4: Calculate Conservative Temperature (CT) from potential temperature (pt) ---
% The gsw_CT_from_pt function requires SA and pt to have the same dimensions.
CT = gsw_CT_from_pt(SA, pt);

end