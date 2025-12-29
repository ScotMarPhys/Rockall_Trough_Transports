function T = calculateTransports(ladcp, CT_pro, SA_pro, rtp)
% calculateTransports Calculates volume, heat, and freshwater transports.
%
% Inputs:
%   ladcp   : Structure containing .v (velocity), .depth, .cumdist, .CT, .SA, .lon
%   CT_pro  : 3D array of reference Conservative Temperature profiles
%   SA_pro  : 3D array of reference Absolute Salinity profiles
%   rtp     : Structure containing required parameters: .rhoCp, .CT_ref, .SA_ref
%
% Output:
%   T       : Structure containing calculated transports: 
%             q, qh, qf (cell transports)
%             Q, Qh, Qf (integrated transports)
%             qh_pro, qf_pro, Qh_pro, Qf_pro (profile-based transports)

% Get dimensions
[n_z, n_x, n_y] = size(ladcp.v);

% --- Calculate dx and dz spacing ---

% Calculate dz spacing (depth difference between cells)
dz = diff(ladcp.depth);
dz = ([0,dz] + [dz,0])/2;

% Calculate dx spacing (horizontal distance between cells)
dx = diff(ladcp.cumdist);
dx = ([0,dx'] + [dx',0])/2; % Assuming cumdist is 1xN_x or Nx x 1

% Reshape for singleton expansion (broadcasting)
dz = reshape(dz, n_z , 1, 1); % Nz x 1 x 1
dx = reshape(dx, 1, n_x, 1);  % 1 x Nx x 1

% --- Calculate cell transports (q) ---

% Volume Transport in each cell (m^3/s per cell)
T.q = dz .* dx .* ladcp.v; 

% Heat Transport (W per cell)
T.qh = rtp.rhoCp .* T.q .* (ladcp.CT - rtp.CT_ref);

% Freshwater Transport (m^3/s freshwater equivalent per cell)
T.qf = -1 .* T.q .* (ladcp.SA - rtp.SA_ref) ./ rtp.SA_ref;

% --- Calculate profile-based transports (q_pro) using input profiles ---

% Heat Transport using profile reference (W per cell)
T.qh_pro = rtp.rhoCp .* T.q .* (CT_pro - rtp.CT_ref);

% Freshwater Transport using profile reference (m^3/s per cell)
T.qf_pro = -1 .* T.q .* (SA_pro - rtp.SA_ref) ./ rtp.SA_ref;


% --- Calculate integrated basin-wide transports (Q) ---
% Use nansum to sum over the first two dimensions (Z and X) for each Y index (time/ensemble)

% Integrated Volume Transport (Sv, 1e6 m^3/s)
T.Q = squeeze(nansum(nansum(T.q, 1), 2)) ./ 1e6;

% Integrated Heat Transport (PW, 1e15 W)
T.Qh = squeeze(nansum(nansum(T.qh, 1), 2)) ./ 1e15;
T.Qh_pro = squeeze(nansum(nansum(T.qh_pro, 1), 2)) ./ 1e15;

% Integrated Freshwater Transport (Sv, 1e6 m^3/s)
T.Qf = squeeze(nansum(nansum(T.qf, 1), 2)) ./ 1e6;
T.Qf_pro = squeeze(nansum(nansum(T.qf_pro, 1), 2)) ./ 1e6;

end