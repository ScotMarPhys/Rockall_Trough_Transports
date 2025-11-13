function ladcp_selected = select_ladcp_lon_range(ladcp_data_in, lon_w, lon_e,z_lim,years_idx)
% SELECT_LADCP_LON_RANGE Selects LADCP data within a specified longitude range.
%
%   ladcp_selected = select_ladcp_lon_range(ladcp, lon_w, lon_e)
%
%   Inputs:
%   ladcp: Structure output from process_loaded_ladcp_data.m.
%   lon_w:         Western longitude boundary (e.g., -12.5 degrees).
%   lon_e:         Eastern longitude boundary (e.g., -5.0 degrees).
%
%   Output:
%   ladcp_selected: A new structure with data filtered by longitude.

% Input validation (optional but good practice)
if lon_w >= lon_e
    error('Western longitude (lon_w) must be less than Eastern longitude (lon_e).');
end

% Find the indices of stations within the desired longitude range
% We are using the 'lon' field from the input structure
indsel = find(ladcp_data_in.lon >= lon_w & ladcp_data_in.lon <= lon_e);
z_idx = find(ladcp_data_in.depth<=z_lim);

if isempty(indsel)
    warning('No stations found within the longitude range [%.2f, %.2f]. Returning empty structure.', lon_w, lon_e);
    ladcp_selected = struct();
    return;
end

% Filter all relevant fields of the input structure using the found indices
% Coordinates and Names are 1D vectors [N_stations x 1]
ladcp_selected.lon = ladcp_data_in.lon(indsel);
ladcp_selected.lat = ladcp_data_in.lat(indsel);
ladcp_selected.cumdist = ladcp_data_in.cumdist(indsel);


% Velocity and std deviation data are 2D matrices [549x20 (Depth x Station)].
% We select all rows (depths) and only the selected columns (stations/longitudes).
ladcp_selected.mean_u = ladcp_data_in.mean_u(z_idx, indsel);
ladcp_selected.mean_v = ladcp_data_in.mean_v(z_idx, indsel);
ladcp_selected.mean_vct = ladcp_data_in.mean_vct(z_idx, indsel);
ladcp_selected.stdu = ladcp_data_in.stdu(z_idx, indsel);
ladcp_selected.stdv = ladcp_data_in.stdv(z_idx, indsel);
ladcp_selected.stdvct = ladcp_data_in.stdvct(z_idx, indsel);
ladcp_selected.SEu = ladcp_data_in.SEu(z_idx, indsel);
ladcp_selected.SEv = ladcp_data_in.SEv(z_idx, indsel);
ladcp_selected.SEvct = ladcp_data_in.SEvct(z_idx, indsel);
ladcp_selected.PTMP = ladcp_data_in.PTMP(z_idx, indsel,years_idx);
ladcp_selected.SAL = ladcp_data_in.SAL(z_idx, indsel,years_idx);
ladcp_selected.u = ladcp_data_in.u(z_idx, indsel,years_idx);
ladcp_selected.v = ladcp_data_in.v(z_idx, indsel,years_idx);

% Depth grid (z_grid) typically remains the same [1x549]
ladcp_selected.depth = ladcp_data_in.depth(z_idx);

disp(['Selected ', num2str(length(indsel)), ' stations between ', num2str(lon_w), '°E and ', num2str(lon_e), '°E.']);

end