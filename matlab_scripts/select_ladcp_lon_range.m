function ladcp_selected = select_ladcp_lon_range(ladcp_data_in, lon_w, lon_e)
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

if isempty(indsel)
    warning('No stations found within the longitude range [%.2f, %.2f]. Returning empty structure.', lon_w, lon_e);
    ladcp_selected = struct();
    return;
end

% Filter all relevant fields of the input structure using the found indices
% Coordinates and Names are 1D vectors [N_stations x 1]
ladcp_selected.lon = ladcp_data_in.lon(indsel);
ladcp_selected.lat = ladcp_data_in.lat(indsel);
ladcp_selected.cast_name = ladcp_data_in.cast_name(indsel);
ladcp_selected.cast_max_depth = ladcp_data_in.cast_max_depth(indsel);

% Velocity and std deviation data are 2D matrices [549x20 (Depth x Station)].
% We select all rows (depths) and only the selected columns (stations/longitudes).
ladcp_selected.mean_u = ladcp_data_in.mean_u(:, indsel);
ladcp_selected.mean_v = ladcp_data_in.mean_v(:, indsel);
ladcp_selected.mean_vct = ladcp_data_in.mean_vct(:, indsel);
ladcp_selected.stdu = ladcp_data_in.stdu(:, indsel);
ladcp_selected.stdv = ladcp_data_in.stdv(:, indsel);
ladcp_selected.stdvct = ladcp_data_in.stdvct(:, indsel);
ladcp_selected.SEu = ladcp_data_in.SEu(:, indsel);
ladcp_selected.SEv = ladcp_data_in.SEv(:, indsel);
ladcp_selected.SEvct = ladcp_data_in.SEvct(:, indsel);
ladcp_selected.PTMP = ladcp_data_in.PTMP (indsel,:,:);
ladcp_selected.SAL = ladcp_data_in.SAL (indsel,:,:);
ladcp_selected.u = ladcp_data_in.u (indsel,:,:);
ladcp_selected.v = ladcp_data_in.v (indsel,:,:);

% Depth grid (z_grid) typically remains the same [1x549]
ladcp_selected.depth = ladcp_data_in.depth;

disp(['Selected ', num2str(length(indsel)), ' stations between ', num2str(lon_w), '°E and ', num2str(lon_e), '°E.']);

end