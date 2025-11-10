function [latitude, longitude] = get_lat_lon_by_moor_id(dataTable, targetID)
% GET_LAT_LON_BY_MOOR_ID Retrieves the latitude and longitude for a given mooring ID.
%
%   [latitude, longitude] = get_lat_lon_by_moor_id(dataTable, targetID) 
%   finds the coordinates for the given targetID string within the provided dataTable.
%   Special functionality: If targetID is 'RTWB', it returns the mean location 
%   of 'RTWB1' and 'RTWB2'.
%
%   Inputs:
%       dataTable - A MATLAB table containing 'ID', 'lat', and 'lon' columns.
%       targetID  - A string containing the specific ID to search for (e.g., 'RTEB1').
%
%   Outputs:
%       latitude  - The latitude value corresponding to the ID (or mean location).
%       longitude - The longitude value corresponding to the ID (or mean location).
%                   Returns NaN if the ID is not found.

    % Check for the special case 'RTWB' first
    if strcmp(targetID, 'RTWB')
        % --- Calculate mean of RTWB1 and RTWB2 locations ---
        
        % CRITICAL FIX: Call the correct function name: get_lat_lon_by_moor_id
        [lat1, lon1] = get_lat_lon_by_moor_id(dataTable, 'RTWB1'); 
        [lat2, lon2] = get_lat_lon_by_moor_id(dataTable, 'RTWB2');
        
        % Check if both sub-locations were found successfully (i.e., not NaN)
        if ~isnan(lat1) && ~isnan(lat2)
            latitude = mean([lat1, lat2]);
            longitude = mean([lon1, lon2]);
            disp('Calculated mean coordinates for RTWB.');
        else
            warning('Could not find both RTWB1 and RTWB2 data to calculate the mean.');
            latitude = NaN;
            longitude = NaN;
        end
        
    % For any other ID, search the table normally (the original logic)
    elseif ischar(targetID) || isstring(targetID)
        rowIndex = strcmp(dataTable.ID, targetID);
        
        if any(rowIndex)
            latitude = dataTable.lat(rowIndex);
            longitude = dataTable.lon(rowIndex);
        else
            warning('ID "%s" not found in the table.', targetID);
            latitude = NaN;
            longitude = NaN;
        end
        
    else
        % Handle invalid input type
        error('Invalid targetID input type. Must be a string.');
    end
end