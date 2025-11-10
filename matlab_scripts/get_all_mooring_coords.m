function mooringCoords = get_all_mooring_coords(dataTable)
% GET_ALL_MOORING_COORDS Retrieves coordinates for all unique mooring IDs in the table.
%
%   mooringCoords = get_all_mooring_coords(dataTable) 
%   iterates through all unique IDs found in the input dataTable 
%   and returns a structure of their coordinates.
%   It handles special cases like 'RTWB' if the helper function is available.
%
%   Inputs:
%       dataTable - A MATLAB table containing 'ID', 'lat', and 'lon' columns.
%
%   Outputs:
%       mooringCoords - A structure with fields named after the IDs.

    % 1. Extract all unique IDs from the table's 'ID' column
    %    unique() ensures we don't process RTWB1 and RTWB2 multiple times individually 
    %    if we intend to call a helper function that manages the mean calculation.
    allIDs = unique(dataTable.ID);
    allIDs = [allIDs; {'RTWB'}];
    
    % Initialize an empty structure
    mooringCoords = struct();
    
    % Loop through each unique ID
    for i = 1:length(allIDs)
        currentID = allIDs{i}; % Use braces {} for cell array access
        
        % Ensure field names are valid MATLAB names (e.g. starting with a letter)
        fieldName = matlab.lang.makeValidName(currentID);
        
        % Call your existing helper function to get lat/lon for the current ID
        % Note: This relies on 'get_lat_lon_by_moor_id' being in your path or a local function
        [lat, lon] = get_lat_lon_by_moor_id(dataTable, currentID);
        
        % Store the results in the output structure
        if ~isnan(lat)
            mooringCoords.(fieldName).lat = lat;
            mooringCoords.(fieldName).lon = lon;
        else
            warning('Could not retrieve valid coordinates for ID: %s', currentID);
        end
    end
end