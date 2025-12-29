function mooringCoords = get_moorings_coords(dataTable, idList)
% GET_MOORINGS_COORDS Retrieves coordinates for a list of mooring IDs.
%
%   mooringCoords = get_moorings_coords(dataTable, idList) 
%   iterates through a cell array of IDs and returns a structure 
%   where each field name is the ID and the value is a struct with lat/lon.
%
%   Inputs:
%       dataTable - A MATLAB table containing 'ID', 'lat', and 'lon' columns.
%       idList    - A cell array of strings containing the target IDs (e.g., {'RTEB1', 'RTWB'}).
%
%   Outputs:
%       mooringCoords - A structure with fields named after the IDs.

    % Initialize an empty structure to store results
    mooringCoords = struct();
    
    % Loop through each ID in the input list
    for i = 1:length(idList)
        currentID = idList{i};
        
        % Call the previous function to get lat/lon for the current ID
        [lat, lon] = get_lat_lon_by_moor_id(dataTable, currentID);
        
        % Store the results in the output structure dynamically
        % MATLAB field names cannot start with a number, so we check/fix the name if needed
        fieldName = matlab.lang.makeValidName(currentID);
        
        mooringCoords.(fieldName).latitude = lat;
        mooringCoords.(fieldName).longitude = lon;
        
    end
end