function gebco_data = load_gebco_data(file_path)
% Loads all variables from a GEBCO netCDF file into a structure

    % Get information about the NetCDF file structure
    fileInfo = ncinfo(file_path);
    
    % Initialize an empty structure
    gebco_data = struct();
    
    % Loop through all variables listed in the file information
    for i = 1:length(fileInfo.Variables)
        varName = fileInfo.Variables(i).Name;
        
        % Read the data for the current variable
        data = ncread(file_path, varName);
        
        % Assign the data to a field in the structure
        gebco_data.(varName) = data;
    end
    
    disp(['Successfully loaded all variables from: ', file_path]);
end