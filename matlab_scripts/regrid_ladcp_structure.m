function ladcp_gridded = regrid_ladcp_structure(ladcp)
    % REGRID_LADCP_STRUCTURE grids the ladcp data structure to a regular 
    % cumulative distance grid.
    %
    % ladcp_gridded = regrid_ladcp_structure(ladcp)

    disp('Starting regridding process...');

    % --- 1. Define the original and target grids ---

    % Calculate cumulative distance along the transect (X-axis for gridding)
    % gsw_distance expects [lat; lon] format for multiple points
    dist = gsw_distance(ladcp.lat, ladcp.lon); 
    cast_cumdist = [0; cumsum(dist)]; % The X coordinates for original data
    
    % Define the target grid's X coordinates (e.g., every 5 km)
    target_cumdist_x = (cast_cumdist(1):5e3:cast_cumdist(end))'; 

    % Create 2D meshgrids for original and target points
    [X, Y] = meshgrid(cast_cumdist, ladcp.depth); % Original grid points
    [Xi, Yi] = meshgrid(target_cumdist_x, ladcp.depth); % Target grid points

    % Interpolate physical lat/lon coordinates onto the new X-axis
    ladcp_gridded.lon = interp1(cast_cumdist, ladcp.lon, Xi(1,:));
    ladcp_gridded.lat = interp1(cast_cumdist, ladcp.lat, Xi(1,:));
    ladcp_gridded.depth = ladcp.depth; % Depth coordinate remains the same

    disp(['Original grid size: ', num2str(size(X, 2)), ' points']);
    disp(['Target grid size: ', num2str(size(Xi, 2)), ' points (every 5 km)']);


    % --- 2. Regrid 3D variables (depth x original_point x year) ---
    % Variables: PTMP, SAL, u, v

    variables_3d = {'PTMP', 'SAL', 'u', 'v'};
    num_years = size(ladcp.v, 3); 

    for var_name = variables_3d
        var_name_str = char(var_name);
        disp(['Regridding 3D variable: ', var_name_str]);

        original_data_3d = ladcp.(var_name_str); 
        % MATLAB matrix dimensions are usually: depth x point x time
        [num_xi_points, num_yi_points] = size(Xi);
        % Pre-allocate the output matrix: [num_depths x num_target_points x num_years]
        regridded_data = NaN(num_xi_points, num_yi_points, num_years);

        for year_idx = 1:num_years
            % Extract the 2D slice for the current year
            data_slice_2d = squeeze(original_data_3d(:,:,year_idx));
   
            % Perform gridding (using 'linear' interpolation)
            % griddata(X_orig, Y_orig, V_orig, X_target, Y_target, Method)
            Vq = griddata(X, Y, data_slice_2d', Xi, Yi, 'linear');
            
            regridded_data(:,:,year_idx) = Vq;
        end
        
        ladcp_gridded.(var_name_str) = regridded_data;
    end

    % --- 3. Regrid 2D variables (depth x original_point) ---
    % Variables: mean_u, mean_v, mean_vct, stdu, stdv, stdvct, SEu, SEv, SEvct
    % These usually represent time averages across all years (no 3rd dimension)

    variables_2d = {'mean_u', 'mean_v', 'mean_vct', 'stdu', 'stdv', 'stdvct', 'SEu', 'SEv', 'SEvct'};

    for var_name = variables_2d
        var_name_str = char(var_name);
        disp(['Regridding 2D variable: ', var_name_str]);
        
        original_data_2d = ladcp.(var_name_str);

        % Perform gridding for the 2D average data
        Vq = griddata(X, Y, original_data_2d, Xi, Yi, 'linear');
        
        ladcp_gridded.(var_name_str) = Vq;
    end
    ladcp_gridded.cumdist = target_cumdist_x
    disp('Regridding complete.');

end