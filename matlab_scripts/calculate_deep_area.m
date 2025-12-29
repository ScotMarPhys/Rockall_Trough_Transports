function area_m2 = calculate_deep_area(lat, lon, bathy, threshold)
% PLOT_AND_CALCULATE_DEEP_AREA Calculates and plots the area below a depth threshold.
%
%   area_m2 = plot_and_calculate_deep_area(lat, lon, bathy, threshold)
%   calculates the cross-sectional area of water deeper than the threshold, 
%   generates a plot of the cross-section with the area highlighted, and 
%   returns the area in square meters.
%
%   Inputs:
%       lat, lon - Vectors of coordinates along the section.
%       bathy    - Vector of bathymetry values (negative for depth below sea level).
%       threshold- The depth threshold (e.g., -1760).
%
%   Output:
%       area_m2  - Total calculated area of the deep section (m^2).

    % Find the indices where the depth is greater than the threshold (more negative)
    deep_indices = find(bathy < threshold);

    % If no data is deeper than the threshold, return 0 area and skip plotting
    if isempty(deep_indices)
        warning('No bathymetry points found deeper than the threshold.');
        area_m2 = 0;
        return;
    end

    % Extract the coordinates for the sections that are deeper than the threshold
    deep_lat = lat(deep_indices);
    deep_lon = lon(deep_indices);
    deep_depths = bathy(deep_indices);

    % Calculate cumulative distance along the track for X-axis values
    % Assuming gsw_distance works on vectors and returns meters
    distances = gsw_distance(lon, lat); 
    track_distance_m = [0; cumsum(distances(:))]; % Cumulative distance in meters

    % Get the distances corresponding to the deep sections
    x_deep = track_distance_m(deep_indices);

    % --- Define the Polygon Vertices (X and Y coordinates for plotting) ---
    % The X vertices: start along the bathymetry, then reverse along the threshold line
    polygon_x = [x_deep(:); flipud(x_deep(:))];

    % The Y vertices (depth): start along the bathymetry depths, then reverse along the threshold
    polygon_y = [deep_depths(:); flipud(repmat(threshold, length(deep_depths), 1))];

    % Calculate the area
    area_m2 = polyarea(polygon_x, polygon_y); 

    %% --- Plotting the Area ---
    figure;
    hold on; 

    % Plot the filled area
    fill(polygon_x, polygon_y, 'b', 'EdgeColor', 'none', 'FaceAlpha', 0.5); 

    % Plot the original bathymetry line on top for clarity
    plot(track_distance_m, bathy, 'k-', 'LineWidth', 1.5);

    % Plot the horizontal threshold line
    yline(threshold, '--r', sprintf('%.0f m threshold', abs(threshold)));

    % --- Format the plot ---
    xlabel('Distance along section (meters)');
    ylabel('Depth (meters)');
    title('Cross-Section Area Below Threshold');
    grid on;
    hold off;
end