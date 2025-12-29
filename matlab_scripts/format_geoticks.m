function format_geoticks(ax_handle, axis_char, type_str,font_type,fs)
% FORMAT_GEOTICKS_CLASSIC formats axis tick labels with degree symbols and cardinal directions.
%
%   format_geoticks_classic(ax_handle, axis_char, type_str)
%
%   Inputs:
%   ax_handle:  The handle to the axes (e.g., gca).
%   axis_char:  Specify 'x' or 'y' for the desired axis.
%   type_str:   Specify 'lon' (Longitude) or 'lat' (Latitude).

% Ensure inputs are valid
if ~ishandle(ax_handle) || ~strcmp(get(ax_handle, 'Type'), 'axes')
    error('First input must be a valid axes handle.');
end
if ~ismember(axis_char, {'x', 'y'})
    error('Axis character must be ''x'' or ''y''.');
end
if ~ismember(type_str, {'lon', 'lat'})
    error('Type string must be ''lon'' or ''lat''.');
end

% Determine the cardinal directions based on the type
if strcmp(type_str, 'lon')
    positive_dir = 'E';
    negative_dir = 'W';
    axis_label_text = 'Longitude';
else % type_str is 'lat'
    positive_dir = 'N';
    negative_dir = 'S';
    axis_label_text = 'Latitude';
end

% Get the current tick values
if strcmp(axis_char, 'x')
    tick_values = get(ax_handle, 'XTick');
else
    tick_values = get(ax_handle, 'YTick');
end

% Format the labels
new_labels = cell(size(tick_values));
degree_symbol = char(176); % Unicode for degree symbol

for i = 1:length(tick_values)
    value = tick_values(i);
    % Determine the direction suffix
    if value > 0
        direction = positive_dir;
    elseif value < 0
        direction = negative_dir;
    else
        direction = ''; % No direction for 0 degrees
    end
    
    % Format the string: absolute value, degree symbol, direction
    % Using '%.0f' for integers, adjust precision as needed (e.g., '%.2f')
    new_labels{i} = sprintf('%.1f%s%s', abs(value), degree_symbol, direction);
end

% Apply the new labels and update the main axis label
if strcmp(axis_char, 'x')
    set(ax_handle, 'XTickLabels', new_labels, ...
        'Fontsize',fs,'Fontname',font_type);
    % xlabel(ax_handle, axis_label_text);
else
    set(ax_handle, 'YTickLabels', new_labels, ...
        'Fontsize',fs,'Fontname',font_type);
    % ylabel(ax_handle, axis_label_text);
end

end