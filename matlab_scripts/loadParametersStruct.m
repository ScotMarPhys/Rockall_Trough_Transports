% loadParametersStruct.m
function params = loadParametersStruct(parameter_script_name)
    % Execute the script to create variables in a temporary workspace
    run(parameter_script_name);
    
    % Get a list of all variables created by the script in this function's workspace
    vars = who;
    
    % Initialize the output structure
    params = struct();
    
    % Iterate through variables and assign them to the structure fields
    for i = 1:length(vars)
        varName = vars{i};
        % Exclude the input argument 'parameter_script_name' and 'vars' itself
        if ~strcmp(varName, 'parameter_script_name') && ~strcmp(varName, 'vars') && ~strcmp(varName, 'i')
            % Assign the value of the variable to a field in the structure
            params.(varName) = eval(varName);
        end
    end
end