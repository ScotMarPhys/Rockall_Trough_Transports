function generateStatsTable(T, filename_tex, region_str)
% generateStatsTable Calculates statistics, prints to console, and saves a LaTeX table.
%
% Inputs:
%   T            : A structure containing integrated transports T.Qf, T.Qf_pro, T.Qh, T.Qh_pro
%   filename_tex : String specifying the output .tex filename (e.g., 'stats_MB.tex')
%   region_str   : String label for the region (e.g., 'Mid-Basin (MB)')

% Extract the relevant integrated transport vectors from the structure T
Qf = T.Qf;         % 'Full' calculation results (e.g., Qf_MB)
Qf_pro = T.Qf_pro; % 'Profile' calculation results (e.g., Qf_MB_pro)
Qh = T.Qh;         % 'Full' calculation results (e.g., Qh_MB)
Qh_pro = T.Qh_pro; % 'Profile' calculation results (e.g., Qh_MB_pro)


%% 1. Calculate Statistics

% Calculate differences (for MBE and RMSE)
Qf_diff = Qf - Qf_pro;
Qh_diff = Qh - Qh_pro;

% Calculate Means
Qf_m     = mean(Qf, 'all', 'omitnan');
Qh_m     = mean(Qh, 'all', 'omitnan');
Qf_pro_m = mean(Qf_pro, 'all', 'omitnan');
Qh_pro_m = mean(Qh_pro, 'all', 'omitnan');

% Calculate Standard Deviations (using robust syntax for compatibility)
Qf_std      = std(Qf(:), 'omitnan');
Qh_std      = std(Qh(:), 'omitnan');
Qf_pro_std  = std(Qf_pro(:), 'omitnan');
Qh_pro_std  = std(Qh_pro(:), 'omitnan');

% Calculate Mean Bias Error (MBE)
MBE_Qf = mean(Qf_diff, 'all', 'omitnan');
MBE_Qh = mean(Qh_diff, 'all', 'omitnan');

% Calculate Root-Mean-Square Error (RMSE) (requires Statistics Toolbox or Manual calc)
% Using manual calculation for broad compatibility
RMSE_Qf = sqrt(mean((Qf - Qf_pro).^2, 'all', 'omitnan'));
RMSE_Qh = sqrt(mean((Qh - Qh_pro).^2, 'all', 'omitnan'));


%% 2. Scale values for 10e-2 units

% Define the scaling factor (10e-2 = 0.01)
scale_factor = 1e-2;

Qf_m_scaled      = Qf_m      / scale_factor;
Qf_pro_m_scaled  = Qf_pro_m  / scale_factor;
MBE_Qf_scaled    = MBE_Qf    / scale_factor;
Qf_std_scaled    = Qf_std    / scale_factor;
Qf_pro_std_scaled= Qf_pro_std/ scale_factor;
RMSE_Qf_scaled   = RMSE_Qf   / scale_factor;

Qh_m_scaled      = Qh_m      / scale_factor;
Qh_pro_m_scaled  = Qh_pro_m  / scale_factor;
MBE_Qh_scaled    = MBE_Qh    / scale_factor;
Qh_std_scaled    = Qh_std    / scale_factor;
Qh_pro_std_scaled= Qh_pro_std/ scale_factor;
RMSE_Qh_scaled   = RMSE_Qh   / scale_factor;


%% 3. Print stats to MATLAB command window

fprintf('\nStatistics Summary for Region: %s\n', region_str);
fprintf('####################################################\n');
fprintf('%20s %15s %15s\n', 'Metric', 'Qf (10e-2 Sv)', 'Qh (10e-2 PW)');
fprintf('####################################################\n');

% Data rows using 4 decimal places
fprintf('%20s %15.4f %15.4f\n', 'Mean full',       Qf_m_scaled,       Qh_m_scaled);
fprintf('%20s %15.4f %15.4f\n', 'Mean profile',    Qf_pro_m_scaled,   Qh_pro_m_scaled);
fprintf('%20s %15.4f %15.4f\n', 'Mean Bias',       MBE_Qf_scaled,     MBE_Qh_scaled);
fprintf('----------------------------------------------------\n');
fprintf('%20s %15.4f %15.4f\n', 'Std Dev full',    Qf_std_scaled,     Qh_std_scaled);
fprintf('%20s %15.4f %15.4f\n', 'Std Dev profile', Qf_pro_std_scaled, Qh_pro_std_scaled);
fprintf('%20s %15.4f %15.4f\n', 'RMSE',            RMSE_Qf_scaled,    RMSE_Qh_scaled);
fprintf('####################################################\n\n');


%% 4. Print as LaTeX table to the specified file

fileID = fopen(filename_tex, 'w'); 

if fileID == -1
    error('Could not open file %s for writing.', filename_tex);
end

% Write LaTeX table preamble
fprintf(fileID, '\\begin{table}[h!]\n');
fprintf(fileID, '\\centering\n');
fprintf(fileID, '\\caption{Summary of Qf and Qh Statistics for %s (Units are in $10^{-2}$ Sv and $10^{-2}$ PW)}\n', region_str);
fprintf(fileID, '\\label{tab:stats_summary_%s}\n', region_str);
fprintf(fileID, '\\begin{tabular}{|l|c|c|}\n');
fprintf(fileID, '\\hline\n');

% Write LaTeX table header row (using & for column separation, \\ for new line)
fprintf(fileID, 'Metric & Qf (10e-2 Sv) & Qh (10e-2 PW) \\\\\n');
fprintf(fileID, '\\hline\n');
fprintf(fileID, '\\hline\n');

% Write data rows (%.4f ensures 4 decimal places)
fprintf(fileID, 'Mean full & %.4f & %.4f \\\\\n', Qf_m_scaled, Qh_m_scaled);
fprintf(fileID, 'Mean profile & %.4f & %.4f \\\\\n', Qf_pro_m_scaled, Qh_pro_m_scaled);
fprintf(fileID, 'Mean Bias & %.4f & %.4f \\\\\n', MBE_Qf_scaled, MBE_Qh_scaled);
fprintf(fileID, '\\hline\n');
fprintf(fileID, 'Std Dev full & %.4f & %.4f \\\\\n', Qf_std_scaled, Qh_std_scaled);
fprintf(fileID, 'Std Dev profile & %.4f & %.4f \\\\\n', Qf_pro_std_scaled, Qh_pro_std_scaled);
fprintf(fileID, 'RMSE & %.4f & %.4f \\\\\n', RMSE_Qf_scaled, RMSE_Qh_scaled);
fprintf(fileID, '\\hline\n');

% Write LaTeX table postamble
fprintf(fileID, '\\end{tabular}\n');
fprintf(fileID, '\\end{table}\n');

% Close the file handle
fclose(fileID);

disp(['Successfully saved LaTeX table to: ', filename_tex]);

end