function ladcp_out = process_loaded_ladcp_data(ladcp_data_in, area_select_str)
% PROCESS_LOADED_LADCP_DATA Processes and sorts a pre-loaded LADCP data structure.
%
%   ladcp_out = process_loaded_ladcp_data(ladcp_data_in, area_select_str)
%
%   Inputs:
%   ladcp_data_in:   Structure loaded directly from the .mat file (e.g., via load()).
%   area_select_str: String specifying the geographic area ('Full', 'RTP', or 'RT').
%
%   Output:
%   ladcp_out:       A structure containing all sorted and processed variables.

%% Map the input area string to the internal 'iarea' integer logic (1, 2, 3)
switch lower(area_select_str)
    case 'full'
        iarea = 1; 
    case 'rtp' % Rockall Trough + Plateau
        iarea = 2;
    case 'rt'  % Rockall Trough only
        iarea = 3;
    otherwise
        warning('Unknown area string provided. Defaulting to full section (iarea = 1).');
        iarea = 1;
end


%% Rename variables using the input structure's fields
% Accessing variables like ladcp_data_in.EELsta.reflat
lat = ladcp_data_in.EELsta.reflat;
lon = ladcp_data_in.EELsta.reflon;
cast_name = ladcp_data_in.EELsta.name;

% Accessing the mean and std deviation variables
mean_u_ladcp     = ladcp_data_in.mean_EEL_U;
mean_v_ladcp     = ladcp_data_in.mean_EEL_V;
mean_vct_ladcp   = ladcp_data_in.mean_EEL_cstr;
stdu_ladcp  = ladcp_data_in.std_EEL_U;
stdv_ladcp  = ladcp_data_in.std_EEL_V;
stdvct_ladcp = ladcp_data_in.std_EEL_cstr;
SEu_ladcp   = ladcp_data_in.SE_EEL_U;
SEv_ladcp   = ladcp_data_in.SE_EEL_V;
SEvct_ladcp = ladcp_data_in.SE_EEL_cstr;

% single sections
PTMP = ladcp_data_in.EEL_PTMP;
SAL = ladcp_data_in.EEL_SAL;
U = ladcp_data_in.EEL_U;
V = ladcp_data_in.EEL_V;

% Assuming z_grid is also loaded in the input structure
z_grid = ladcp_data_in.z_grid; 


%% Remove columns/stations with only NaN data
ibad = [];
for ill = 1:size(mean_u_ladcp, 1) % Iterate through rows/stations
    if isempty(find(~isnan(mean_u_ladcp(ill,:)), 1))
        ibad = [ibad ill];
    end
end

% Remove the bad stations from all variables
lat(ibad) = [];
lon(ibad) = [];
cast_name(ibad) = [];
mean_u_ladcp(ibad,:) = [];
mean_v_ladcp(ibad,:) = [];
mean_vct_ladcp(ibad,:) = [];
stdu_ladcp(ibad,:) = [];
stdv_ladcp(ibad,:) = [];
stdvct_ladcp(ibad,:) = [];
SEu_ladcp(ibad,:) = [];
SEv_ladcp(ibad,:) = [];
SEvct_ladcp(ibad,:) = [];

PTMP(ibad,:,:) = [];
SAL(ibad,:,:) = [];
U(ibad,:,:) = [];
V(ibad,:,:) = [];


%% Sort stations by longitude (S<60N) then latitude (S>=60N)
i1 = find(lat < 60);
i2 = find(lat >= 60);

[~, Isort1] = sort(lon(i1));
[~, Isort2] = sort(-lat(i2)); % Sort descending latitude

I1 = i1(Isort1);
I2 = i2(Isort2);
I = [I2; I1]; % Combined sorted indices

% Sort all variables
cast_lon_sorted = lon(I);
cast_lat_sorted = lat(I);
cast_name_sorted = cast_name(I);
% Note: Transposing the data matrices here to have depth as rows and stations as columns
mean_u_ladcp_sorted = mean_u_ladcp(I, :)'; 
mean_v_ladcp_sorted = mean_v_ladcp(I, :)';
mean_vct_ladcp_sorted = mean_vct_ladcp(I, :)';
stdu_ladcp_sorted = stdu_ladcp(I, :)';
stdv_ladcp_sorted = stdv_ladcp(I, :)';
stdvct_ladcp_sorted = stdvct_ladcp(I, :)';
SEu_ladcp_sorted = SEu_ladcp(I, :)';
SEv_ladcp_sorted = SEv_ladcp(I, :)';
SEvct_ladcp_sorted = SEvct_ladcp(I, :)';

PTMP_sorted = PTMP(I, :,:);
SAL_sorted = SAL(I, :,:);
U_sorted = U(I, :,:);
V_sorted = V(I, :,:);

% Calculate max depth for sorted casts
cast_depth_sorted = zeros(1, length(cast_lon_sorted));
for ill = 1:length(cast_lon_sorted)
    cast_depth_sorted(ill) = max(z_grid(~isnan(mean_u_ladcp_sorted(:, ill))));
end

%% Select subset based on iarea switch
indsel = [];
switch iarea
    case 1 % Full
        indsel = 1:length(cast_lon_sorted); 
    case 2 % RTP
        sta1 = 'IB12';
        sta2 = 'T';
        i1 = find(strcmp(cast_name_sorted, sta1));
        i2 = find(strcmp(cast_name_sorted, sta2));
        indsel = i1:i2;
    case 3 % RT
        sta1 = 'IB1';
        sta2 = 'R';
        i1 = find(strcmp(cast_name_sorted, sta1));
        i2 = find(strcmp(cast_name_sorted, sta2));           
        indsel = i1:i2;     
end

% Apply the selection indices to all sorted variables and assign to output structure
ladcp_out.lon = cast_lon_sorted(indsel);
ladcp_out.lat = cast_lat_sorted(indsel);
ladcp_out.cast_name = cast_name_sorted(indsel);
ladcp_out.cast_max_depth = cast_depth_sorted(indsel);
ladcp_out.mean_u = mean_u_ladcp_sorted(:, indsel);
ladcp_out.mean_v = mean_v_ladcp_sorted(:, indsel);
ladcp_out.mean_vct = mean_vct_ladcp_sorted(:, indsel);
ladcp_out.stdu = stdu_ladcp_sorted(:, indsel);
ladcp_out.stdv = stdv_ladcp_sorted(:, indsel);
ladcp_out.stdvct = stdvct_ladcp_sorted(:, indsel);
ladcp_out.SEu = SEu_ladcp_sorted(:, indsel);
ladcp_out.SEv = SEv_ladcp_sorted(:, indsel);
ladcp_out.SEvct = SEvct_ladcp_sorted(:, indsel);
ladcp_out.PTMP = PTMP_sorted(indsel,:,:);
ladcp_out.SAL = SAL_sorted(indsel,:,:);
ladcp_out.u = U_sorted(indsel,:,:);
ladcp_out.v = V_sorted(indsel,:,:);
ladcp_out.depth = z_grid;

end