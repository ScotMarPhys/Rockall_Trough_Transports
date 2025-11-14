filter_length = 90     # (days)
corr_model = 0.0759    # Correction parameter (m/s) to adjust GLORYS DAC V component to the 8 month DAC from ADCP
z_cutoff_EW = 750     # Depth (m) where transition from GLORYS-ADCP data to CM data at Eastern wedge
z_cutoff_WW = 250     # Depth (m) limit of upper-left corner cutout
lon_cutoff_WW = -12.906  # Longitude limit of upper-left corner cutout
NX_WW = 100            # Horizontal grid size in Western wedge
NX_MB = 100           # Horizontal grid size in Mid-basin (needed for Ekman calc.)
NX_EW = 100            # Horizontal grid size in Eastern wedge
SA_ref = 35.34         # Reference Absolute Salinity (g/kg) defined as mean at WB1
CT_ref = 7.14          #  Reference Conservative Temperature (C) defined as mean at WB1
Cp = 3991           # Constant:Specific heat capacity (J kg^-1 C^-1) defined as max, 1 atmos
rho0 = 1027.5            # Reference density kg m^-3 defined as mean
rhoCp = Cp*rho0           # Constant: Reference density times specific heat capacity (J m^-3 C^-1)       
Omega = 7.2921e-5     # Earth's rotation

# Praemble for despike time series after vertical extrapolation to surface - velocity
stddy_tol  = 10; # set max velocities range (diff velocity pm stddy_tol)
std_win    = 3.5; # 3.5 * std of the time series (median pm std_win)
nloop    = 5; # max number of despiking repetitions