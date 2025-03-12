# Band constants
RGB_BANDS = ['R', 'G', 'B']
IR_BANDS = ['NIR', 'RED_EDGE', 'COASTAL']
RED_BAND = 'R'
NIR_BAND = 'NIR'

ALTITUDE_BAND = 'altitude'
ASPECT_BAND = 'aspect'
SLOPE_BAND = 'slope'

# SRTM constants for feature scaling (computed from train_latlon_ls8_dynamic.csv)
ALTITUDE_MIN = -23.0
ALTITUDE_MAX = 4228.0
SLOPE_MIN = 0.0
SLOPE_MAX = 8421.0
ASPECT_MIN = -17954.0
ASPECT_MAX = 18000.0

SRTM_SCALING = {
    ALTITUDE_BAND: (ALTITUDE_MIN, ALTITUDE_MAX),
    SLOPE_BAND: (SLOPE_MIN, SLOPE_MAX),
    ASPECT_BAND: (ASPECT_MIN, ASPECT_MAX)
}

NCEP_TEMP_AVG = 'tavg'
NCEP_TEMP_MAX = 'tmax'
NCEP_TEMP_MIN = 'tmin'
NCEP_TEMP = 'temp'
NCEP_TEMP_BANDS = [NCEP_TEMP_AVG, NCEP_TEMP_MAX, NCEP_TEMP_MIN]

ALBEDO_BAND = 'albedo'
DOWN_LONG_FLUX_BAND = 'clear-sky_downward_longwave_flux'
DOWN_SOLAR_FLUX_BAND = 'clear-sky_downward_solar_flux'
UP_LONG_FLUX_BAND = 'clear-sky_upward_longwave_flux'
UP_SOLAR_FLUX_BAND = 'clear-sky_upward_solar_flux'
EVAP_BAND = 'direct_evaporation_bare_soil'
DOWN_LONG_RAD_FLUX_BAND = 'downward_longwave_radiation_flux'
UP_LONG_RAD_FLUX_BAND = 'upward_longwave_radiation_flux'
DOWN_SHORT_RAD_FLUX_BAND = 'downward_shortwave_radiation_flux'
UP_SHORT_RAD_FLUX_BAND = 'upward_shortwave_radiation_flux'
GROUND_HEAT_BAND = 'ground_heat_net_flux'
LATENT_HEAT_BAND = 'latent_heat_net_flux'
MAX_HUMIDITY_BAND = 'max_specific_humidity'
MIN_HUMIDITY_BAND = 'min_specific_humidity'
POTENTIAL_EVAP_BAND = 'potential_evaporation_rate'
PREC_BAND = 'prec'
SENSIBLE_HEAT_BAND = 'sensible_heat_net_flux'
SOIL_MOIST1_BAND = 'soilmoist1'
SOIL_MOIST2_BAND = 'soilmoist2'
SOIL_MOIST3_BAND = 'soilmoist3'
SOIL_MOIST4_BAND = 'soilmoist4'
HUMIDITY_BAND = 'specific_humidity'
SUBLIMATION_BAND = 'sublimation'
SURFACE_PRESSURE_BAND = 'surface_pressure'
U_WIND_BAND = 'u_wind_10m'
V_WIND_BAND = 'v_wind_10m'
WATER_RUNOFF_BAND = 'water_runoff'

NCEP_BANDS = [
    ALBEDO_BAND,
    DOWN_LONG_FLUX_BAND,
    DOWN_SOLAR_FLUX_BAND,
    UP_LONG_FLUX_BAND,
    UP_SOLAR_FLUX_BAND,
    EVAP_BAND,
    DOWN_LONG_RAD_FLUX_BAND,
    DOWN_SHORT_RAD_FLUX_BAND,
    GROUND_HEAT_BAND,
    LATENT_HEAT_BAND,
    MAX_HUMIDITY_BAND,
    MIN_HUMIDITY_BAND,
    POTENTIAL_EVAP_BAND,
    PREC_BAND,
    SENSIBLE_HEAT_BAND,
    SOIL_MOIST1_BAND,
    SOIL_MOIST2_BAND,
    SOIL_MOIST3_BAND,
    SOIL_MOIST4_BAND,
    HUMIDITY_BAND,
    SUBLIMATION_BAND,
    SURFACE_PRESSURE_BAND,
    NCEP_TEMP_AVG,
    NCEP_TEMP_MAX,
    NCEP_TEMP_MIN,
    U_WIND_BAND,
    UP_LONG_RAD_FLUX_BAND,
    UP_SHORT_RAD_FLUX_BAND,
    V_WIND_BAND,
    WATER_RUNOFF_BAND
]

# NCEP constants for feature scaling (computed from train_latlon_ls8_dynamic.csv)
NCEP_SCALING = {
    ALBEDO_BAND: (268., 1103.),

    DOWN_LONG_FLUX_BAND: (274., 434.),
    UP_LONG_FLUX_BAND: (366., 503.),

    DOWN_SOLAR_FLUX_BAND: (233., 346.),
    UP_SOLAR_FLUX_BAND: (13., 48.),

    EVAP_BAND: (0., 262.),

    DOWN_LONG_RAD_FLUX_BAND: (276., 449.),
    DOWN_SHORT_RAD_FLUX_BAND: (5., 345.),

    GROUND_HEAT_BAND: (-37., 21.),
    LATENT_HEAT_BAND: (-2., 789.),

    HUMIDITY_BAND: (49., 222.),
    MAX_HUMIDITY_BAND: (7., 405.),
    MIN_HUMIDITY_BAND: (7., 405.),

    POTENTIAL_EVAP_BAND: (0., 2171.),

    PREC_BAND: (0., 3218.),

    SENSIBLE_HEAT_BAND: (-604., 239.),

    SOIL_MOIST1_BAND: (0., 4550.),
    SOIL_MOIST2_BAND: (0., 4525.),
    SOIL_MOIST3_BAND: (0., 4523.),
    SOIL_MOIST4_BAND: (0., 4604.),

    SURFACE_PRESSURE_BAND: (7553., 10216.),

    U_WIND_BAND: (-968., 1215.),
    V_WIND_BAND: (-1148., 1065.),

    WATER_RUNOFF_BAND: (0., 29115.),

    SUBLIMATION_BAND: (0., 11.),

    NCEP_TEMP: (27286., 31693.),

    UP_LONG_RAD_FLUX_BAND: (366., 502.),
    UP_SHORT_RAD_FLUX_BAND: (0., 43.),
}

# OSM constants for feature scaling in km (computed from train_latlon_ls8_dynamic.csv)
STREET_MIN = 0.00327
CITY_MIN = 0.19590
STREET_MAX = 513.49534
CITY_MAX = 513.49534

OSM_SCALING = {
    'city': (CITY_MIN, CITY_MAX),
    'street': (STREET_MIN, STREET_MAX)
}


