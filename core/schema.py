from pathlib import Path
import math
ROOT=Path(__file__).resolve().parents[1]
FEATURES=['temperature_2_m_above_gnd','relative_humidity_2_m_above_gnd','mean_sea_level_pressure_MSL','total_precipitation_sfc','snowfall_amount_sfc','total_cloud_cover_sfc','high_cloud_cover_high_cld_lay','medium_cloud_cover_mid_cld_lay','low_cloud_cover_low_cld_lay','shortwave_radiation_backwards_sfc','wind_speed_10_m_above_gnd','wind_direction_10_m_above_gnd','wind_speed_80_m_above_gnd','wind_direction_80_m_above_gnd','wind_speed_900_mb','wind_direction_900_mb','wind_gust_10_m_above_gnd','angle_of_incidence','zenith','azimuth']
TARGET='generated_power_kw'
LABELS=['Air temperature · 2 m','Relative humidity · 2 m','Sea-level pressure','Total precipitation','Snowfall amount','Total cloud cover','High cloud cover','Medium cloud cover','Low cloud cover','Shortwave radiation','Wind speed · 10 m','Wind direction · 10 m','Wind speed · 80 m','Wind direction · 80 m','Wind speed · 900 mb','Wind direction · 900 mb','Wind gust · 10 m','Angle of incidence','Solar zenith','Solar azimuth']
UNITS=['source temperature units','%','source pressure units','source precipitation units','source snowfall units','%','%','%','%','source radiation units','source wind units','°','source wind units','°','source wind units','°','source wind units','°','°','°']

def validate_features(payload):
    if not isinstance(payload,dict):raise ValueError('features must be an object with all 20 column names')
    missing=[x for x in FEATURES if x not in payload]
    if missing:raise ValueError('Missing columns: '+', '.join(missing))
    values=[]
    for name in FEATURES:
        if isinstance(payload[name],bool):raise ValueError(name+' must be numeric')
        try:v=float(payload[name])
        except (ValueError,TypeError):raise ValueError(name+' must be numeric')
        if not math.isfinite(v) or abs(v)>1e8:raise ValueError(name+' must be finite and within numeric limits')
        values.append(v)
    return values
