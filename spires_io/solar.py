# spires_io/solar.py
"""Sensor-neutral solar position (NOAA General Solar Position algorithm)."""
import numpy as np

_UNIX_EPOCH = np.datetime64("1970-01-01T00:00:00", "s")


def solar_zenith_angle(time, lat, lon):
    """Solar zenith angle in degrees at UTC ``time`` for geodetic ``lat``/``lon`` in degrees.

    ``lat`` and ``lon`` may be scalars or broadcastable arrays. ``time`` is a single
    instant (``str`` or ``numpy.datetime64``). Accuracy ~0.01 degree (NOAA spreadsheet
    algorithm), far below the 1 km MODIS geometry used so far. Values exceed 90 at night.
    """
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    lon = np.asarray(lon, dtype=np.float64)

    seconds = (np.datetime64(time, "s") - _UNIX_EPOCH) / np.timedelta64(1, "s")
    julian_day = seconds / 86400.0 + 2440587.5
    century = (julian_day - 2451545.0) / 36525.0

    mean_longitude = (280.46646 + century * (36000.76983 + century * 0.0003032)) % 360.0
    mean_anomaly = 357.52911 + century * (35999.05029 - 0.0001537 * century)
    eccentricity = 0.016708634 - century * (0.000042037 + 0.0000001267 * century)

    anomaly = np.radians(mean_anomaly)
    equation_of_center = (
        np.sin(anomaly) * (1.914602 - century * (0.004817 + 0.000014 * century))
        + np.sin(2 * anomaly) * (0.019993 - 0.000101 * century)
        + np.sin(3 * anomaly) * 0.000289
    )
    true_longitude = mean_longitude + equation_of_center
    omega = np.radians(125.04 - 1934.136 * century)
    apparent_longitude = np.radians(true_longitude - 0.00569 - 0.00478 * np.sin(omega))

    mean_obliquity = 23.0 + (
        26.0 + (21.448 - century * (46.815 + century * (0.00059 - century * 0.001813))) / 60.0
    ) / 60.0
    obliquity = np.radians(mean_obliquity + 0.00256 * np.cos(omega))

    declination = np.arcsin(np.sin(obliquity) * np.sin(apparent_longitude))

    y = np.tan(obliquity / 2.0) ** 2
    mean_longitude_rad = np.radians(mean_longitude)
    equation_of_time_minutes = 4.0 * np.degrees(
        y * np.sin(2 * mean_longitude_rad)
        - 2 * eccentricity * np.sin(anomaly)
        + 4 * eccentricity * y * np.sin(anomaly) * np.cos(2 * mean_longitude_rad)
        - 0.5 * y**2 * np.sin(4 * mean_longitude_rad)
        - 1.25 * eccentricity**2 * np.sin(2 * anomaly)
    )

    minutes_utc = (seconds % 86400.0) / 60.0
    true_solar_minutes = (minutes_utc + equation_of_time_minutes + 4.0 * lon) % 1440.0
    hour_angle = np.radians(true_solar_minutes / 4.0 - 180.0)

    cos_zenith = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    return np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))


def per_pixel_sza_provider(acquisition_time):
    """Build a ``SensorSpec.sza_provider`` that computes SZA per pixel from a time reader.

    ``acquisition_time(scene_dir) -> np.datetime64`` is the spec's own reader. The
    returned callable has the provider signature ``(scene_dir, template) -> DataArray``
    and is shared by every sensor whose product ships no SZA band (Landsat, E84 S2).
    """
    from pyproj import Transformer
    import xarray as xr

    def provider(scene_dir, template):
        to_lonlat = Transformer.from_crs(template.rio.crs, "EPSG:4326", always_xy=True)
        xx, yy = np.meshgrid(template.x.values, template.y.values)
        lon, lat = to_lonlat.transform(xx, yy)
        sza = solar_zenith_angle(acquisition_time(scene_dir), lat, lon)
        return xr.DataArray(sza.astype(np.float32), dims=("y", "x"), coords={"y": template.y, "x": template.x})

    return provider
