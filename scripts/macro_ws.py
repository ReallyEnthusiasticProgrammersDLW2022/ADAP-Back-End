# Initial Imports
import pandas as pd
import numpy as np

# Constants
theme_weights = {
    "education": 1,
    "entertainment": 1,
    "financial": 1,
    "healthcare": 1,
    "public service": 2,
    "sustenance": 3,
}

# Functions
def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))

def distance_to_attenuation(distance):
  if distance < 0.4:
    return 1
  elif distance < 0.800:
    return 0.9
  elif distance < 1.2:
    return 0.55
  elif distance < 1.6:
    return 0.25
  else:
    return 0.08

def macro_ws(lat, lng):
    
  poi_df = pd.read_csv("data/sg_poi.csv")
  poi_df = poi_df.loc[poi_df["theme"] != "transport"]
  
  lat1 = pd.Series([lat] * len(poi_df))
  lon1 = pd.Series([lng] * len(poi_df))
  lat2 = poi_df["lat"]
  lon2 = poi_df["lon"]

  point_df = poi_df.copy()
  point_df["distance"] = haversine(lat1, lon1, lat2, lon2)
  point_df = point_df[point_df["distance"] <= 2]
  point_df["attenuation"] = point_df["distance"].apply(distance_to_attenuation)
  point_df["weights"] = point_df["theme"].map(theme_weights)
  point_df["macro"] = point_df["attenuation"] * point_df["weights"]
  sum = point_df["macro"].sum()
  return sum
