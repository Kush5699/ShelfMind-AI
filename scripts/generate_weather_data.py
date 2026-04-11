"""
Dataset 4: Weather Data Generator
Fetches historical weather data from Open-Meteo API (free, no key needed)
for Walmart store locations matching M5 dataset states (CA, TX, WI).
"""

import requests
import pandas as pd
from datetime import datetime
import os
import time

# Store locations matching M5 dataset states
LOCATIONS = {
    "CA": {"lat": 34.05, "lon": -118.24, "name": "Los Angeles, CA"},
    "TX": {"lat": 30.27, "lon": -97.74, "name": "Austin, TX"},
    "WI": {"lat": 43.07, "lon": -89.40, "name": "Madison, WI"},
}

# Match M5 dataset date range
START_DATE = "2011-01-29"
END_DATE = "2016-06-19"

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "weather")


def fetch_weather(state_id: str, lat: float, lon: float) -> pd.DataFrame:
    """Fetch daily weather from Open-Meteo Historical API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "precipitation_sum",
            "rain_sum",
            "snowfall_sum",
            "windspeed_10m_max",
        ]),
        "timezone": "America/Chicago",
    }

    print(f"  Fetching weather for {state_id} ({lat}, {lon})...")
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    daily = data["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "state_id": state_id,
        "temp_max": daily["temperature_2m_max"],
        "temp_min": daily["temperature_2m_min"],
        "temp_mean": daily["temperature_2m_mean"],
        "precipitation_mm": daily["precipitation_sum"],
        "rain_mm": daily["rain_sum"],
        "snowfall_cm": daily["snowfall_sum"],
        "wind_max_kmh": daily["windspeed_10m_max"],
    })
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_dfs = []
    for state_id, loc in LOCATIONS.items():
        df = fetch_weather(state_id, loc["lat"], loc["lon"])
        all_dfs.append(df)
        time.sleep(1)  # Be nice to the free API

    weather_df = pd.concat(all_dfs, ignore_index=True)

    # Save combined
    out_path = os.path.join(OUTPUT_DIR, "historical_weather.csv")
    weather_df.to_csv(out_path, index=False)
    print(f"\n[OK] Weather data saved to {out_path}")
    print(f"   Shape: {weather_df.shape}")
    print(f"   Date range: {weather_df['date'].min()} to {weather_df['date'].max()}")
    print(f"   States: {weather_df['state_id'].unique().tolist()}")

    # Save per-state files too
    for state_id in weather_df["state_id"].unique():
        state_df = weather_df[weather_df["state_id"] == state_id]
        state_path = os.path.join(OUTPUT_DIR, f"weather_{state_id}.csv")
        state_df.to_csv(state_path, index=False)
        print(f"   Saved {state_path} ({len(state_df)} rows)")


if __name__ == "__main__":
    print("=" * 50)
    print("ShelfMind AI - Weather Data Generator")
    print("=" * 50)
    main()
