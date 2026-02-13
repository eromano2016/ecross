import numpy as np
import pandas as pd
import methods
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(message)s")

# === PARAMETERS ===
file_path = ''
SoC_initial_default = 15
n_days = 365
ayear =2018 #Enter year for analysis (2018, 2020 or 2023)

charging_station = {
    "home_charging": 11,
    "work_charging": 22,
    "home_discharge": 11,
    "work_discharge": 22
}

# === LOAD TRIP DATA ===
trip = pd.read_csv(f'{file_path}trip_all_included.csv')
max_trips =len(trip)

# Load electricity prices for France and Switzerland
db = {
    'CH': methods.get_EPEX("CH", ayear),
    'FR': methods.get_EPEX("FR", ayear)
}
# Prices in EUR/MWh → convert to EUR/kWh
prix_FR_full = db['FR']['Price'].values/1000
prix_CH_full = db['CH']['Price'].values/1000
timestamps = db['FR'].index

# === MAIN LOOP ===
all_trip_results = []
global_cost = 0

#Process optimization for each pair of cities
for i, (trip_idx, trip_row) in enumerate(trip.head(max_trips).iterrows()):
    trajet_km = trip_row['distance']
    depart = trip_row['local_admin_unit_id_from']
    arrival = trip_row['local_admin_unit_id_to']
    nb_cars = trip_row['nb_vehicles_adj']
    SoC_initial = SoC_initial_default
    day_summaries = []  # reset per trip
    charge_load =[]
    logging.info(f"Processing trip {i+1}/{max_trips}: {depart} -> {arrival}")

    #Process optimization for each days (at hourly granularity)
    for day in range(n_days):
        start = day * 24
        end = start + 24

        prix_FR_day = prix_FR_full[start:end]
        prix_CH_day = prix_CH_full[start:end]
        timestamps_day = timestamps[start:end]

        results_day, cost_day, SoC_final_day, summary = methods.optimize_vehicle_charging_day(
            prix_FR_day, prix_CH_day,
            trajet_km=trajet_km,
            timestamps_day=timestamps_day,
            SoC_initial=SoC_initial,
            charge_profile=charging_station
        )

        if results_day is None:
            logging.warning(f"  Skipping trip {trip_idx}, day {day} due to infeasibility.")
            continue

        global_cost += cost_day
        SoC_initial = SoC_final_day

        # Add weekday/weekend label
        day_name = pd.to_datetime(timestamps_day[0]).day_name()
        month = pd.to_datetime(timestamps_day[0]).month

        summary['day_type'] = (
            'Saturday' if day_name == 'Saturday'
            else 'Sunday' if day_name == 'Sunday'
            else 'Weekday'
        )

        summary['season'] = (
            'Winter' if month in [12, 1, 2]
            else 'Spring' if month in [3, 4, 5]
            else 'Summer' if month in [6, 7, 8]
            else 'Autumn'
        )

        day_summaries.append(summary)

    # Create summary DataFrame for the trip
    if not day_summaries:
        logging.warning(f"No valid days for trip {trip_idx}")
        continue

    trip_summary = pd.DataFrame(day_summaries)
    trip_summary['trip_from'] = depart
    trip_summary['trip_to'] = arrival
    trip_summary['trip_distance'] = trajet_km
    trip_summary['trip_vehicles'] = nb_cars

    # Group by day_type
    trip_summary_by_season_day = trip_summary.groupby(['season', 'day_type']).agg({
        'trip_from': 'unique',
        'trip_to': 'unique',
        'trip_distance': 'mean',
        'trip_vehicles': 'mean',
        'travel_energie_kWh': 'mean',
        'volume_charge_kWh': ['mean', 'min', 'max'],
        'volume_discharge_kWh': ['mean', 'min', 'max'],
        'charge_maison_kWh': ['mean', 'min', 'max'],
        'charge_travail_kWh': ['mean', 'min', 'max'],
        'decharge_maison_kWh': ['mean', 'min', 'max'],
        'decharge_travail_kWh': ['mean', 'min', 'max'],
        'daily_cost_CHF': ['mean', 'min', 'max'],
        'daily_bene_CHF': ['mean', 'min', 'max'],
        'travel_cost_CHF': ['mean', 'min', 'max'],
        'travel_cost_FR_CHF': ['mean', 'min', 'max'],
        'travel_cost_CH_CHF': ['mean', 'min', 'max']

    }).reset_index()

    # Flatten multi-level column names
    trip_summary_by_season_day.columns = [
        '_'.join(col) if isinstance(col, tuple) else col
        for col in trip_summary_by_season_day.columns
    ]

    # Append to results
    all_trip_results.append(trip_summary_by_season_day)

# === COMBINE ALL RESULTS ===
if all_trip_results:
    final_result = pd.concat(all_trip_results, ignore_index=True)
    logging.info("\nSimulation complete. Final result:")
else:
    logging.warning("No feasible trips processed.")

# Optional: Save results
final_result.to_csv(
    f'trip_summary_results_trip_scenario_{ayear}_{max_trips}_{charging_station["home_charging"]}_{charging_station["work_charging"]}_{charging_station["home_discharge"]}_{charging_station["work_discharge"]}.csv',
    index=False
)
