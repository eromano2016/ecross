import pandas as pd
import sys
import numpy as np
from datetime import datetime
import pyomo.environ as pyo

def get_EPEX(country, year):
    # This function imports market electricity prices from Filemaker DB, access to Filemaker DB is not public
    db_host = "IP of DB"   #Not publicly available
    db_name = "Name of DB" #Not publicly available
    db_user = ""           #Not publicly available
    db_password = ""       #Not publicly available
    path_to_jar_driver ='/...'   #Not publicly available
    path_to_ODBC_driver ='/...'   #Not publicly available
  
    try: #run on Linux with jdbc driver for filemaker
        import jaydebeapi
        fm_uri = "jdbc:filemaker://" + db_host+ "/" + db_name
        fm_creds = [db_user, db_password]
        
        connection = jaydebeapi.connect("com.filemaker.jdbc.Driver",
                                fm_uri, fm_creds,
                                path_to_jar,
                                )
    except: #run on PC or MacOS with ODBC installed
        import pyodbc
        con_str = 'Driver='+path_to_ODBC_driver+';Server=' + db_host + ';Database=' + db_name + ';UID=' + db_user + ';PWD=' + db_password
        connection = pyodbc.connect(con_str)
        connection.setdecoding(pyodbc.SQL_CHAR, encoding='utf-8')
        connection.setdecoding(pyodbc.SQL_WCHAR, encoding='utf-8')
        connection.setencoding(encoding='utf-8')

    query = f'SELECT  date_stamp, price FROM SPOT_{country} where (YEAR(date_stamp)={year})'
    db = pd.read_sql(sql=query, con=connection)
    db.columns = ['Time', 'Price']
    db['Time'] = pd.to_datetime(db['Time'])
    db.set_index('Time', inplace=True)
    db = db.sort_index()
  
    # Generate full expected hourly index as some prices from DB can be missing
    start = pd.Timestamp(f'{year}-01-01 00:00:00')
    end = pd.Timestamp(f'{year + 1}-01-01 00:00:00') - pd.Timedelta(hours=1)
    full_index = pd.date_range(start=start, end=end, freq='h')
    db = db[~db.index.duplicated(keep='first')]
  
    # Reindex to full hourly index (fills missing timestamps with NaN)
    db = db.reindex(full_index)

    # Optional: print info about missing timestamps
    n_missing = db['Price'].isna().sum()
    if n_missing > 0:
        print(f"⚠️ Filled {n_missing} missing hourly prices with NaN for {country} in {year}.")
    else:
        print(f"✅ All {len(full_index)} hourly timestamps present for {year}.")
    connection.close()
    return db

def optimize_vehicle_charging_day(
    prix_FR_day, prix_CH_day, trajet_km, timestamps_day, SoC_initial, charge_profile
):
  
    # Parameters of battery and scenarios regaring charging and discharing station
    Capacité = 65  # kWh Maximum battery capacity
    c_max_maison = charge_profile["home_charging"]  # kW
    c_max_travail = charge_profile["work_charging"]  # kW
    d_max_maison = charge_profile["home_discharge"]  # kW
    d_max_travail = charge_profile["work_discharge"]  # kW

    #Efficiency of battery when charging and discharing
    eta_c = 0.90 
    eta_d = 0.90 

    # Conversion assumption about energy needs per 100 km → kWh (18 kWh / 100 km) assumption on consumption
    trajet_energie = trajet_km * 0.18 * 2  # (considering back and forth trip)

    if trajet_energie > Capacité:
        print(f"⚠️ Trip energy ({trajet_energie:.1f} kWh) exceeds battery capacity ({Capacité} kWh). Skipping.")
        return None, float('inf'), SoC_initial, None

    T = len(prix_FR_day)
    model = pyo.ConcreteModel()
    model.T = pyo.RangeSet(0, T - 1)

    #Define hours when cars are at home or at work
    def est_maison(t):
        h = t % 24
        return h >= 18 or h < 8

    def est_travail(t):
        h = t % 24
        return 8 <= h <= 17

    # Variables of the Pyomo optimisation models
    model.c = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # kW
    model.d = pyo.Var(model.T, domain=pyo.NonNegativeReals)  # kW
    model.SoC = pyo.Var(model.T, bounds=(0, Capacité))       # kWh

    # Objective: minimize gross charging cost
    def cout_total(m):
        return sum(
            (m.c[t] - m.d[t]) * (prix_CH_day[t] if est_travail(t) else prix_FR_day[t])
            for t in m.T
        )
    model.obj = pyo.Objective(rule=cout_total, sense=pyo.minimize)

    # Add Constraints
    soc_min = 0.15 * Capacité #Minimum state of charing of the batery at each hour
    model.min_soc = pyo.Constraint(model.T, rule=lambda m, t: m.SoC[t] >= soc_min)

    # With driving profile over the day (normalized) from OFS statistics
    # define availability of the car to be charged (1-prob of drive_profile)
  
    drive_profile = [
        0.2, 0.2, 0.2, 0.4, 1.4, 5, 12.5, 11.3, 5.5, 3.1, 2.6, 5.0,
        6.2, 5.4, 3.0, 4.2, 7.9, 10.6, 7.1, 3.3, 1.9, 1.4, 1.2, 0.7
    ]
  
    #Check if drive_profile is a normalized over the day
    drive_profile = np.array(drive_profile, dtype=float)
    drive_profile /= drive_profile.sum()
    p_raw = drive_profile[:]  
  
    if max(p_raw) > 1.0:  # looks like percentages
        avail = [max(0.0, 1.0 - x / 100.0) for x in p_raw]
    else:  # already 0..1 probabilities
        avail = [max(0.0, 1.0 - x) for x in p_raw]

    # Make the driving profile a Pyomo Param over T (assumes model.T are hours 0..23)
    model.avail = pyo.Param(model.T, initialize=lambda m, t: avail[t],
                            within=pyo.UnitInterval, mutable=False)

    # Scaling for the limits of charging and discharghing capacity by availability
    def limite_charge(m, t):
        if est_maison(t):
            return m.c[t] <= c_max_maison * m.avail[t]
        elif est_travail(t):
            return m.c[t] <= c_max_travail * m.avail[t]
        else:
            return m.c[t] == 0

    model.limite_c = pyo.Constraint(model.T, rule=limite_charge)

    def limite_decharge(m, t):
        if est_maison(t):
            return m.d[t] <= d_max_maison * m.avail[t]
        elif est_travail(t):
            return m.d[t] <= d_max_travail * m.avail[t]
        else:
            return m.d[t] == 0

    model.limite_d = pyo.Constraint(model.T, rule=limite_decharge)
    
    # Fix initial SoC
    model.SoC[0].fix(SoC_initial)

   # Define SoC dynamic equation (charged energy - discharged energy - driven_energy) for each hour

    def dynamique_soc(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        h = t % 24
        drive_energy = trajet_energie * drive_profile[h]  # kWh this hour
        # SoC[t] = SoC[t-1] + eta_c * c - d/eta_d - drive
        return m.SoC[t] == m.SoC[t - 1] + eta_c * m.c[t] - m.d[t] / eta_d - drive_energy

    model.dyn_soc = pyo.Constraint(model.T, rule=dynamique_soc)

    # Solve the model
    solver = pyo.SolverFactory('cbc')
    results = solver.solve(model, tee=False)

    ok = (results.solver.status == pyo.SolverStatus.ok) and \
         (results.solver.termination_condition == pyo.TerminationCondition.optimal)
    if not ok:
        print("Optimization failed or model infeasible.")
        return None, float('inf'), SoC_initial, None

    # Extract values
    c_vals = [pyo.value(model.c[t]) for t in model.T]
    d_vals = [pyo.value(model.d[t]) for t in model.T]
    soc_vals = [pyo.value(model.SoC[t]) for t in model.T]
    cost_vals = sum([pyo.value(model.c[t])*prix_FR_day[t] for t in model.T])
    cost_home = sum(
        pyo.value(model.c[t]) * prix_FR_day[t]
        for t in model.T
        if est_maison(t)
    )
    cost_work = sum(
        pyo.value(model.c[t]) * prix_CH_day[t]
        for t in model.T
        if est_travail(t)
    )

    resultats = pd.DataFrame({
        'time': timestamps_day,
        'charge_kW': c_vals,
        'decharge_kW': d_vals,
        'SoC_kWh': soc_vals,
    })

    # By-location volumes (kWh over the day; dt=1h)
    charge_maison = 0.0
    charge_travail = 0.0
    decharge_maison = 0.0
    decharge_travail = 0.0

    for t in range(T):
        h = t % 24
        if h >= 18 or h < 8:
            charge_maison += c_vals[t]
            decharge_maison += d_vals[t]
        elif 8 <= h <= 17:
            charge_travail += c_vals[t]
            decharge_travail += d_vals[t]

    daily_summary = {
        'volume_charge_kWh': float(sum(c_vals)),
        'volume_discharge_kWh': float(sum(d_vals)),
        'travel_energie_kWh':float(trajet_energie),
        'charge_maison_kWh': float(charge_maison),
        'charge_travail_kWh': float(charge_travail),
        'decharge_maison_kWh': float(decharge_maison),
        'decharge_travail_kWh': float(decharge_travail),
        'daily_cost_CHF': float(pyo.value(model.obj)),  # rename if using EUR
        'travel_cost_CHF': float((cost_work+cost_home)*trajet_energie/(charge_maison+charge_travail)),  # rename if using EUR
        'travel_cost_FR_CHF': float((cost_home) * trajet_energie / (charge_maison + charge_travail)),  # rename if using EUR
        'travel_cost_CH_CHF': float((cost_work) * trajet_energie / (charge_maison + charge_travail)), # rename if using EUR
        'daily_bene_CHF': (float(pyo.value(model.obj))-float((cost_work+cost_home)*trajet_energie/(charge_maison+charge_travail)))*-1,  # rename if using EUR
    }

    final_soc = float(pyo.value(model.SoC[T - 1]))

    

    
    return resultats, daily_summary['daily_cost_CHF'], final_soc, daily_summary
    
