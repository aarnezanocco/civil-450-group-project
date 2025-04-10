import pandas as pd
import numpy as np
import os
import nbimporter
from Task2_functions import iterate_T_cl

# Constants
SEUIL_VA = 0.05   # Threshold for airspeed to determine the convection type
OP_VA = 0.2       # Threshold for airspeeed to select the appropriate Operative temperature formula
SIGMA = 5.67e-8   # Stefan-Boltzmann constant
D = 0.04          # Diameter of the Globe (mm)
EPSILON_G = 0.95  # Emissivity of the Globe
EPSILON_C = 0.95  # Emissivity of the clothing enable surface
F_EFF = 0.72      # Effective radiation area factor of a sitting person 
I_CL = 0.38       # Clothing insulation #For T_cl

# Load skin temperatures from 8 files in 'CALERA' folder
calera_folder = 'CALERA'
skin_temp_series = []

for filename in os.listdir(calera_folder):
    if filename.endswith('.xlsx'):
        filepath = os.path.join(calera_folder, filename)
        df = pd.read_excel(filepath)
        skin_temp_series.append(df.iloc[:, 2])  # 3rd column is index 2

# Combine and average the 8 skin temp series
t_skin_df = pd.concat(skin_temp_series, axis=1)
t_skin_avg = t_skin_df.mean(axis=1)

# List of CSV URLs corresponding to three different heights
csv_files = {
    'LOW' :'https://raw.githubusercontent.com/aarnezanocco/civil-450-group-project/main/HOBO/Group3-Low.csv',
    'MID' :'https://raw.githubusercontent.com/aarnezanocco/civil-450-group-project/main/HOBO/Group3-Mid.csv',
    'TOP' :'https://raw.githubusercontent.com/aarnezanocco/civil-450-group-project/main/HOBO/Group3-Top.csv'
}

# Function to compute MRT
def compute_mrt(tg, ta, va, D):
    if va < SEUIL_VA:  # Natural convection
        return ((tg + 273) ** 4 + (0.25e8 / (EPSILON_G * D)) * ((tg - ta) ** (1/4)) * (tg - ta)) ** (1/4) - 273
    else:  # Forced convection
        return ((tg + 273) ** 4 + (1.1e8 * va ** 0.6 / (EPSILON_G * D ** 0.4)) * (tg - ta)) ** (1/4) - 273

#Function to compute LOT
def compute_LOT(ta, va, mrt, t_cl):
    if va < OP_VA :
        return (ta + mrt) / 2   
    else :
        h_c = 8.3 * (va ** 0.6) 
        h_r = 4 * EPSILON_C * SIGMA * F_EFF * (273.15 + (t_cl + mrt) / 2) ** 3  
        A = h_c / (h_c + h_r)
        return A * ta + (1-A)* mrt

# Placeholder dictionary to store results
results = {}

# Process each CSV file
for height, url in csv_files.items():
    df = pd.read_csv(url, encoding='ISO-8859-1')
    
    # Extract necessary columns (assuming they are ordered correctly)
    ta = df.iloc[:, 1]  # Ambient temperature
    tg = df.iloc[:, 2]  # Globe temperature
    va = df.iloc[:, 3]  # Air speed
    date_time = pd.to_datetime(df.iloc[:, 0])  # Convert Date & Time to datetime

    # Compute MRT
    mrt_values = np.array([compute_mrt(tg_i, ta_i, va_i, D) for tg_i, ta_i, va_i in zip(tg, ta, va)])

    # Compute T_cl using Osman's function #Clothing temperature
    
    #t_cl = np.array([
        #iterate_T_cl(t_skin_i, ta_i, mrt_i, va_i, I_CL)
        #for t_skin_i, ta_i, mrt_i, va_i in zip(t_skin_series, ta, mrt_values, va)])

    t_cl, _, _, _ = iterate_T_cl(t_skin_avg, ta, mrt_values, I_CL, va, epsilon=0.95, max_iter=100, tol=0.01)

    #Compute LOT
    LOT_values = np.array([compute_LOT(ta_i, va_i, mrt_i, t_cl_i) for ta_i, va_i, mrt_i, t_cl_i in zip(ta, va, mrt_values, t_cl)])
       
    mot_values = np.mean(LOT_values)  #Calculation for MOT

    # Store results in a DataFrame
    df_results = pd.DataFrame({
        'Date & Time' : date_time,
        'T_air': ta,
        'T_globe': tg,
        'Air_speed': va,
        'T_MRT': mrt_values,
        'T_cl' : t_cl,
        'T_OP': LOT_values,
        'T_MOP': np.array([mot_values] * len(ta))  # Repeat MOT value for all rows, for dimensions compatibility
    })
    
    results[height] = df_results

# Save results to CSV files
for height, df in results.items():
    df.to_csv(f'results_{height}.csv', index=False)

print("Calculations completed for all heights.")
