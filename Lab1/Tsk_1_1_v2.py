import pandas as pd
import numpy as np
import os

# List of CSV URLs corresponding to three different heights
csv_files = {
    'LOW' :'https://raw.githubusercontent.com/aarnezanocco/civil-450-group-project/main/HOBO/Group3-Low.csv',
    'MID' :'https://raw.githubusercontent.com/aarnezanocco/civil-450-group-project/main/HOBO/Group3-Mid.csv',
    'TOP' :'https://raw.githubusercontent.com/aarnezanocco/civil-450-group-project/main/HOBO/Group3-Top.csv'
}

# Constants
SEUIL_VA = 0.05  # Threshold for airspeed to determine the convection type
OP_VA = 0.2        # Threshold for airspeeed to select the appropriate Operative temperature formula
SIGMA = 5.67e-8  # Stefan-Boltzmann constant
D = 0.04           # Diameter of the Globe (mm)
epsilon_g = 0.95  # Emissivity of the Globe

# Function to compute MRT
def compute_mrt(tg, ta, va, D):
    if va < SEUIL_VA:  # Natural convection
        return ((tg + 273) ** 4 + (0.25e8 / (epsilon_g * D)) * ((tg - ta) ** (1/4)) * (tg - ta)) ** (1/4) - 273
    else:  # Forced convection
        return ((tg + 273) ** 4 + (1.1e8 * va ** 0.6 / (epsilon_g * D ** 0.4)) * (tg - ta)) ** (1/4) - 273

#function to compute LOT
def compute_LOT(ta, va, mrt):
    if va < OP_VA :
        return (ta + mrt) / 2 
    elif va < 0.6 :
        A = 0.6
        return A * ta + (1-A)* mrt
    else :     #elif va < 1 :  (before)
        A = 0.7
        return A * ta + (1-A)* mrt
    
    # else : (before)
    #     print("Error : air speed out of bouds", va)
    #     return np.nan #for safety

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

    #Compute LOT
    LOT_values = np.array([compute_LOT(ta_i, va_i, mrt_i) for ta_i, va_i, mrt_i in zip(ta, va, mrt_values)])
       
    mot_values = np.mean(LOT_values)  #Calculation for MOT

    # Store results in a DataFrame
    df_results = pd.DataFrame({
        'Date & Time' : date_time,
        'T_air': ta,
        'T_globe': tg,
        'Air_speed': va,
        'T_MRT': mrt_values,
        'T_OP': LOT_values,
        #'MOT': mot_values   not good dimension
        'T_MOP': np.array([mot_values] * len(ta))  # Repeat MOT value for all rows, for dimensions compatibility
    })
    
    results[height] = df_results

# Save results to CSV files
for height, df in results.items():
    df.to_csv(f'results_{height}.csv', index=False)

print("Calculations completed for all heights.")

#ka = 0.026  # Thermal conductivity of air
#rho_a = 1.225  # Air density at 20°C
#mu_a = 1.81e-5  # Dynamic viscosity of air

# Function to compute h_conv based on convection type
#def compute_h_conv(tg, ta, va, D):
#    if va < SEUIL_VA:
#        return 1.4 * ((tg - ta) / D) ** (1/4)  # Equation 4-9 (natural convection)
#    else:
#        return 0.32 * ka * ((rho_a / mu_a) ** 0.6) * ((va ** 0.6) / (D ** 0.4))  # Equation 4-10a (forced convection)

#compute h_conv values
#h_conv_values = [compute_h_conv(tg_i, ta_i, va_i, D) for tg_i, ta_i, va_i in zip(tg, ta, va)]

#in df results:
#'h_conv': h_conv_values,        'h_conv': h_conv_values,
