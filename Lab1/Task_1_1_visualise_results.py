import pandas as pd
import matplotlib.pyplot as plt

# Define the heights and corresponding CSV files
heights = ['LOW', 'MID', 'TOP']
excel_file = 'results_presentation_simple.xlsx'

# Create a Pandas Excel writer using openpyxl as the engine.
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    for height in heights:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(f'results_{height}.csv')
        
        # Write the DataFrame to a specific sheet in the Excel file
        df.to_excel(writer, sheet_name=height, index=False)
        
        # Create a plot for MRT, LOT, and MOT against Date & Time
        plt.figure(figsize=(12, 6))  # Set the figure size

        # Extracting Date & Time and the last three columns
        date_time = pd.to_datetime(df['Date & Time'])  # Use the 'Date & Time' column
        mrt = df['T_MRT']
        lot = df['T_OP']
        mot = df['T_MOP']

        # Plotting
        plt.plot(date_time, mrt, marker='o', label='T_MRT', color='blue')
        plt.plot(date_time, lot, marker='x', label='T_OP', color='orange')
        plt.plot(date_time, mot, marker='s', label='T_MOP', color='green')

        # Adding titles and labels
        plt.title(f'T_MRT, T_OP, and T_MOP vs Date & Time at {height} Height')
        plt.xlabel('Date & Time')
        plt.ylabel('Temperature (°C)')
        plt.legend()
        plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
        plt.grid()

        # Save the figure
        plt.savefig(f'plot_{height}.png')  # Save each plot as a PNG file
        plt.close()  # Close the plot to free memory

print(f"Results have been written to {excel_file} and plots have been saved.")
