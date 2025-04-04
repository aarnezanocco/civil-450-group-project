import pandas as pd

# Define the heights and corresponding CSV files
heights = ['LOW', 'MID', 'TOP']
excel_file = 'results_presentation_simple.xlsx'

# Create a Pandas Excel writer using XlsxWriter as the engine.
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    for height in heights:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(f'results_{height}.csv')
        
        # Write the DataFrame to a specific sheet in the Excel file
        df.to_excel(writer, sheet_name=height, index=False)

print(f"Results have been written to {excel_file}.")
