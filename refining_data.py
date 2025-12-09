import os
import pandas as pd


def data_refiner(data, data_saving_path):
    columns_to_drop = [
        'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 
        'Unnamed: 5.1', 'Unnamed: 6.1', 'Unnamed: 7.1', 'Unnamed: 8.1',
        'Cooling load (GJ)', 'Heating load (GJ)', 'Hot water load (GJ)',
        'Cooling load (GJ).1', 'Heating load (GJ).1', 'Hot water load (GJ).1',
        'Boiler (m3)', 'Fuel cell (m3)', 'Gas engine (m3)',
        'Absorption chiller 1 (m3)', 'Absorption chiller 2 (m3)', 'Absorption chiller 3 (m3)'
    ]

    renaming_map = {
        'Solar energy generation (kW)': 'PV_Generation_kW',
        'Electricity imported from grid (kW)': 'Grid_Import_kW',
        'Gas engine (kW)': 'GasEngine_Generation_kW',
        'Fuel cell (kW)': 'FuelCell_Generation_kW',        
        'Electricity load (kW)': 'Electric_Load_A_kW',      
        'Cooling load (kW)': 'Cooling_Load_A_kW',
        'Heating load (kW)': 'Heating_Load_A_kW',
        'Hot water load (kW)': 'HotWater_Load_A_kW',
        'Electricity load (kW).1': 'Electric_Load_B_kW', 
        'Cooling load (kW).1': 'Cooling_Load_B_kW',
        'Heating load (kW).1': 'Heating_Load_B_kW',
        'Hot water load (kW).1': 'HotWater_Load_B_kW',
        'Horizontal solar irradition (W)': 'Solar_Irradiation_W',
        'Outdoor air temperature (℃)': 'Outdoor_Air_Temp_C',
        'Outdoor air humidity (%)': 'Outdoor_Air_Humidity_percent',
        'Wind speed (m/s)': 'Wind_Speed_ms',
        'Wind direction': 'Wind_Direction_deg',
        'Gas 1(GJ/h)': 'Gas1_Input_GJh',
        'Gas 2(GJ/h)': 'Gas2_Input_GJh',
        'Fuel cell 1(GJ/h)': 'FuelCell1_Input_GJh',
        'Fuel cell 2(GJ/h)': 'FuelCell2_Input_GJh',
        'Fuel cell 3(GJ/h)': 'FuelCell3_Input_GJh'
    }


    final_combined_df = data.drop(columns=columns_to_drop, errors='ignore')
    print("Dropped empty/sparse/ambiguous columns.")

    final_combined_df = final_combined_df.rename(columns=renaming_map)
    print("Renamed remaining columns for clarity.")

    final_combined_df['Date'] = pd.to_datetime(final_combined_df['Date'], errors='coerce')
    initial_rows = len(final_combined_df)
    final_combined_df = final_combined_df.dropna(subset=['Date'])
    print(f"Removed {initial_rows - len(final_combined_df)} rows with missing or invalid 'Date'.")

    CLEANED_OUTPUT_FILE = os.path.join(data_saving_path, 'Cleaned_Combined_Time_Series_Data.csv')
    final_combined_df.to_csv(CLEANED_OUTPUT_FILE, index=False)

    print(f"Cleaned data saved to '{CLEANED_OUTPUT_FILE}'.")
    print(f"Final DataFrame Shape: {final_combined_df.shape}")
    print(final_combined_df.info())

