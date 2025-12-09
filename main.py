
import os
import pandas as pd
from merging_data import data_merger, merge_paths
from refining_data import data_refiner


def main():
    data_save_path = 'Data_Output/'
    os.makedirs(data_save_path, exist_ok=True)
    data_folder = 'Project_data/' 
    data_paths = merge_paths(data_folder)
    data_merger(data_paths, data_save_path)

    data = pd.read_csv(os.path.join(data_save_path, 'Combined_Time_Series_Joined_Data.csv'))
    data_refiner(data, data_save_path)

if __name__ == "__main__":
    main()