import pandas as pd
import glob
import os


def merge_paths(data_folder):

    lst = os.listdir(data_folder)

    data_paths = []
    for data in lst:
        print(data)
        path = glob.glob(os.path.join(data_folder, data, "*.xlsx"))
        if len(path) != 0:

            data_paths.append(path)
        print(path)
    
    return data_paths

def data_merger(data_paths, data_saving_path):
    DATE_COLUMN_NAME = 'Date'

    master_dfs_to_join = []
    total_groups_processed = 0

    print("Starting group combination")

    for group_index, group_paths in enumerate(data_paths):
        group_dfs_list = []
        print(f"Processing Group {group_index + 1} with {len(group_paths)} files...")

        for file_path in group_paths:
            normalized_path = os.path.normpath(file_path)
            try:
                df = pd.read_excel(normalized_path, engine='openpyxl')
                group_dfs_list.append(df)
            except Exception as e:
                print(f"Error reading '{normalized_path}': {e}. Skipping.")

        if not group_dfs_list:
            print(f"Skipping Group {group_index + 1}: No files read successfully.")
            continue

        master_group_df = pd.concat(group_dfs_list, ignore_index=True)
        
        try:
            master_group_df[DATE_COLUMN_NAME] = pd.to_datetime(
                master_group_df[DATE_COLUMN_NAME], errors='coerce'
            )        
            master_group_df = master_group_df.dropna(subset=[DATE_COLUMN_NAME])
            master_group_df = master_group_df.set_index(DATE_COLUMN_NAME)
            
            master_dfs_to_join.append(master_group_df)
            total_groups_processed += 1
            print(f"Group {group_index + 1} appended and prepared for join (Rows: {len(master_group_df)}).")

        except KeyError:
            print(f"ERROR in Group {group_index + 1}!")
            print(f"Date column '{DATE_COLUMN_NAME}' not found. Please check column headers.")
            
    if total_groups_processed < 2:
        print("Cannot perform a join: Less than two metric groups were successfully processed.")
    else:
        
        final_combined_df = pd.concat(master_dfs_to_join, axis=1, join='outer')
        final_combined_df = final_combined_df.reset_index().rename(columns={'index': DATE_COLUMN_NAME})
        
        final_combined_df = final_combined_df.sort_values(by=DATE_COLUMN_NAME).reset_index(drop=True)

        OUTPUT_FILE_NAME = os.path.join(data_saving_path, 'Combined_Time_Series_Joined_Data.csv')
        final_combined_df.to_csv(OUTPUT_FILE_NAME, index=False)

        print(f"Final joined data saved to '{OUTPUT_FILE_NAME}'.")
        print(f"Final shape: {final_combined_df.shape}")
        print(f"Note: Missing values (NaN) in the output indicate a timestamp where a specific metric had no data.")




