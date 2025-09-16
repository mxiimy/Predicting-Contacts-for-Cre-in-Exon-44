import pandas as pd
import os

def create_detailed_contact_summary(unique_csv, repeat_csv, output_csv):
    """
    Reads two CSV files (unique and repeat contacts), counts the number
    of entries for each PDB ID in both, and saves a summary to a new CSV.

    Parameters:
    - unique_csv (str): Path to the CSV with unique contacts.
    - repeat_csv (str): Path to the CSV with repeat contacts.
    - output_csv (str): Path where the summary CSV will be saved.
    """
    # Create empty DataFrames to hold the counts
    unique_counts = pd.DataFrame()
    repeat_counts = pd.DataFrame()

    # Process the unique contacts file
    if os.path.exists(unique_csv):
        print(f"Reading unique contacts from '{unique_csv}'...")
        try:
            df_unique = pd.read_csv(unique_csv)
            if 'pdb_id' in df_unique.columns:
                unique_counts = df_unique['pdb_id'].value_counts().reset_index()
                unique_counts.columns = ['PDB_id', 'number of unique contacts']
        except Exception as e:
            print(f"Error reading unique contacts file: {e}")
    else:
        print(f"Warning: Unique contacts file '{unique_csv}' not found. Unique contact counts will be 0.")

    # Process the repeat contacts file
    if os.path.exists(repeat_csv):
        print(f"Reading repeat contacts from '{repeat_csv}'...")
        try:
            df_repeat = pd.read_csv(repeat_csv)
            if 'pdb_id' in df_repeat.columns:
                repeat_counts = df_repeat['pdb_id'].value_counts().reset_index()
                repeat_counts.columns = ['PDB_id', 'number of nonunique contacts']
        except Exception as e:
            print(f"Error reading repeat contacts file: {e}")
    else:
        print(f"Warning: Repeat contacts file '{repeat_csv}' not found. Non-unique contact counts will be 0.")

    # Merge the two count DataFrames based on PDB_id
    if not unique_counts.empty and not repeat_counts.empty:
        summary_df = pd.merge(unique_counts, repeat_counts, on='PDB_id', how='outer')
    elif not unique_counts.empty:
        summary_df = unique_counts
    elif not repeat_counts.empty:
        summary_df = repeat_counts
    else:
        print("No valid data found in either file to create a summary.")
        return

    # Fill any missing counts with 0
    summary_df.fillna(0, inplace=True)
    
    # Ensure the columns are in the correct order and type
    summary_df = summary_df[['PDB_id', 'number of unique contacts', 'number of nonunique contacts']]
    summary_df['number of unique contacts'] = summary_df['number of unique contacts'].astype(int)
    summary_df['number of nonunique contacts'] = summary_df['number of nonunique contacts'].astype(int)

    # Save the final summary to a new CSV file
    summary_df.to_csv(output_csv, index=False)
    
    print(f"\nDetailed summary saved successfully to '{output_csv}'.")
    print("\nFinal Summary DataFrame preview:")
    print(summary_df)

if __name__ == "__main__":
    unique_file = "recombinase_unique_contacts.csv"
    repeat_file = "recombinase_total_contacts.csv"
    output_file = "recombinase_summary.csv"
    
    create_detailed_contact_summary(unique_file, repeat_file, output_file)