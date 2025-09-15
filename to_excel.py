import pandas as pd
import os

def convert_csv_to_excel(csv_file, excel_file):
    """
    Converts a CSV file to an Excel file using pandas.
    """
    if not os.path.exists(csv_file):
        print(f"Error: The file '{csv_file}' was not found.")
        return

    # Read the CSV file into a pandas DataFrame
    print(f"Reading data from '{csv_file}'...")
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Write the DataFrame to an Excel file
    print(f"Writing data to '{excel_file}'...")
    try:
        df.to_excel(excel_file, index=False)
        print("Conversion complete!")
    except Exception as e:
        print(f"Error writing Excel file: {e}")

if __name__ == "__main__":
    csv_input_file = "recombinase_contacts.csv"
    excel_output_file = "recombinase_contacts.xlsx"
    convert_csv_to_excel(csv_input_file, excel_output_file)