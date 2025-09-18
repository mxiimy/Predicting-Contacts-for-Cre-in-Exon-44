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

    try:
        df.to_excel(excel_file, index=False)
    except Exception as e:
        print(f"Error writing Excel file: {e}")

if __name__ == "__main__":

    for csv_in, csv_out in [
        ("lox_site_results.csv", "lox_site_results.xlsx"),
        ("recombinase_unique_contacts.csv", "recombinase_unique_contacts.xlsx"),
        ("recombinase_total_contacts.csv", "recombinase_total_contacts.xlsx"),
        ("recombinase_summary.csv", "recombinase_summary.xlsx")
    ]:
        convert_csv_to_excel(csv_in, csv_out)