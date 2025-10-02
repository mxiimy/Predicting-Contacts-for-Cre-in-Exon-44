# This script analyzes PDB files to find and count protein-DNA contacts,
# including unique and total counts, specifically at Lox sites.
# It uses a provided CSV file for site and chain information.

import os
import io
import warnings
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Directory where PDB files are stored.
PDB_DIR = "pdb_files"

# The distance threshold in Angstroms for considering two atoms to be in contact.
CONTACT_THRESHOLD = 5.0

def is_protein(residue):
    """Checks if a residue is a standard amino acid."""
    return residue.get_resname() in [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU',
        'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

def is_dna(residue):
    """Checks if a residue is a standard DNA base."""
    return residue.get_resname() in ['DA', 'DT', 'DG', 'DC']

def load_lox_sites_data(csv_data):
    """
    Loads and processes the Lox site data from the provided CSV string.

    Args:
        csv_data (str): The content of the Lox site CSV file as a string.

    Returns:
        dict: A dictionary mapping PDB IDs to a list of dictionaries,
              each containing the chain and start residue for a Lox site.
    """
    df = pd.read_csv(io.StringIO(csv_data))
    lox_data = {}
    for _, row in df.iterrows():
        pdb_id = row['pdb_id'].upper()
        if pdb_id not in lox_data:
            lox_data[pdb_id] = []
        
        # Add the first Lox site and its chain
        lox_data[pdb_id].append({
            'chain': row['chain 1'],
            'start_residue': row['lox site start 1']
        })
        # Add the second Lox site and its chain
        lox_data[pdb_id].append({
            'chain': row['chain 2'],
            'start_residue': row['lox site start 2']
        })
    return lox_data

def get_lox_residues(structure, lox_sites):
    """
    Extracts all DNA residues corresponding to the specified Lox sites.
    A Lox site is defined as 34 bases long.
    """
    lox_residues = []
    for site in lox_sites:
        chain_id = site['chain']
        start_res_id = site['start_residue']
        end_res_id = start_res_id + 33  # Lox sites are 34 base pairs long
        
        for model in structure:
            if chain_id in model:
                chain = model[chain_id]
                for res in chain:
                    res_id = res.get_id()[1]
                    if is_dna(res) and start_res_id <= res_id <= end_res_id:
                        lox_residues.append(res)
    return lox_residues

def analyze_contacts(pdb_path, lox_sites):
    """
    Parses a single PDB file to find all protein-DNA contacts
    at specific Lox sites and returns both a unique and total list of contacts.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("complex", pdb_path)
        except Exception as e:
            print(f"Error parsing PDB file {pdb_path}: {e}")
            return [], []
            
    # Collect all protein atoms and DNA atoms from the structure.
    all_protein_atoms = []
    all_dna_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_protein(residue):
                    all_protein_atoms.extend([atom for atom in residue])
                elif is_dna(residue):
                    all_dna_atoms.extend([atom for atom in residue])
    
    if not all_protein_atoms or not all_dna_atoms:
        print(f"Skipping {os.path.basename(pdb_path)}: Missing protein or DNA atoms.")
        return [], []

    lox_dna_residues = get_lox_residues(structure, lox_sites)

    lox_dna_atoms = []
    for res in lox_dna_residues:
        for atom in res:
            lox_dna_atoms.append(atom)
    
    if not lox_dna_atoms:
        print(f"No loxP site atoms found in {os.path.basename(pdb_path)}.")
        return [], []

    total_protein_dna_contacts = []
    unique_contact_keys = set()
    dna_search = NeighborSearch(lox_dna_atoms)
    
    for p_atom in all_protein_atoms:
        protein_res = p_atom.get_parent()
        nearby_dna_atoms = dna_search.search(p_atom.get_coord(), CONTACT_THRESHOLD)
        
        for d_atom in nearby_dna_atoms:
            dna_res = d_atom.get_parent()
            
            contact_key = (
                f"{protein_res.get_resname()}{protein_res.get_id()[1]}",
                f"{dna_res.get_resname()}{dna_res.get_id()[1]}"
            )
            
            contact_data = {
                "pdb_id": os.path.basename(pdb_path).split('.')[0].upper(),
                "protein_chain": protein_res.get_parent().get_id(),
                "residue_type": protein_res.get_resname(),
                "residue_index": protein_res.get_id()[1],
                "dna_chain": dna_res.get_parent().get_id(),
                "dna_base_type": dna_res.get_resname(),
                "dna_base_index": dna_res.get_id()[1],
                "is_contact": True
            }
            
            total_protein_dna_contacts.append(contact_data)
            unique_contact_keys.add(contact_key)

    unique_protein_dna_contacts = []
    temp_unique_list = []
    
    for contact in total_protein_dna_contacts:
        contact_tuple = (contact["residue_type"] + str(contact["residue_index"]),
                         contact["dna_base_type"] + str(contact["dna_base_index"]))
        
        if contact_tuple not in temp_unique_list:
            unique_protein_dna_contacts.append(contact)
            temp_unique_list.append(contact_tuple)

    return total_protein_dna_contacts, unique_protein_dna_contacts

def create_detailed_contact_summary(unique_csv, repeat_csv, output_csv):
    """
    Reads two CSV files (unique and non-unique contacts), counts the number
    of entries for each PDB ID in both, and saves a summary to a new CSV.

    Parameters:
    - unique_csv (str): Path to the CSV with unique contacts.
    - repeat_csv (str): Path to the CSV with non-unique contacts.
    - output_csv (str): Path where the summary CSV will be saved.
    """
    unique_counts = pd.DataFrame()
    repeat_counts = pd.DataFrame()

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

    if os.path.exists(repeat_csv):
        print(f"Reading non-unique contacts from '{repeat_csv}'...")
        try:
            df_repeat = pd.read_csv(repeat_csv)
            if 'pdb_id' in df_repeat.columns:
                repeat_counts = df_repeat['pdb_id'].value_counts().reset_index()
                repeat_counts.columns = ['PDB_id', 'number of nonunique contacts']
        except Exception as e:
            print(f"Error reading non-unique contacts file: {e}")
    else:
        print(f"Warning: Non-unique contacts file '{repeat_csv}' not found. Non-unique contact counts will be 0.")

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
    
    summary_df = summary_df[['PDB_id', 'number of unique contacts', 'number of nonunique contacts']]
    summary_df['number of unique contacts'] = summary_df['number of unique contacts'].astype(int)
    summary_df['number of nonunique contacts'] = summary_df['number of nonunique contacts'].astype(int)
    summary_df.to_csv(output_csv, index=False)
    
    print(f"\nDetailed summary saved successfully to '{output_csv}'.")
    print("\nFinal Summary DataFrame preview:")
    print(summary_df)

def main():
    """
    Main function to run the full analysis workflow.
    """
    lox_csv_file_path = "lox_site_results.csv"

    if not os.path.exists(lox_csv_file_path):
        print(f"Error: The file '{lox_csv_file_path}' was not found.")
        print("Please ensure it's in the same directory as the script.")
        return
        
    with open(lox_csv_file_path, 'r') as f:
        lox_csv_data = f.read()
    
    lox_sites_data = load_lox_sites_data(lox_csv_data)
    
    all_protein_dna_contacts_total = []
    all_protein_dna_contacts_unique = []

    for pdb_id in lox_sites_data.keys():
        file_path = os.path.join(PDB_DIR, f"{pdb_id.lower()}.pdb")
        
        if os.path.exists(file_path):
            print(f"Analyzing {pdb_id}...")
            total_contacts, unique_contacts = analyze_contacts(file_path, lox_sites_data[pdb_id])
            all_protein_dna_contacts_total.extend(total_contacts)
            all_protein_dna_contacts_unique.extend(unique_contacts)
            print(f"  Found {len(total_contacts)} total protein-DNA contacts ({len(unique_contacts)} unique).")
        else:
            print(f"File for {pdb_id} not found. Skipping.")
    
    if all_protein_dna_contacts_total:
        df_total = pd.DataFrame(all_protein_dna_contacts_total)
        total_csv_file = "recombinase_total_contacts.csv"
        df_total.to_csv(total_csv_file, index=False)
        print(f"Total protein-DNA contacts found: {len(df_total)}")
        print(f"All contacts saved to {total_csv_file}")
        
    if all_protein_dna_contacts_unique:
        df_unique = pd.DataFrame(all_protein_dna_contacts_unique)
        unique_csv_file = "recombinase_unique_contacts.csv"
        df_unique.to_csv(unique_csv_file, index=False)
        print(f"Total unique protein-DNA contacts found: {len(df_unique)}")
        print(f"All unique contacts saved to {unique_csv_file}")
    
    create_detailed_contact_summary(
        unique_csv="recombinase_unique_contacts.csv",
        repeat_csv="recombinase_total_contacts.csv",
        output_csv="recombinase_summary.csv"
    )

if __name__ == "__main__":
    main()