import os
import requests
import warnings
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# List of PDB IDs for recombinase/integrase complexes
PDB_IDS = ["1PVR", "1CRE", "1CRX", "1FLR", "1FU7", "2BRX", "5YV1", "1P71", "1D4Y", "2RCK"]

CONTACT_THRESHOLD = 5.0 # in Angstroms

PDB_DIR = "pdb_files"

def download_pdb_files_direct(pdb_ids, download_dir):
    """
    Downloads PDB files directly from the RCSB PDB database using requests.
    """
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    for pdb_id in pdb_ids:
        pdb_id_lower = pdb_id.lower()
        url = f"https://files.rcsb.org/download/{pdb_id_lower}.pdb"
        file_path = os.path.join(download_dir, f"{pdb_id_lower}.pdb")
        
        print(f"Downloading {pdb_id} from {url}")
        
        response = requests.get(url)
        if response.status_code == 200:
            with open(file_path, 'w') as f:
                f.write(response.text)
            print(f"Successfully downloaded {pdb_id}")
        else:
            print(f"Failed to download {pdb_id}. Status code: {response.status_code}")

def is_protein(residue):
    """Checks if a residue is a standard amino acid."""
    return residue.get_resname() in [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU',
        'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']

def is_dna(residue):
    """Checks if a residue is a standard DNA base."""
    return residue.get_resname() in ['DA', 'DT', 'DG', 'DC']

def parse_contacts(pdb_path):
    """
    Parses a single PDB file to find protein-DNA contacts.
    Returns a list of dictionaries, one for each unique contact.
    """
    # Suppress warnings for missing atoms in the PDB file
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", pdb_path)
    
    contacts = []
    protein_atoms = []
    dna_atoms = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if is_protein(residue):
                    for atom in residue:
                        protein_atoms.append(atom)
                elif is_dna(residue):
                    for atom in residue:
                        dna_atoms.append(atom)

    if not dna_atoms:
        return [[], []]
    
    ns = NeighborSearch(dna_atoms)
    unique_contacts = set()
    total_contacts = []


    for p_atom in protein_atoms:
        protein_residue = p_atom.get_parent()
        nearby_atoms = ns.search(p_atom.get_coord(), CONTACT_THRESHOLD)
        
        if len(nearby_atoms) > 0:
            dna_residue = nearby_atoms[0].get_parent()
            
            # Create a unique key to prevent duplicate contact entries
            contact_key = (
                f"{protein_residue.get_resname()}{protein_residue.get_id()[1]}-"
                f"{dna_residue.get_resname()}{dna_residue.get_id()[1]}"
            )
            contact_data = {
                    "contact_key": contact_key,
                    "pdb_id": os.path.basename(pdb_path).split('.')[0].upper(),
                    "protein_chain": protein_residue.get_parent().get_id(),
                    "residue_type": protein_residue.get_resname(),
                    "residue_index": protein_residue.get_id()[1],
                    "dna_chain": dna_residue.get_parent().get_id(),
                    "dna_base_type": dna_residue.get_resname(),
                    "dna_base_index": dna_residue.get_id()[1],
                    "is_contact": True
                }
            
            if contact_key not in unique_contacts:
                unique_contacts.add(contact_key)
                contacts.append(contact_data)
            total_contacts.append(contact_data)
                
    return [contacts, total_contacts]

def main():
    """
    Runs the full data parsing workflow: downloads files,
    parses contacts, and saves the data to a CSV.
    """
    # Download the PDB files
    print("--- Starting PDB Download ---")
    download_pdb_files_direct(PDB_IDS, PDB_DIR)

    # Parse the downloaded files to find contacts
    all_contacts = []
    print("\n--- Starting Parsing Unique Contacts ---")
    for pdb_id in PDB_IDS:
        file_path = os.path.join(PDB_DIR, f"{pdb_id.lower()}.pdb")
        if os.path.exists(file_path):
            print(f"Parsing {pdb_id}...")
            contacts = parse_contacts(file_path)[0]  # Get only unique contacts
            all_contacts.extend(contacts)
            print(f"  Found {len(contacts)} contacts for {pdb_id}.")
        else:
            print(f"File for {pdb_id} not found. Skipping.")

    print(f"\n--- Starting Parsing Total Contacts ---")   
    total_contacts = []
    for pdb_id in PDB_IDS:
        file_path = os.path.join(PDB_DIR, f"{pdb_id.lower()}.pdb")
        if os.path.exists(file_path):
            print(f"Parsing {pdb_id}...")
            contacts = parse_contacts(file_path)[1]  # Get total contacts
            total_contacts.extend(contacts)
            print(f"  Found {len(contacts)} contacts for {pdb_id}.")
        else:
            print(f"File for {pdb_id} not found. Skipping.")     

    # Convert to DataFrame and save to CSV
    if all_contacts:
        df = pd.DataFrame(all_contacts)
        # Drop the temporary contact key column
        df = df.drop(columns=['contact_key'])
        
        csv_file = "recombinase_unique_contacts.csv"
        df.to_csv(csv_file, index=False)
        
        print("\n--- Summary for unique contacts ---")
        print(f"Total unique residue-DNA contacts found: {len(df)}")
        print(f"All contacts saved to {csv_file}")
        print("\nDataFrame preview:")
        print(df.head())
    else:
        print("No contacts were found for unique residues.")

    if total_contacts:
        df_total = pd.DataFrame(total_contacts)
        # Drop the temporary contact key column
        df_total = df_total.drop(columns=['contact_key'])
        
        csv_file = "recombinase_total_contacts.csv"
        df_total.to_csv(csv_file, index=False)
        
        print("\n--- Summary for Total Contacts ---")
        print(f"Total residue-DNA contacts found (including repeats): {len(df_total)}")
        print(f"All total contacts saved to {csv_file}")
        print("\nDataFrame preview:")
        print(df_total.head())

if __name__ == "__main__":
    main()