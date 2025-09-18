import os
import re
import requests
import warnings
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Define the two parts of the loxP sequence
LOXP_PART1_SEQ = "ATAACTTCGTATA"
LOXP_PART2_SEQ = "TATACGAAGTTAT"
PDB_IDS = ["1NZB", "1PVR", "1KBU", "1PVP", "1PVQ", "1XNS", "1XO0", "2HOF", "3C28", "3C29", "3MGV", "5CRX", "7RHX", "7RHY", "7RHZ","1CRX", "5YV1", "1P71"]

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

def is_dna(residue):
    """
    Checks if a residue is a standard DNA base, including common
    naming conventions.
    """
    return residue.get_resname() in ['DA', 'DT', 'DG', 'DC', 'A', 'T', 'G', 'C']

def find_lox_sites(pdb_id):
    """
    Finds and reports on lox sites by searching for the two separate
    inverted repeat sequences and their spatial relationship.
    Returns a list of dictionaries with detailed information.
    """
    pdb_id_lower = pdb_id.lower()
    file_path = os.path.join(PDB_DIR, f"{pdb_id_lower}.pdb")
    if not os.path.exists(file_path):
        print(f"PDB file for {pdb_id} not found. Skipping.")
        return []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_id, file_path)

    part1_matches = []
    part2_matches = []

    for model in structure:
        for chain in model:
            dna_sequence = ""
            dna_residues = []
            
            for residue in chain:
                if is_dna(residue):
                    base = residue.get_resname()[-1]
                    dna_sequence += base
                    dna_residues.append(residue)
            for match in re.finditer(LOXP_PART1_SEQ, dna_sequence):
                part1_matches.append({
                    "chain_id": chain.get_id(),
                    "start_index": match.start(),
                    "start_residue": dna_residues[match.start()].get_id()[1]
                })
            for match in re.finditer(LOXP_PART2_SEQ, dna_sequence):
                part2_matches.append({
                    "chain_id": chain.get_id(),
                    "start_index": match.start(),
                    "start_residue": dna_residues[match.start()].get_id()[1]
                })

    full_lox_sites = []

    for part1 in part1_matches:
        for part2 in part2_matches:
            same_chain = (part1["chain_id"] == part2["chain_id"])

            if same_chain:
                distance = part2["start_residue"] - (part1["start_residue"] + len(LOXP_PART1_SEQ))
                if distance == 8:
                    full_lox_sites.append({
                        "pdb_id": pdb_id,
                        "lox site start 1": part1["start_residue"],
                        "lox site start 2": part2["start_residue"],
                        "same_chain": True,
                        "distance between": 8
                    })
            else:
                # If on different chains, they are considered a "full" lox site in the recombination intermediate.
                # The distance between them is not meaningful and is null.
                full_lox_sites.append({
                    "pdb_id": pdb_id,
                    "lox site start 1": part1["start_residue"],
                    "lox site start 2": part2["start_residue"],
                    "same_chain": False,
                    "distance between": None
                })
    return full_lox_sites

if __name__ == "__main__":
    print("--- Starting PDB Download ---")
    download_pdb_files_direct(PDB_IDS, PDB_DIR)
    
    all_results = []
    print("\n--- Starting Lox Site Analysis ---")
    for pdb_id in PDB_IDS:
        print(f"\nProcessing PDB ID: {pdb_id}")
        results = find_lox_sites(pdb_id)
        if results:
            all_results.extend(results)
            print(f"Found {len(results)} potential lox sites.")
        else:
            print(f"No loxP sites found for {pdb_id}.")

    if all_results:
        df = pd.DataFrame(all_results)
        
        csv_filename = "lox_site_results.csv"
        df.to_csv(csv_filename, index=False)
        print(f"\n--- Analysis Complete ---")
        print("\nDataFrame preview:")
        print(df.head())
    else:
        print("\nNo lox site patterns were found across all PDB IDs.")
