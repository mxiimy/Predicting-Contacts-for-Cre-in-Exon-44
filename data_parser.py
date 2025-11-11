import os
import warnings
import pandas as pd
from Bio.PDB import PDBParser, NeighborSearch
from Bio.PDB.PDBExceptions import PDBConstructionWarning

# Unit =Angstroms
CONTACT_THRESHOLD = 5.0

# Only the first 13 and last 13 bases need to match (middle 8 can be anything)
TARGET_SEQUENCE = "ATAACTTCGTATAATGTATCCTCTATACGAACTTAT"
FIRST_13 = TARGET_SEQUENCE[:13]  # "ATAACTTCGTATA"
LAST_13 = TARGET_SEQUENCE[-13:]  # "ATACGAACTTAT"
TOTAL_LENGTH = 34  # Total length of the lox site (13 + 8 + 13)


def is_protein(residue):
    """Checks if a residue is a standard amino acid."""
    return residue.get_resname() in [
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE', 'LEU',
        'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    ]


def is_dna(residue):
    """Checks if a residue is a standard DNA base."""
    # Includes standard DNA bases and the Uracil Monophosphate (U) that may be present
    return residue.get_resname() in ['DA', 'DT', 'DG', 'DC', 'DU', 'A', 'T', 'C', 'G', 'U']


def get_base_letter(resname):
    """Convert DNA residue name to single letter."""
    mapping = {'DA': 'A', 'DT': 'T', 'DG': 'G', 'DC': 'C', 'DU': 'U', 'A': 'A', 'T': 'T', 'G': 'G', 'C': 'C', 'U': 'U'}
    return mapping.get(resname, 'N')


def find_sequence_in_chain(chain, first_13, last_13, total_length):
    """
    Finds sequences matching EITHER the first 13 OR last 13 bases (middle can be anything).
    Returns a list of tuples: (start_residue_id, end_residue_id, strand_direction, match_type, sequence)
    """
    # Get all DNA residues sorted by residue number
    dna_residues = [res for res in chain if is_dna(res)]
    dna_residues.sort(key=lambda r: r.get_id()[1])

    if len(dna_residues) < total_length:
        return []

    matches = []

    # Extract sequence from chain
    chain_sequence = ''.join([get_base_letter(res.get_resname()) for res in dna_residues])

    # Search for forward match (first 13 matched)
    for i in range(len(chain_sequence) - total_length + 1):
        segment = chain_sequence[i:i + total_length]
        if segment[:13] == first_13:
            start_idx = i
            end_idx = i + total_length - 1
            matches.append((
                dna_residues[start_idx].get_id()[1],
                dna_residues[end_idx].get_id()[1],
                'forward',
                'first_13_match',
                segment
            ))

    # Search for forward match (last 13 matched)
    for i in range(len(chain_sequence) - total_length + 1):
        segment = chain_sequence[i:i + total_length]
        if segment[-13:] == last_13:
            start_idx = i
            end_idx = i + total_length - 1
            # Check if already added (both first and last match)
            already_added = any(
                m[0] == dna_residues[start_idx].get_id()[1] and
                m[1] == dna_residues[end_idx].get_id()[1] and
                m[2] == 'forward'
                for m in matches
            )
            if not already_added:
                matches.append((
                    dna_residues[start_idx].get_id()[1],
                    dna_residues[end_idx].get_id()[1],
                    'forward',
                    'last_13_match',
                    segment
                ))
            else:
                # Update existing match to indicate both matched
                for j, m in enumerate(matches):
                    if (m[0] == dna_residues[start_idx].get_id()[1] and
                            m[1] == dna_residues[end_idx].get_id()[1] and
                            m[2] == 'forward'):
                        matches[j] = (m[0], m[1], m[2], 'both_match', m[4])

    # Search for reverse complement
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'U': 'A'}  

    # Reverse complement of first_13 and last_13
    first_13_rc = ''.join([complement.get(base, 'N') for base in reversed(first_13)])
    last_13_rc = ''.join([complement.get(base, 'N') for base in reversed(last_13)])

    # For reverse complement, last_13_rc is at the start, first_13_rc is at the end
    for i in range(len(chain_sequence) - total_length + 1):
        segment = chain_sequence[i:i + total_length]
        if segment[:13] == last_13_rc:
            start_idx = i
            end_idx = i + total_length - 1
            matches.append((
                dna_residues[start_idx].get_id()[1],
                dna_residues[end_idx].get_id()[1],
                'reverse_complement',
                'first_13_match',
                segment
            ))

    for i in range(len(chain_sequence) - total_length + 1):
        segment = chain_sequence[i:i + total_length]
        if segment[-13:] == first_13_rc:
            start_idx = i
            end_idx = i + total_length - 1
            # Check if already added
            already_added = any(
                m[0] == dna_residues[start_idx].get_id()[1] and
                m[1] == dna_residues[end_idx].get_id()[1] and
                m[2] == 'reverse_complement'
                for m in matches
            )
            if not already_added:
                matches.append((
                    dna_residues[start_idx].get_id()[1],
                    dna_residues[end_idx].get_id()[1],
                    'reverse_complement',
                    'last_13_match',
                    segment
                ))
            else:
                # Update existing match to indicate both matched
                for j, m in enumerate(matches):
                    if (m[0] == dna_residues[start_idx].get_id()[1] and
                            m[1] == dna_residues[end_idx].get_id()[1] and
                            m[2] == 'reverse_complement'):
                        matches[j] = (m[0], m[1], m[2], 'both_match', m[4])

    return matches


def get_target_dna_residues(structure, first_13, last_13, total_length):
    """
    Finds all DNA residues that match EITHER the first 13 OR last 13 bases across all chains.
    Returns a list of residues and information about where matches were found.
    """
    target_residues = []
    match_info = []

    for model in structure:
        for chain in model:

            matches = find_sequence_in_chain(chain, first_13, last_13, total_length)

            for start_res_id, end_res_id, direction, match_type, full_sequence in matches:
                match_info.append({
                    'chain': chain.get_id(),
                    'start': start_res_id,
                    'end': end_res_id,
                    'direction': direction,
                    'match_type': match_type,
                    'full_sequence': full_sequence
                })

                # Collect residues in this range
                for res in chain:
                    if is_dna(res):
                        res_id = res.get_id()[1]
                        if start_res_id <= res_id <= end_res_id:
                            target_residues.append(res)

    return target_residues, match_info


def analyze_contacts(pdb_path, first_13, last_13, total_length):
    """
    Parses a single PDB file to find all protein-DNA contacts
    within sequences matching the first 13 and last 13 bases.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PDBConstructionWarning)
        parser = PDBParser(QUIET=True)
        try:
            structure = parser.get_structure("complex", pdb_path)
        except Exception as e:
            print(f"Error parsing PDB file {pdb_path}: {e}")
            return [], [], []

    # Collect all protein atoms
    all_protein_atoms = []
    for model in structure:
        for chain in model:
            chain_id = chain.get_id()
            for residue in chain:
                if is_protein(residue):
                    all_protein_atoms.extend([atom for atom in residue])

    if not all_protein_atoms:
        print(f"No protein atoms found in {os.path.basename(pdb_path)}.")
        return [], [], []

    # Find target DNA sequence
    target_dna_residues, match_info = get_target_dna_residues(structure, first_13, last_13, total_length)

    if not target_dna_residues:
        print(f"Target DNA sequence pattern not found in {os.path.basename(pdb_path)}.")
        return [], [], match_info

    print(f"Found {len(match_info)} match(es) of the target sequence pattern:")
    for match in match_info:
        print(f"  Chain {match['chain']}: residues {match['start']}-{match['end']} ({match['direction']})")
        print(f"    Match type: {match['match_type']}")
        print(f"    Full sequence: {match['full_sequence']}")

    # Get atoms from target DNA residues
    target_dna_atoms = []
    for res in target_dna_residues:
        for atom in res:
            target_dna_atoms.append(atom)

    total_protein_dna_contacts = []
    dna_search = NeighborSearch(target_dna_atoms)

    for p_atom in all_protein_atoms:
        protein_res = p_atom.get_parent()
        nearby_dna_atoms = dna_search.search(p_atom.get_coord(), CONTACT_THRESHOLD)

        for d_atom in nearby_dna_atoms:
            dna_res = d_atom.get_parent()

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

    # Create unique contacts list
    unique_protein_dna_contacts = []
    seen_contacts = set()

    for contact in total_protein_dna_contacts:
        contact_tuple = (
            contact["protein_chain"],
            contact["residue_type"] + str(contact["residue_index"]),
            contact["dna_chain"],
            contact["dna_base_type"] + str(contact["dna_base_index"])
        )

        if contact_tuple not in seen_contacts:
            unique_protein_dna_contacts.append(contact)
            seen_contacts.add(contact_tuple)

    return total_protein_dna_contacts, unique_protein_dna_contacts, match_info


def create_summary(total_contacts, unique_contacts, pdb_id, match_info):
    """
    Creates a summary of the contact analysis.
    """
    summary = {
        'PDB_id': [pdb_id],
        'sequence_matches_found': [len(match_info)],
        'number of unique contacts': [len(unique_contacts)],
        'number of total contacts': [len(total_contacts)]
    }
    return pd.DataFrame(summary)


def main():
    """
    Main function to run the analysis workflow.
    """
    # Specify PDB file path here
    pdb_file_path = input("Enter the path to your PDB file: ").strip()

    if not os.path.exists(pdb_file_path):
        print(f"Error: The file '{pdb_file_path}' was not found.")
        return

    pdb_id = os.path.basename(pdb_file_path).split('.')[0].upper()

    total_contacts, unique_contacts, match_info = analyze_contacts(pdb_file_path, FIRST_13, LAST_13, TOTAL_LENGTH)

    if not match_info:
        print("Target DNA sequence not found in the PDB file.")
        return

    if not total_contacts:
        print("No protein-DNA contacts found in the target sequence region.")
        return

    print(f"\nFound {len(total_contacts)} total protein-DNA contacts ({len(unique_contacts)} unique).")

    os.makedirs("results", exist_ok=True) 

    # Save total contacts
    if total_contacts:
        total_contacts_filepath = os.path.join("results", pdb_id + ".csv")
        df_total = pd.DataFrame(total_contacts)
        # total_csv_file = f"{pdb_id}_total_contacts.csv"
        df_total.to_csv(total_contacts_filepath, index=False)
        print(f"\nAll contacts saved to {total_contacts_filepath}")

    # Save unique contacts
    if unique_contacts:
        df_unique = pd.DataFrame(unique_contacts)
        unique_contacts_filepath = os.path.join("results", f"{pdb_id}_unique_contacts.csv")
        df_unique.to_csv(unique_contacts_filepath, index=False)
        print(f"Unique contacts saved to {unique_contacts_filepath}")

    # Create and save summary
    summary_df = create_summary(total_contacts, unique_contacts, pdb_id, match_info)
    summary_csv_file = os.path.join("results", f"{pdb_id}_summary.csv") 
    summary_df.to_csv(summary_csv_file, index=False)
    print(f"Summary saved to {summary_csv_file}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    # This file is a parser that only looks at one pdb file at a time 
    # in case we need to look carefully at one singular pdb's results
    main()
