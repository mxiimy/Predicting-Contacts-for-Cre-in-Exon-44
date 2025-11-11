import os
import openprotein
from openprotein.protein import Protein
from openprotein.chains import DNA
from dotenv import load_dotenv 
from CIF_to_PDB import convert_cif_to_pdb

load_dotenv() 

def fold(session, name, p_seq, dna_seq):
    proteins = [Protein(sequence=p_seq)]
    proteins[0].chain_id = ["A", "B"]

    chain_ids = ["C", "D", "E", "F", "G", "H", "I", "J", "K","L", "M", "N", 
                 "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    dnas = []
    for dna in dna_seq:
        dnas.append(DNA(sequence=dna, chain_id=chain_ids.pop()))

    msa_query = []
    for p in proteins:
        if p.chain_id is not None and isinstance(p.chain_id, list):
            for _ in p.chain_id:
                msa_query.append(p.sequence.decode())
        else:
            msa_query.append(p.sequence.decode())
    msa = session.align.create_msa(seed=":".join(msa_query))

    for p in proteins:
        p.msa = msa

    fold_job = session.fold.boltz2.fold(
        proteins=proteins,
        dnas=dnas
    )
    
    fold_job.wait_until_done(verbose=True)

    result = fold_job.get()

    # Make sure this folder exists
    cif_dir = "cif_results"
    os.makedirs(cif_dir, exist_ok=True) 
    cif_file_path = os.path.join(cif_dir, name + ".cif")

    # Write the intermediate CIF file
    with open(cif_file_path, "wb") as f:
        f.write(result)

    # Define the final PDB output path
    pdb_output_path = os.path.join("pdb_results", name + ".pdb")
    
    # Convert from CIF to PDB and save to results
    convert_cif_to_pdb(cif_file_path, pdb_output_path)

def run_all_folds():
    """Connects to OpenProtein and runs all folding jobs."""
    user = os.environ.get("OPENPROTEIN_USERNAME")
    pw = os.environ.get("OPENPROTEIN_PASSWORD")

    session = openprotein.connect(username=user, password=pw, timeout=600)

    names = []
    protein_list = []
    # Assume the data is in the form:
    #   - each protein to fold in its own file and are folded as a dimer
    #   - each file is one line of a protein sequence
    for protein in os.listdir("sequences/proteins"):
        file_path = os.path.join("sequences", "proteins", protein)
        names.append(protein) # name of file will be the name of the fold
        with open(file_path, 'r') as file:
            lines = file.readlines()
            protein_list.append(lines[0].strip()) # Add protein with list of proteins to fold and get rid of white space
    
    # Clean the DNA first:
    for dna in os.listdir("sequences/raw_dna"):
        file_path = os.path.join("sequences", "raw_dna", dna)
        if dna in os.listdir("sequences/dnas"):
            continue # Already cleaned
        else:
            with open(file_path, 'r') as file:
                d = []
                for line in file.readlines():
                    line = line.upper()
                    clean = line.replace('U', 'T') # Replace all instances of U with T
                    d.append(clean.strip()) # Add dna with list of dna and get rid of white space
            os.makedirs("dnas", exist_ok=True)
            file_path_w = os.path.join("sequences", "dnas", dna)
            with open(file_path_w, "w") as f:
                for i in d:
                    f.write(i + "\n")

    dna_list = []
    # Assume the data is in the form:
    #   - each dna to fold in its own file and are folded as a dimer
    for dna in os.listdir("sequences/dnas"):
        file_path = os.path.join("sequences", "dnas", dna)
        with open(file_path, 'r') as file:
            d = []
            for line in file.readlines():
                d.append(line.strip()) # Add dna with list of dna to fold and get rid of white space
            dna_list.append(d)

    # Assume every protein is to be folded with each dna sequence
    for name, protein in zip(names, protein_list):
        for j in range(len(dna_list)):
            foldname = name + str(j) # To identify which DNA the protein is folded with
            fold(session, foldname, protein, dna_list[j])

if __name__ == "__main__":
    run_all_folds()