from Bio.PDB import MMCIFParser, PDBIO, Select
import sys
import os

class AllSelect(Select):
    """A helper class to select all atoms when writing."""
    def accept_model(self, model):
        return True
    def accept_chain(self, chain):
        return True
    def accept_residue(self, residue):
        return True
    def accept_atom(self, atom):
        return True

def convert_cif_to_pdb(input_file="input_file.cif", output_file="output_file.pdb"):
    """
    Reads a macromolecular structure file using MMCIFParser 
    and converts/writes it to the legacy PDB format using PDBIO.
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating directory '{output_dir}': {e}")
            return

    if not os.path.exists(input_file):
        print(f"Error: Input file not found at '{input_file}'")
        return

    try:
        structure_id = os.path.basename(input_file).split('.')[0]
        
        parser = MMCIFParser()
        # The parser returns a Structure object (S) containing Models (M)
        # Chains (C), Residues (R), and Atoms (A) -> SMCRA
        structure = parser.get_structure(structure_id, input_file)

        # Write to output file using PDBIO
        io = PDBIO()
        io.set_structure(structure)
        
        # Save structure
        io.save(output_file, select=AllSelect())

        print(f"Successfully converted and wrote structure '{structure_id}'.")

    except Exception as e:
        print(f"An error occurred during conversion: {e}")
        print("Please ensure your input file is a valid mmCIF file and Biopython is installed.")

if __name__ == "__main__":
    cif_file = "cif_results/1NZB0.cif"
    base_name = os.path.basename(cif_file).replace(".cif", "")
    out_file = os.path.join("results", f"{base_name}.pdb")
    
    try:
        convert_cif_to_pdb(cif_file, out_file)
    except Exception as e:
        # Catch any exceptions that happen outside of the main function
        print(f"An unexpected error occurred: {e}")