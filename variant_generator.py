# From SLiDE paper, take Cre wild type and write script to implement mutations so 
# that they can represent the mutated Cre variants. Use this as the training data 
# set for activity. Focus on loxH but if possible, also look into loxH and loxP 
# recombinase sequences 

import os

# Downloaded from Uniprot:
CRE = "MSNLLTVHQNLPALPVDATSDEVRKNLMDMFRDRQAFSEHTWKMLLSVCRSWAAWCKLNNRKWFPAEPEDVRDYLLYLQARGLAVKTIQQHLGQLNMLHRRSGLPRPSDSNAVSLVMRRIRKENVDAGERAKQALAFERTDFDQVRSLMENSDRCQDIRNLAFLGIAYNTLLRIAEIARIRVKDISRTDGGRMLIHIGRTKTLVSTAGVEKALSLGVTKLVERWISVSGVADDPNNYLFCRVRKNGVAAPSATSQLSTRALEGIFEATHRLIYGAKDDSGQRYLAWSGHSARVGAARDMARAGVSIPEIMQAGGWTNVNIVMNYIRNLDSETGAMVRLLEDGD"

clones = {
    "Fre1": ["N3D", 'D29G', 'L284Q'],
    'Fre2': ["N3D", 'V7L', 'P107L', 'A285T'],
    'Fre3': ["S2T", 'N3K', 'L4S', 'I180V'],
    'Fre4': ["G93S", 'S108G', 'E262A'],
    'Fre5': ["N3D", 'L14S', 'G190S', 'I320M'],
    'Fre6': ["V7L", 'P107L', 'M193V', 'F278G'],
    'Fre7': ["F239L", 'L284R', 'D343G'],
    'Fre8': ["I88T", 'E262Q'],
    'Fre9': ["N3D", 'V7A', 'F64L', 'Q156R', 'E262A', 'D343E'],
    'Fre10': ["S2P", 'N3Y', 'L14S', 'S51L', 'P105L', 'E262G', 'D278N'],
    'Fre11': ["N3D", 'V7L', 'S51L', 'P107L', 'E150G', 'D153G', 'S254N', 'D278G', 'I320S'],
    'Fre12': ["N3D", 'S186T', 'V209I', 'E262A'],
    'Fre13': ['N3D', 'V7L', 'H8Y', 'G93S', 'A260T', 'D278G'],
    'Fre14': ['N3D', 'V7L', 'M193V', 'E262A', 'I320S'],
    'Fre15': ['S2P', 'N3D', 'H196R', 'Q255R', 'E262Q', 'N317T', 'I320S'],
    'Fre16': ['N3D', 'V7L', 'M30I', 'P107L', 'N151H', 'E262Q', 'S305P'],
    'Fre17': ['S2P', 'V7L', 'N10D', 'Q144H', 'H196Q', 'E262Q', 'T316S', 'N317T', 'I320S'],
    'Fre18': ['S2F', 'N3D', 'V7L', 'D21E', 'M30L', 'Q94L', 'N124T', 'E129G', 'D189G', 'A260T', 'Y273H', 'D278G', 'N317T', 'I320S'],
    'Fre19': ['V7L', 'N10S', 'M30L', 'F64I', 'Q94L', 'S108G', 'I158V', 'E176Q', 'P234S', 'N236S', 'E262Q', 'N317T', 'I320S'],
    'Fre20': ['T6P', 'V7L', 'T19M', 'Q94L', 'P107L', 'N111S', 'N151D', 'D232G', 'E262Q', 'N317T', 'I320S'],
    'Fre21': ['S2F', 'N3D', 'V7L', 'D21E', 'M30L', 'Q94L', 'D153G', 'E176Q', 'S186T', 'G198S', 'T218A', 'E262Q', 'N317T', 'I320S'],
    'Fre22': ['V7L', 'N10K', 'M30L', 'N59S', 'V85A', 'K86N', 'Q94L','E129G', 'N151K', 'G216R', 'E262Q', 'L284Q', 'N317T', 'N319K', 'I320S'],
    'Fre23': ['N3K', 'V7L', 'E39G', 'A53V', 'N59D', 'Q94L', 'R101Q', 'S114T', 'N160T', 'D233G', 'E262Q', 'N319K', 'I320S'],
    'Fre24': ['V7L', 'D73Y', 'Y77H', 'S108G', 'E150A', 'E262Q', 'N317H', 'I320S', 'M322I']
}

def generate_variant(sequence, mutations):
    """Generates a protein variant sequence given a list of mutations.
    
    Args:
        sequence (str): The original protein sequence.
        mutations (list): A list of mutations in the format 'A23T' where A is the 
                          original amino acid, 23 is the position (1-based), and T 
                          is the new amino acid.
    
    Returns:
        str: The mutated protein sequence.
    """
    seq_list = list(sequence)
    for mutation in mutations:
        original_aa = mutation[0]
        position = int(mutation[1:-1]) - 1  # Convert to 0-based index
        new_aa = mutation[-1]
        
        if seq_list[position] != original_aa:
            raise ValueError(f"Original amino acid at position {position + 1} "
                             f"does not match: expected {original_aa}, "
                             f"found {seq_list[position]}")
        
        seq_list[position] = new_aa
    
    return ''.join(seq_list)

def main():
    os.makedirs("CRE_clones", exist_ok=True) 
    for clone_name, mutations in clones.items():
        file_path = os.path.join("CRE_clones", clone_name)
        try:
            with open(file_path, "w") as f:
                clone = generate_variant(CRE, mutations)
                f.write(clone)
            print(f">{clone_name}\n{clone}\n")
            # if clone_name in os.listdir("CRE_clones"):
            #     continue
            # else:
            #     with open(file_path, "w") as f:
            #         clone = generate_variant(CRE, mutations)
            #         f.write(clone)
            #     print(f">{clone_name}\n{clone}\n")
        except ValueError as e:
            print(f"Error generating variant for {clone_name}: {e}")

if __name__ == "__main__":
    main()