import os

if __name__ == "__main__":
    for dna in os.listdir("sequences/raw_dna"):
        file_path = os.path.join("sequences", "raw_dna", dna)
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