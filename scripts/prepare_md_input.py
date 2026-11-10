from pathlib import Path

root = Path(__file__).resolve().parents[1]
lines = (root / "data/raw/1L2Y.pdb").read_text().splitlines()
model = []
in_first_model = False
for line in lines:
    if line.startswith("MODEL") and line[10:14].strip() == "1":
        in_first_model = True
        continue
    if in_first_model and line.startswith("ENDMDL"):
        break
    if in_first_model and line.startswith(("ATOM", "TER")):
        model.append(line)
model.append("END")
(root / "data/prepared/1L2Y_model1.pdb").write_text("\n".join(model) + "\n")

sequence = "NLYIQWLKDGGPSSGRPPPS"
(root / "data/prepared/1L2Y.fasta").write_text(">1L2Y_model1\n" + sequence + "\n")
print(f"Saved {len(model) - 1} PDB records and {len(sequence)} residues")
