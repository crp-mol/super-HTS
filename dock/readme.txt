# Download PDB 4e3q
wget https://files.rcsb.org/download/4E3Q.pdb

# Prepare input structure for Rosetta
1. Delete mol C D from PDB
2. Clean structure
3. Add the ligand to the structure in approx. the location of the binding site
4. Save pdb -> 4e3q.pdb
5. Add in line1: "REMARK   0 BONE TEMPLATE X SUB    0 MATCH MOTIF A LYS   285   1"

# Generate Rosetta params file
${ROSETTA_DIR}/main/source/scripts/python/public/molfile_to_params.py 04S.mol2 -n SUB --chain=X --clobber --keep-names
Add the following string to the last line of SUB.params: "PDB_ROTAMERS SUB_rotamersA.pdb"

# Run Rosetta enzyme design application to dock and score the mutants
cd W57F_F19D_F85I_V225D_A228S; ${ROSETTA_DIR}/main/source/bin/enzyme_design.static.linuxgccrelease @flags -resfile resfile -database $ROSETTA_DIR/main/database/ -enzdes::cstfile enzdes.cst -nstruct 10 -s 4e3q.pdb > log
cd Y150T_F86I; ${ROSETTA_DIR}/main/source/bin/enzyme_design.static.linuxgccrelease @flags -resfile resfile -database $ROSETTA_DIR/main/database/ -enzdes::cstfile enzdes.cst -nstruct 10 -s 4e3q.pdb > log

# read the binding energy from `score.sc`
awk '{print $29}' score.sc
