# dataset   Nhot    Lmax    Naa    substrate
D1          4       4       20      4
D2          6       6       20      4
D3          8       8       20      4
D4          8       4       20      4
D5          4       4       10      4
D6          4       4       20      1
D7          4       4       20      2
D8          4       4       20      3
D9          4       4       20      5

* Naa: number of aminoacids allowed as target mutation. Standard is 20, but you can try to reduce this number to see if the NN can later generalize to unseen amino acids.

# How to read:
-11.0340 0.8502 10  F19A_F85M_W57E_Y150T
-11.5380 0.6346 10  F19A_F85M_Y150R_W57D
-12.0190 0.2365 10  F19A_F85S_Y150V
    ^       ^   ^       ^
    |       |   |       |
    |       |   |       mutant_id
    |       |   number of replicas
    |     stdev
average binding energy
