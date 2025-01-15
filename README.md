
# GMJRL
GMJRL: Geometry-enhanced Multi-scale Joint Representation Learning for Drug-Target Interaction Prediction

##### Dependencies：

- python = 3.8.18
- pytorch = 1.12.1
- torchvision = 0.13.1
- cudatoolkit = 11.3.1
- rdkit = 2023.9.1
- torch-cluster = 1.6.0+pt112cu113
- torch-scatter = 2.10+pt112cu113
- torch-sparse = 0.6.16+pt112cu113
- torch-spline-conv = 1.2.1+pt112cu113
- torch-geometric = 2.4.0
- networkx = 3.1
- torchdrug = 0.2.1
- bio = 1.6.2
- openpyxl = 3.1.2

##### Using

1. `Data` stores four datasets.
2. `smile_to_features.py` generates chemical text information. 
3. `smiles_k_gram.py` lets the chemical text be divided into words according to the k-gram method. 
4. `protein_k_gram` lets the protein sequences be divided into words according to the k-gram method. 
5. `cluster.py` stores the neighborhood-enhanced graph contrastive learning algorithm.
6. `main.py` trains NEGCDTI model.
