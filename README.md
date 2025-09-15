
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
2. `protein_structrue` stores target graphs. 
3. `create_drug_feat.py` is used to ​​extract structural features from drug molecules (represented as SMILES)​​ and convert them into a format suitable for processing by ​​Graph Neural Networks​​. 
4. `main.py` trains GMJRL model.

##### Training

If you use the data we provide, you can run main.py directly.

For a new dataset, you need to prepare the following files:
1. drugs.xlsx: This file stores the SMILES string information of the drugs.
2. targets.xlsx: This file stores the Fasta sequence information of the targets.
3. dti_mat.xlsx: This file stores the interaction information between drugs and targets.
4. protein_structrue: This file stores target graphs.

Next, you need to run the following scripts:
1. Run create_drug_feat.py to extract structural features and convert them into a format suitable for processing by ​​Graph Neural Networks​​.
2. Run main.py to train the GMJRL model.
