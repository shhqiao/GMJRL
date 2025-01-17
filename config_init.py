import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="This is a template of machine learning developping source code.")
    parser.add_argument('-epoch_num', '--epoch_num_topofallfeature', type=int, default=50,help='ratio of drop the graph nodes.')
    parser.add_argument('-lr', '--lr_topofallfeature', type=float, default=0.00005,help='number of epoch.')
    parser.add_argument('-batch_size', '--batch_size_topofallfeature', type=int, default=512)
    parser.add_argument('-device', '--device_topofallfeature', type=str, default='cuda:0')
    parser.add_argument('-dropout', '--dropout_topofallfeature', type=float, default=0.)
    parser.add_argument('-embedding_dim', '--embedding_dim_topofallfeature', type=float, default=192)
    parser.add_argument('--readout', type=str, default='mean')
    parser.add_argument('--layer_num', type=int, default=2)
    parser.add_argument('-topk', '--topk_topofallfeature', type=int, default=1)
    parser.add_argument('-n_splits', '--n_splits_topofallfeature', type=int, default=10)
    parser.add_argument('-data_folder', '--data_folder_topofallfeature', type=str, default="./Data/davis")
    parser.add_argument('-drug_sim_path', '--drug_sim_path_topofallfeature', type=str, default="drug_affinity_mat.txt",help='number of epoch.')
    parser.add_argument('-target_sim_path', '--target_sim_path_topofallfeature', type=str,default="target_affinity_mat.txt",help='number of epoch.')
    parser.add_argument('-DTI_path', '--DTI_path_topofallfeature', type=str, default="dti_mat.xlsx")
    parser.add_argument('-drug_smiles_path', '--drug_smiles_path_topofallfeature', type=str, default='drugs.xlsx')
    parser.add_argument('-target_fasta_path', '--target_fasta_path_topofallfeature', type=str, default='targets.xlsx')
    return parser.parse_args()