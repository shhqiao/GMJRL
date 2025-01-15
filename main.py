import pickle
import random
from model import GMJRL
from feature_extraction import GraphDataset, protein_graph, multi_DRP_dataset
from utils import *
torch.cuda.manual_seed(1223)
from config_init import parse_args
from create_drug_feat import *
from torch_geometric import data
import os
import gc

if __name__=="__main__":
    args = parse_args()
    readout = args.readout
    layer_num = args.layer_num
    hgcn_dim = args.embedding_dim_topofallfeature
    dropout = args.dropout_topofallfeature
    epoch_num = args.epoch_num_topofallfeature
    lr = args.lr_topofallfeature
    device = args.device_topofallfeature
    batch_size = args.batch_size_topofallfeature
    topk = args.topk_topofallfeature
    n_splits = args.n_splits_topofallfeature
    data_folder = args.data_folder_topofallfeature
    drug_sim_path = args.drug_sim_path_topofallfeature
    target_sim_path = args.target_sim_path_topofallfeature
    DTI_path = args.DTI_path_topofallfeature

    proteins = protein_graph()
    drug_atom_dict, drug_bond_dict = load_drug_feat()
    SR,SD,A_orig,A_orig_arr,known_sample = read_data(data_folder,drug_sim_path,target_sim_path,DTI_path)
    drug_num = A_orig.shape[0]
    target_num = A_orig.shape[1]
    SR = SR[300:]
    SR = SR.flatten()
    SR = string_float(SR)
    SR = SR.reshape(drug_num, drug_num)
    SD = SD[200:]
    SD = SD.flatten()
    SD = string_float(SD)
    SD = SD.reshape(target_num, target_num)
    SR = Global_Normalize(SR)
    SD = Global_Normalize(SD)
    drug_dissimmat = get_drug_dissimmat(SR, topk=topk).astype(int)
    train_all, test_all = kf_split(known_sample,n_splits)
    overall_auroc = 0
    overall_aupr = 0
    overall_f1 = 0
    overall_acc = 0
    overall_recall = 0
    overall_specificity = 0
    overall_precision = 0
    with open(f'{data_folder}/pro_data.pkl', 'rb') as file2:
        pro_data = pickle.load(file2)
    if len(pro_data) == 379:
        catch = 'LKMNKLPSNRGNTLREVQLMNRLRHPNILRFMGVCVHQGQLHALTEYMNGGTLEXLLSSPEPLSWPVRLHLALDIARGLRYLHSKGVFHRDLTSKNCLVRREDRGFTAVVGDFGLAEKIPVYREGARKEPLAVVGSPYWMAPEVLRGELYDEKADVFAFGIVLCELIARVPADPDYLPRTEDFGLDVPAFRTLVGDDCPLPFLLLAIHCCNLEPSTRAPFTEITQHLEWILEQLPEPAPLTXTA'
        rows_to_delete = [54, 241]
        temp_list = list(pro_data[catch])
        temp_list[0] = 242
        temp_list[1] = np.delete(temp_list[1], rows_to_delete, axis=0)
        pro_data[catch] = tuple(temp_list)
    if len(pro_data) == 1447:
        catch = 'P07203'
        rows_to_delete = [48]
        temp_list = list(pro_data[catch])
        temp_list[0] = 202
        temp_list[1] = np.delete(temp_list[1], rows_to_delete, axis=0)
        pro_data[catch] = tuple(temp_list)
    drug_graphs_Data = multi_DRP_dataset(drug_atom_dict, drug_bond_dict)
    target_graphs_Data = GraphDataset(graphs_dict=pro_data, dttype="target")
    drug_graphs_DataLoader = torch.utils.data.DataLoader(drug_graphs_Data, shuffle=False, collate_fn=graph_collate, batch_size=drug_num)
    target_graphs_DataLoader = torch.utils.data.DataLoader(target_graphs_Data, shuffle=False, collate_fn=graph_collate, batch_size=target_num)
    protein_Dataloader = torch.utils.data.DataLoader(proteins, shuffle=False, collate_fn=graph_collate, batch_size=target_num)
    drug_atom = next(iter(drug_graphs_DataLoader))[0].to(device)
    drug_bond = next(iter(drug_graphs_DataLoader))[1].to(device)
    target_graphs = next(iter(target_graphs_DataLoader)).to(device)
    protein_graphs = next(iter(protein_Dataloader)).to(device)
    for fold_int in range(10):
        print('fold_int:',fold_int)
        A_train_id = train_all[fold_int]
        A_test_id = test_all[fold_int]
        A_train = known_sample[A_train_id]
        A_test = known_sample[A_test_id]
        A_unknown_mask = 1 - A_orig
        A_train_list = np.zeros_like(A_orig_arr)
        A_train_list[A_train] = 1
        A_test_list = np.zeros_like(A_orig_arr)
        A_test_list[A_test] = 1
        A_train_mask = A_train_list.reshape((A_orig.shape[0],A_orig.shape[1]))
        A_train_mat = A_train_mask
        train_neg_mask_candidate = get_negative_samples(A_train_mask, drug_dissimmat)
        train_neg_mask = np.multiply(train_neg_mask_candidate, A_unknown_mask)
        G = Construct_G(A_train_mat, SR, SD).to(device)
        H = Construct_H(A_train_mat, SR, SD).to(device)
        A_test_mask = A_test_list.reshape((A_orig.shape[0],A_orig.shape[1]))
        pos_train_dti = np.where(A_train_mask==1)
        pos_drug_index = pos_train_dti[0]
        pos_target_index = pos_train_dti[1]
        pos_labels = torch.ones(len(pos_drug_index))
        neg_train_dti = np.where(train_neg_mask == 1)
        neg_drug_index = neg_train_dti[0].astype(np.int64)
        neg_target_index = neg_train_dti[1].astype(np.int64)
        neg_labels = torch.zeros(len(neg_drug_index))
        drug_index = np.hstack((pos_drug_index,neg_drug_index))
        target_index = np.hstack((pos_target_index,neg_target_index))
        drug_index = torch.from_numpy(drug_index).long()
        target_index = torch.from_numpy(target_index).long()
        train_labels = torch.cat((pos_labels,neg_labels), 0)
        train_idx = torch.arange(target_index.size(0))
        model = GMJRL(in_dim=H.size(0), hgcn_dim=hgcn_dim, dropout=dropout).to(device)
        optimizer = torch.optim.Adam(list(model.parameters()),lr=lr)
        # train_loader = torch.utils.data.DataLoader(dataset=train_idx, batch_size=batch_size, shuffle=True)
        train_loader = torch.utils.data.DataLoader(dataset=train_idx, batch_size=batch_size, shuffle=True, drop_last=True)
        for epoch in range(epoch_num):
            print("epoch:",epoch)
            model.train()
            for _, train_batch_idx in enumerate(train_loader):
                batch_drug_indice = drug_index[train_batch_idx].numpy()
                batch_target_indice = target_index[train_batch_idx].numpy()
                batch_labels = train_labels[train_batch_idx]
                y = model(H, G, drug_atom, drug_bond, protein_graphs, target_graphs,
                          drug_num, target_num, batch_drug_indice, batch_target_indice).view(-1)
                pos_score = y[batch_labels == 1].to(device)
                neg_score = y[batch_labels == 0].to(device)
                loss = loss_function(pos_score, neg_score, batch_size)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                los_  = loss.detach().cpu().item()
            print('los_:',los_)
        model.eval()
        pos_test_dti = np.where(A_test_mask==1)
        pos_test_drug_index = pos_test_dti[0]
        pos_test_target_index = pos_test_dti[1]
        pos_test_labels = torch.ones(len(pos_test_drug_index))
        """
        test negative samples
        """
        test_neg_mask_candidate = get_negative_samples(A_test_mask, drug_dissimmat)
        test_neg_mask_init = np.multiply(test_neg_mask_candidate, A_unknown_mask)
        test_neg_mask = np.multiply(test_neg_mask_init, 1-train_neg_mask)
        neg_test_dti = np.where(test_neg_mask == 1)
        neg_test_drug_index = neg_test_dti[0].astype(np.int64)
        neg_test_target_index = neg_test_dti[1].astype(np.int64)
        neg_test_labels = torch.zeros(len(neg_test_drug_index), dtype=torch.int64)
        test_drug_index = np.hstack((pos_test_drug_index,neg_test_drug_index))
        test_target_index = np.hstack((pos_test_target_index,neg_test_target_index))
        drug_test_index = torch.from_numpy(test_drug_index).long()
        target_test_index = torch.from_numpy(test_target_index).long()
        test_idx = torch.arange(target_test_index.size(0))
        test_labels = torch.cat((pos_test_labels,neg_test_labels), 0)
        test_loader = torch.utils.data.DataLoader(dataset=test_idx, batch_size=batch_size, shuffle=False)
        total_preds = torch.Tensor()
        for _, test_batch_idx in enumerate(test_loader):
            test_batch_drug_indice = drug_test_index[test_batch_idx].numpy()
            test_batch_target_indice = target_test_index[test_batch_idx].numpy()
            output = model(H, G, drug_atom, drug_bond, protein_graphs, target_graphs,
                      drug_num, target_num, test_batch_drug_indice, test_batch_target_indice).view(-1)
            total_preds = torch.cat((total_preds, output.detach().cpu()), 0)
        TP,FP,FN,TN,fpr,tpr,auroc,aupr,f1_score, accuracy, recall, specificity, precision = get_metric(test_labels.numpy(),total_preds.numpy())
        # print('TP:',TP)
        # print('FP:',FP)
        # print('FN:',FN)
        # print('TN:',FN)
        # print('fpr:',fpr)
        # print('tpr:',tpr)
        print('auroc:', auroc)
        print('aupr:', aupr)
        print('f1_score:', f1_score)
        print('acc:', accuracy)
        print('recall:', recall)
        print('specificity:', specificity)
        print('precision:', precision)
        overall_auroc += auroc
        overall_aupr += aupr
        overall_f1 += f1_score
        overall_acc += accuracy
        overall_recall += recall
        overall_specificity += specificity
        overall_precision += precision
    auroc_ = overall_auroc / n_splits
    aupr_ = overall_aupr / n_splits
    f1_ = overall_f1 / n_splits
    acc_ = overall_acc / n_splits
    recall_ = overall_recall / n_splits
    specificity_ = overall_specificity / n_splits
    precision_ = overall_precision / n_splits
    print('mean_auroc:', auroc_)
    print('mean_aupr:', aupr_)
    print('mean_f1:', f1_)
    print('mean_acc:', acc_)
    print('mean_recall:', recall_)
    print('mean_specificity:', specificity_)
    print('mean_precision:', precision_)
