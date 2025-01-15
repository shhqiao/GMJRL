import torch
import os 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import KFold
from torch_geometric.data import InMemoryDataset, Batch
import torch.nn.functional as F
from torch_geometric import data as DATA
from collections.abc import Mapping, Sequence
from torchdrug import data


def get_metric(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num*np.arange(1, 1000)/1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1)-TP
    FN = real_score.sum()-TP
    TN = len(real_score.T)-TP-FP-FN
    fpr = FP/(FP+TN)
    tpr = TP/(TP+FN)
    ROC_dot_matrix = np.mat(sorted(np.column_stack((fpr, tpr)).tolist())).T
    ROC_dot_matrix.T[0] = [0, 0]
    ROC_dot_matrix = np.c_[ROC_dot_matrix, [1, 1]]
    x_ROC = ROC_dot_matrix[0].T
    y_ROC = ROC_dot_matrix[1].T
    auc = 0.5*(x_ROC[1:]-x_ROC[:-1]).T*(y_ROC[:-1]+y_ROC[1:])
    recall_list = tpr
    precision_list = TP/(TP+FP)
    PR_dot_matrix = np.mat(sorted(np.column_stack(
        (recall_list, precision_list)).tolist())).T
    PR_dot_matrix.T[0] = [0, 1]
    PR_dot_matrix = np.c_[PR_dot_matrix, [1, 0]]
    x_PR = PR_dot_matrix[0].T
    y_PR = PR_dot_matrix[1].T
    aupr = 0.5*(x_PR[1:]-x_PR[:-1]).T*(y_PR[:-1]+y_PR[1:])
    f1_score_list = 2*TP/(len(real_score.T)+TP-TN)
    accuracy_list = (TP+TN)/len(real_score.T)
    specificity_list = TN/(TN+FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return TP,FP,FN,TN,fpr,tpr,auc[0, 0], aupr[0, 0],f1_score, accuracy, recall, specificity, precision

def loss_function(pos_score,neg_score,batch_size):
    lamda = neg_score.size(0)/pos_score.size(0)
    term_one = lamda * torch.sum(torch.log(pos_score)) 
    term_two = torch.sum(torch.log(1.0-neg_score))
    term = term_one + term_two 
    coeff = (-1.0)/(batch_size*batch_size)
    # coeff = (-1.0) / batch_size
    result = coeff * term 
    return result

def read_data(data_folder,drug_sim_path,target_sim_path,DTI_path):
    SR = pd.read_csv(os.path.join(data_folder, drug_sim_path),header=None).values
    SD = pd.read_csv(os.path.join(data_folder, target_sim_path), header=None).values
    A_orig = pd.read_excel(os.path.join(data_folder, DTI_path),header=None).values
    A_orig_arr = A_orig.flatten()
    known_sample = np.nonzero(A_orig_arr)[0]
    return SR,SD,A_orig,A_orig_arr,known_sample

def get_drug_dissimmat(drug_affinity_matrix,topk):
    drug_num  = drug_affinity_matrix.shape[0]
    drug_dissim_mat = np.zeros((drug_num,topk))
    index_list = np.arange(drug_num)
    for i in range(drug_num):
        score = drug_affinity_matrix[i]
        index_score = list(zip(index_list,score))
        sorted_result = sorted(index_score,key=lambda x: x[1],reverse=False)[1:topk+1]
        drug_id_list = np.zeros(topk)
        for j in range(topk):
            drug_id_list[j] = sorted_result[j][0]            
        drug_dissim_mat[i] = drug_id_list
    return drug_dissim_mat

def get_negative_samples(mask,drug_dissimmat):    
    pos_num = np.sum(mask)
    pos_id = np.where(mask==1)
    drug_id = pos_id[0]
    t_id = pos_id[1]
    neg_mask = np.zeros_like(mask)  
    for i in range(pos_num):
        d = drug_id[i]
        t = t_id[i] 
        pos_drug = drug_dissimmat[d]
        for j in range(len(pos_drug)):
            neg_mask[pos_drug[j]][t] = 1 
    return neg_mask 

def kf_split(known_sample,n_splits):
    kf = KFold(n_splits, shuffle=True)
    train_all=[]
    test_all=[]
    for train_ind,test_ind in kf.split(known_sample):  
        train_all.append(train_ind) 
        test_all.append(test_ind)
    return train_all, test_all

def string_float(arr):
    for i in range(len(arr)):
        if isinstance(arr[i], str):
            arr[i] = float(arr[i])
        else:
            arr[i] = arr[i]
    return arr

def BCEloss(pre, ture):
    bce_loss = torch.nn.BCELoss(reduction='mean')
    loss = bce_loss(pre, ture)
    return loss


def Construct_G(A_train_mat,SR,SD):
    SR_ = np.where(SR > 0.8, 1, 0)
    SD_ = np.where(SD > 0.8, 1, 0)
    A_row1 = np.hstack((SR_,A_train_mat))
    A_row2 = np.hstack((A_train_mat.T,SD_))
    G = np.vstack((A_row1,A_row2))
    G = G.astype(np.float64)
    G = torch.FloatTensor(G)
    G = Normalize_adj(G)
    return G

def Construct_H(A_train_mat,SR,SD):
    H_row1 = np.hstack((SR,A_train_mat))
    H_row2 = np.hstack((A_train_mat.T,SD))
    H = np.vstack((H_row1,H_row2))
    H = H.astype(np.float64)
    H = torch.FloatTensor(H)
    return H

def Normalize_adj(adj):
    adj_T = adj.transpose(0,1)
    I = torch.eye(adj.shape[0])
    adj = adj + adj_T*(adj_T > adj).float() - adj*(adj_T > adj).float()
    adj = F.normalize(adj + I, p=1)
    return adj

def Global_Normalize(mat):
    max_val = np.max(mat)
    min_val = np.min(mat)
    mat = (mat-min_val)/(max_val-min_val)
    return mat

def collate(data_list):
    batch = Batch.from_data_list(data_list)
    return batch

def collate_multi_omics(samples):
    drug_atom, drug_bond = map(list, zip(*samples))
    batched_drug_atom = Batch.from_data_list(drug_atom)
    batched_drug_bond = Batch.from_data_list(drug_bond)
    return batched_drug_atom, batched_drug_bond

def graph_collate(batch):
    elem = batch[0]
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, (str, bytes)):
        return batch
    elif isinstance(elem, data.Graph):
        return elem.pack(batch)
    elif isinstance(elem, Mapping):
        return {key: graph_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, DATA.Data):
        return Batch.from_data_list(batch)
    elif isinstance(elem, Sequence):
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('Each element in list of batch should be of equal size')
        return [graph_collate(samples) for samples in zip(*batch)]

    raise TypeError("Can't collate data with type `%s`" % type(elem))




