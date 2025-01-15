import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gep
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool, JumpingKnowledge
from torch_geometric.nn.norm import GraphNorm
import torch_geometric.nn as pyg_nn
from create_drug_feat import get_atom_int_feature_dims,get_bond_feature_int_dims
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch_scatter import scatter_mean, scatter_add, scatter_max, scatter_min, scatter_sum
from torchdrug.layers import functional
from torchdrug import core, data, layers, utils

class atom_embedding_net(nn.Module):
    def __init__(self, hgcn_dim) -> None:
        super(atom_embedding_net, self).__init__()
        self.embed_dim = hgcn_dim
        self.atom_embedding = nn.ModuleList()
        self.num_atom_feature = len(get_atom_int_feature_dims())
        for i in range(self.num_atom_feature):
            self.atom_embedding.append(nn.Embedding(get_atom_int_feature_dims()[i], self.embed_dim))
            torch.nn.init.xavier_uniform_(self.atom_embedding[i].weight.data)

    def forward(self, x):
        out = 0
        for i in range(self.num_atom_feature):
            out += self.atom_embedding[i](x[:, i].to(dtype=torch.int64))
        return out


class bond_embedding_net(nn.Module):
    def __init__(self, hgcn_dim) -> None:
        super(bond_embedding_net, self).__init__()
        self.embed_dim = hgcn_dim
        self.bond_embedding = nn.ModuleList()
        self.num_bond_feature = len(get_bond_feature_int_dims())
        for i in range(self.num_bond_feature):
            self.bond_embedding.append(nn.Embedding(get_bond_feature_int_dims()[i] + 3, self.embed_dim))
            torch.nn.init.xavier_uniform_(self.bond_embedding[i].weight.data)

    def forward(self, x):
        out = 0
        for i in range(self.num_bond_feature):
            out += self.bond_embedding[i](x[:, i].to(dtype=torch.int64))
        return out

## Graph geom learning
class Drug_3d_Encoder(nn.Module):
    """
    Drug_3d_Encoder is a deep learning model designed for encoding 3D molecular structures of drugs.

    Attributes:
        embed_dim (int): Dimension of the embeddings for atoms and bonds.
        dropout_rate (float): Dropout rate applied to the layers to prevent overfitting.
        layer_num (int): Number of layers in the graph convolutional network.
        readout (str): Type of readout function to aggregate node features into a graph-level representation.

    Args:
        model_config (dict): Configuration dictionary specifying various hyperparameters for the model.
    """

    def __init__(self, hgcn_dim, dropout):
        super(Drug_3d_Encoder, self).__init__()
        self.embed_dim = hgcn_dim
        self.dropout_rate = dropout
        self.layer_num = 3
        self.readout = 'mean'
        self.atom_init_nn = atom_embedding_net(hgcn_dim)
        self.bond_init_nn = bond_embedding_net(hgcn_dim)
        self.atom_conv = nn.ModuleList()
        self.bond_conv = nn.ModuleList()
        self.bond_embed_nn = nn.ModuleList()
        self.bond_angle_embed_nn = nn.ModuleList()
        self.batch_norm_atom = nn.ModuleList()
        self.layer_norm_atom = nn.ModuleList()
        self.graph_norm_atom = nn.ModuleList()
        self.batch_norm_bond = nn.ModuleList()
        self.layer_norm_bond = nn.ModuleList()
        self.graph_norm_bond = nn.ModuleList()
        self.molFC1 = nn.Linear(hgcn_dim, 1024)
        self.molFC2 = nn.Linear(1024, hgcn_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        for i in range(self.layer_num):
            self.atom_conv.append(pyg_nn.GINEConv(
                nn=nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2), nn.ReLU(),
                                 nn.Linear(self.embed_dim * 2, self.embed_dim)), edge_dim=self.embed_dim))
            self.batch_norm_atom.append(nn.BatchNorm1d(self.embed_dim))
            self.layer_norm_atom.append(nn.LayerNorm(self.embed_dim))
            self.graph_norm_atom.append(GraphNorm(self.embed_dim))
            self.bond_conv.append(pyg_nn.GINEConv(
                nn=nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim * 2), nn.ReLU(),
                                 nn.Linear(self.embed_dim * 2, self.embed_dim)), edge_dim=self.embed_dim))
            self.bond_embed_nn.append(bond_embedding_net(hgcn_dim))
            self.bond_angle_embed_nn.append(
                nn.Sequential(nn.Linear(1, self.embed_dim), nn.ReLU(), nn.Linear(self.embed_dim, self.embed_dim)))
            self.batch_norm_bond.append(nn.BatchNorm1d(self.embed_dim))
            self.batch_norm_bond.append(nn.BatchNorm1d(self.embed_dim))
            self.layer_norm_bond.append(nn.LayerNorm(self.embed_dim))
            self.graph_norm_bond.append(GraphNorm(self.embed_dim))
        if self.readout == 'max':
            self.read_out = pyg_nn.global_max_pool
        elif self.readout == 'mean':
            self.read_out = pyg_nn.global_mean_pool
        elif self.readout == 'add':
            self.read_out = pyg_nn.global_mean_pool

    def forward(self, drug_atom, drug_bond):
        x, edge_index, edge_attr, batch = drug_atom.x, drug_atom.edge_index, drug_atom.edge_attr, drug_atom.batch
        x = self.atom_init_nn(x.to(dtype=torch.int64))
        edge_hidden = self.bond_init_nn(edge_attr.to(dtype=torch.int64))
        hidden = [x]
        hidden_edge = [edge_hidden]
        for i in range(self.layer_num):
            x = self.atom_conv[i](x=x, edge_attr=hidden_edge[i], edge_index=edge_index)
            x = self.layer_norm_atom[i](x)
            x = self.graph_norm_atom[i](x)
            if i == self.layer_num - 1:
                x = nn.Dropout(self.dropout_rate)(nn.ReLU()(x)) + hidden[i]
            else:
                x = nn.Dropout(self.dropout_rate)(x) + hidden[i]
            cur_edge_attr = self.bond_embed_nn[i](edge_attr)
            cur_angle_attr = self.bond_angle_embed_nn[i](drug_bond.edge_attr)
            edge_hidden = self.bond_conv[i](x=cur_edge_attr, edge_attr=cur_angle_attr, edge_index=drug_bond.edge_index)
            edge_hidden = self.layer_norm_bond[i](edge_hidden)
            edge_hidden = self.graph_norm_bond[i](edge_hidden)
            if i == self.layer_num - 1:
                edge_hidden = nn.Dropout(self.dropout_rate)(nn.ReLU()(edge_hidden)) + hidden_edge[i]
            else:
                edge_hidden = nn.Dropout(self.dropout_rate)(edge_hidden) + hidden_edge[i]
            hidden.append(x)
            hidden_edge.append(edge_hidden)
        x = hidden[-1]
        graph_repr = self.read_out(x, batch)
        graph_repr = self.dropout(self.relu(self.molFC1(graph_repr)))
        graph_repr = self.dropout(self.molFC2(graph_repr))
        return graph_repr

    @property
    def output_dim(self):
        self.out_dim = self.embed_dim * (self.layer_num + 1)
        return self.out_dim

class Drug_encoder(nn.Module):
    def __init__(self, hgcn_dim, dropout, num_features_xd = 78):
        super().__init__()
        self.molGconv1 = GCNConv(num_features_xd, num_features_xd * 2)
        self.molGconv2 = GCNConv(num_features_xd * 2, num_features_xd * 4)
        self.molGconv3 = GCNConv(num_features_xd * 4, hgcn_dim)
        self.molFC1 = nn.Linear(hgcn_dim, 1024)
        self.molFC2 = nn.Linear(1024, hgcn_dim)
        self.bn1 = nn.LayerNorm(num_features_xd * 2)
        self.bn2 = nn.LayerNorm(num_features_xd * 4)
        self.bn3 = nn.LayerNorm(hgcn_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, drug_graph_batchs):
        x, edge_index, batch = drug_graph_batchs.x, drug_graph_batchs.edge_index, drug_graph_batchs.batch
        x = self.bn1(self.relu(self.molGconv1(x, edge_index)))
        x = self.bn2(self.relu(self.molGconv2(x, edge_index)))
        x = self.bn3(self.relu(self.molGconv3(x, edge_index)))
        x = gep(x, batch)  # global mean pooling
        x = self.dropout(self.relu(self.molFC1(x)))
        x = self.dropout(self.molFC2(x))
        return x

class AtomPositionGather(nn.Module, core.Configurable):

    def from_3_points(self, p_x_axis, origin, p_xy_plane, eps=1e-10):
        """
            Adpated from torchfold
            Implements algorithm 21. Constructs transformations from sets of 3
            points using the Gram-Schmidt algorithm.
            Args:
                x_axis: [*, 3] coordinates
                origin: [*, 3] coordinates used as frame origins
                p_xy_plane: [*, 3] coordinates
                eps: Small epsilon value
            Returns:
                A transformation object of shape [*]
        """
        p_x_axis = torch.unbind(p_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(p_x_axis, origin)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        return rots

    def forward(self, graph):
        residue_mask = \
            scatter_add((graph.atom_name == graph.atom_name2id["N"]).float(), graph.atom2residue, dim_size=graph.num_residue) + \
            scatter_add((graph.atom_name == graph.atom_name2id["CA"]).float(), graph.atom2residue, dim_size=graph.num_residue) + \
            scatter_add((graph.atom_name == graph.atom_name2id["C"]).float(), graph.atom2residue, dim_size=graph.num_residue)
        residue_mask = (residue_mask == 3)
        atom_mask = residue_mask[graph.atom2residue] & (graph.atom_name == graph.atom_name2id["CA"])
        graph = graph.subresidue(residue_mask)

        atom_pos = torch.full((graph.num_residue, len(graph.atom_name2id), 3), float("inf"), dtype=torch.float, device=graph.device)
        atom_pos[graph.atom2residue, graph.atom_name] = graph.node_position
        atom_pos_mask = torch.zeros((graph.num_residue, len(graph.atom_name2id)), dtype=torch.bool, device=graph.device)
        atom_pos_mask[graph.atom2residue, graph.atom_name] = 1

        graph = graph.subgraph(graph.atom_name == graph.atom_name2id["CA"])
        frame = self.from_3_points(
            atom_pos[:, graph.atom_name2id["N"]],
            atom_pos[:, graph.atom_name2id["CA"]],
            atom_pos[:, graph.atom_name2id["C"]]
        ).transpose(-1, -2)

        graph.view = 'residue'
        with graph.residue():
            graph.atom_pos = atom_pos
            graph.atom_pos_mask = atom_pos_mask
            graph.frame = frame

        return graph, atom_mask


class DDGAttention(nn.Module):

    def __init__(self, input_dim, output_dim, value_dim=16, query_key_dim=16, num_heads=4):
        super(DDGAttention, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_heads = num_heads

        self.query = nn.Linear(input_dim, query_key_dim * num_heads, bias=False)
        self.key = nn.Linear(input_dim, query_key_dim * num_heads, bias=False)
        self.value = nn.Linear(input_dim, value_dim * num_heads, bias=False)

        self.out_transform = nn.Linear(
            in_features=(num_heads * value_dim) + (num_heads * (3 + 3 + 1)),
            out_features=output_dim,
        )
        self.layer_norm = nn.LayerNorm(output_dim)

    def _alpha_from_logits(self, logits):
        alpha = torch.softmax(logits, dim=2)
        return alpha

    def _heads(self, x, n_heads, n_ch):
        """
        Args:
            x:  (..., num_heads * num_channels)
        Returns:
            (..., num_heads, num_channels)
        """
        s = list(x.size())[:-1] + [n_heads, n_ch]
        return x.view(*s)

    def forward(self, x, pos_CA, pos_CB, frame):
        # Attention logits
        query = self._heads(self.query(x), self.num_heads, self.query_key_dim)
        key = self._heads(self.key(x), self.num_heads, self.query_key_dim)
        logits_node = torch.einsum('blhd, bkhd->blkh', query, key)
        alpha = self._alpha_from_logits(logits_node)

        value = self._heads(self.value(x), self.num_heads, self.value_dim)
        feat_node = torch.einsum('blkh, bkhd->blhd', alpha, value).flatten(-2)

        rel_pos = pos_CB.unsqueeze(1) - pos_CA.unsqueeze(2)
        atom_pos_bias = torch.einsum('blkh, blkd->blhd', alpha, rel_pos)
        feat_distance = atom_pos_bias.norm(dim=-1)
        feat_points = torch.einsum('blij, blhj->blhi', frame, atom_pos_bias)
        feat_direction = feat_points / (feat_points.norm(dim=-1, keepdim=True) + 1e-10)
        feat_spatial = torch.cat([
            feat_points.flatten(-2),
            feat_distance,
            feat_direction.flatten(-2),
        ], dim=-1)

        feat_all = torch.cat([feat_node, feat_spatial], dim=-1)
        feat_all = self.out_transform(feat_all)
        if x.shape[-1] == feat_all.shape[-1]:
            x_updated = self.layer_norm(x + feat_all)
        else:
            x_updated = self.layer_norm(feat_all)

        return x_updated

class Target_encoder(nn.Module):
    def __init__(self, hgcn_dim, dropout, num_features_pro = 33):
        super().__init__()
        self.proGconv1 = GCNConv(num_features_pro, hgcn_dim)
        self.proGconv2 = GCNConv(hgcn_dim, hgcn_dim)
        self.proGconv3 = GCNConv(hgcn_dim, hgcn_dim)
        self.attn_layers1 = DDGAttention(hgcn_dim, hgcn_dim)
        self.attn_layers2 = DDGAttention(hgcn_dim, hgcn_dim)
        self.attn_layers3 = DDGAttention(hgcn_dim, hgcn_dim)
        self.proFC1 = nn.Linear(hgcn_dim, 1024)
        self.proFC2 = nn.Linear(1024, hgcn_dim)
        self.bn1 = nn.LayerNorm(hgcn_dim)
        self.bn2 = nn.LayerNorm(hgcn_dim)
        self.bn3 = nn.LayerNorm(hgcn_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.atom_position_gather = AtomPositionGather()
        self.readout = layers.MeanReadout()

    def forward(self, graph, target_graph_batchs):
        residue_graph, _ = self.atom_position_gather(graph)
        pos_CA, _ = functional.variadic_to_padded(residue_graph.node_position, residue_graph.num_nodes, value=0)
        pos_CB = torch.where(
            residue_graph.atom_pos_mask[:, residue_graph.atom_name2id["CB"], None].expand(-1, 3),
            residue_graph.atom_pos[:, residue_graph.atom_name2id["CB"]],
            residue_graph.atom_pos[:, residue_graph.atom_name2id["CA"]]
        )
        pos_CB, _ = functional.variadic_to_padded(pos_CB, residue_graph.num_nodes, value=0)
        frame, _ = functional.variadic_to_padded(residue_graph.frame, residue_graph.num_nodes, value=0)
        p_x, p_edge_index, p_batch = target_graph_batchs.x, target_graph_batchs.edge_index, target_graph_batchs.batch
        p_x = self.bn1(self.relu(self.proGconv1(p_x, p_edge_index)))
        x, _ = functional.variadic_to_padded(p_x, residue_graph.num_nodes, value=0)
        residue_hidden = self.attn_layers1(x, pos_CA, pos_CB, frame)
        residue_hidden = functional.padded_to_variadic(residue_hidden, residue_graph.num_nodes)
        p_x += residue_hidden
        p_x = self.bn2(self.relu(self.proGconv2(p_x, p_edge_index)))
        x, _ = functional.variadic_to_padded(p_x, residue_graph.num_nodes, value=0)
        residue_hidden = self.attn_layers2(x, pos_CA, pos_CB, frame)
        residue_hidden = functional.padded_to_variadic(residue_hidden, residue_graph.num_nodes)
        p_x += residue_hidden
        # p_x = self.bn3(self.relu(self.proGconv3(p_x, p_edge_index)))
        # x, _ = functional.variadic_to_padded(p_x, residue_graph.num_nodes, value=0)
        # residue_hidden = self.attn_layers3(x, pos_CA, pos_CB, frame)
        # residue_hidden = functional.padded_to_variadic(residue_hidden, residue_graph.num_nodes)
        # p_x += residue_hidden
        graph_feature = self.readout(residue_graph, p_x)
        graph_feature = self.dropout(self.relu(self.proFC1(graph_feature)))
        graph_feature = self.dropout(self.proFC2(graph_feature))
        return graph_feature

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 drop_ratio=0.,
                 ):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)

    def forward(self, x):
        B, C = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        y = (attn @ v).reshape(B, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        return x + y

class Cross_Attention(nn.Module):
    def __init__(self,
                 dim,
                 drop_ratio=0.,
                 num_heads=8,
                 qkv_bias=False
                 ):
        super(Cross_Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop_ratio)

    def forward(self, x, y):
        B, C = x.shape
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3)
        qkv2 = self.qkv(y).reshape(B, 3, self.num_heads, C // self.num_heads).permute(1, 0, 2, 3)
        q, k, v = qkv2[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).reshape(B, C)
        return out

class Interaction(nn.Module):
    def __init__(self,
                 dim,
                 drop_ratio=0.,
                 num_heads=8,
                 norm_layer=nn.LayerNorm
                 ):
        super(Interaction, self).__init__()
        self.norm = norm_layer(dim)
        self.cross_attn = Cross_Attention(dim, drop_ratio=drop_ratio, num_heads=num_heads)

    def forward(self, x, y):
        out = y + self.cross_attn(self.norm(x), self.norm(y))
        return out


class GraphConvolution(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.weight = nn.Parameter(torch.FloatTensor(in_feature, out_feature))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_feature))
        nn.init.xavier_normal_(self.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.sparse.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output + self

class GCN_decoder(nn.Module):
    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.gc1 = GraphConvolution(in_dim, hgcn_dim)
        self.gc2 = GraphConvolution(hgcn_dim, hgcn_dim)
        # self.gc3 = GraphConvolution(hgcn_dim, hgcn_dim)
        self.dropout = dropout

    def forward(self, H, G):
        H = self.gc1(H, G)
        H = F.leaky_relu(H, 0.25)
        H = F.dropout(H, self.dropout, training=True)
        H = self.gc2(H, G)
        H = F.leaky_relu(H, 0.25)
        # H = F.dropout(H, self.dropout, training=True)
        # H = self.gc3(H, G)
        # H = F.leaky_relu(H, 0.25)
        return H

class GMJRL(nn.Sequential):

    def __init__(self, in_dim, hgcn_dim, dropout):
        super().__init__()
        self.drug_3d = Drug_3d_Encoder(hgcn_dim, dropout)
        self.target_encoder = Target_encoder(hgcn_dim = hgcn_dim, dropout = dropout)
        self.GCN_decoder = GCN_decoder(in_dim, hgcn_dim, dropout)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(4 * hgcn_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 1)
        self.fc4.weight.data.normal_()
        self.fc4.bias.data.normal_()
        self.drug_ffn = Attention(dim=hgcn_dim, drop_ratio=dropout)
        self.target_ffn = Attention(dim=hgcn_dim, drop_ratio=dropout)
        self.interaction = Interaction(dim=hgcn_dim * 2, drop_ratio=dropout)

    def forward(self, H, G, drug_atom, drug_bond, protein_graph, target_graph, drug_num, target_num, batch_drug_indice, batch_target_indice):
        feature = self.GCN_decoder(H,G)
        v_smiles = feature[0:drug_num]
        v_fasta = feature[drug_num:]
        v_drug_graph = self.drug_3d(drug_atom, drug_bond)
        v_target_graph = self.target_encoder(protein_graph, target_graph)
        drug_feature = torch.cat((v_smiles,v_drug_graph),0)
        target_feature = torch.cat((v_fasta, v_target_graph), 0)
        drug_ffn_feature = self.drug_ffn(drug_feature)
        target_ffn_feature = self.target_ffn(target_feature)
        x = torch.cat((drug_ffn_feature[0:drug_num], drug_ffn_feature[drug_num:]), 1)
        y = torch.cat((target_ffn_feature[0:target_num], target_ffn_feature[target_num:]), 1)
        out1 = self.interaction(x[batch_drug_indice], y[batch_target_indice])
        out2 = self.interaction(y[batch_target_indice], x[batch_drug_indice])
        v = torch.cat((out1, out2), 1)
        v = F.leaky_relu(self.fc1(v))
        v = self.dropout(v)
        v = F.leaky_relu(self.fc2(v))
        v = self.dropout(v)
        v = self.fc3(v)
        v = self.fc4(v)
        result = torch.sigmoid(v)
        return result