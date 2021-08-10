import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch.nn import Parameter
import pdb
import time

class LGN_single(torch.nn.Module):
    def __init__(self,graph, num_user, num_item, dim_id, num_layer, has_id, dropout,
                 dim_latent=None):
        super(LGN_single, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.graph = graph
        self.dim_id = dim_id
        self.dim_latent = dim_latent
        # self.features = features
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout

        if self.dim_latent:
            self.embedding_user = torch.nn.Embedding(
                num_embeddings=self.num_user, embedding_dim=self.dim_latent)
            self.embedding_item = torch.nn.Embedding(
                num_embeddings=self.num_item, embedding_dim=self.dim_latent)
            nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            # self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            # self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, self.dropout, aggr=self.aggr_mode)
            # self.conv_embed_1 = BaseModel_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self,):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        # temp_features = self.MLP(features) if self.dim_latent else features
        all_emb = torch.cat([users_emb, items_emb])
        # x = F.normalize(x).to(device)

        embs = [all_emb]
        for layer in range(self.num_layer):
            all_emb = torch.sparse.mm(self.graph, all_emb)
            # pdb.set_trace()
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)

        # x_hat =h + x +h_1
        return light_out, self.embedding_user,self.embedding_item

class LightGCN(torch.nn.Module):
    def __init__(self, features,sparse_graph, batch_size, num_user, num_item, aggr_mode, concate,
                 num_layer, has_id, dim_x, reg_weight, dropout, user_item_dict,device =None):
        super(LightGCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.reg_weight = reg_weight
        self.user_item_dict = user_item_dict
        self.dropout = dropout
        self.device = device
        self.v_rep = None
        self.a_rep = None
        self.t_rep = None
        self.v_preference = None
        self.a_preference = None
        self.t_preference = None
        self.dim_latent = dim_x
        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_a = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        # self.user_graph = user_graph_index
        self.sparse_graph = sparse_graph.to(self.device)
        
        self.lgn_single = LGN_single(self.sparse_graph, num_user, num_item, dim_x,
                                num_layer=2, has_id=has_id, dropout=self.dropout, dim_latent=self.dim_latent)

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x), requires_grad=True)).to(
            self.device)
        self.result_embed = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x)))).to(self.device)

    def forward(self, user_nodes, pos_item_nodes, neg_item_nodes):
        
        representation, users_emb, items_emb = self.lgn_single()
        
        item_rep = representation[self.num_user:]
        user_rep = representation[:self.num_user]
        
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        # self.result_embed = representation
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores, users_emb, items_emb
        # return pos_scores, neg_scores,users_emb_v,items_emb_v,users_emb_a,items_emb_a,users_emb_t,items_emb_t
        # return pos_scores, neg_scores
    def loss(self, data):
        user, pos_items, neg_items = data
        pos_scores, neg_scores, users_emb, items_emb = self.forward(user.to(self.device), pos_items.to(self.device), neg_items.to(self.device))
        # pos_scores, neg_scores,users_emb_v,items_emb_v,users_emb_a,items_emb_a,users_emb_t,items_emb_t = self.forward(user.to(device), pos_items.to(device), neg_items.to(device))
        # pos_scores, neg_scores = self.forward(user.to(device), pos_items.to(device),
        #                                                             neg_items.to(device))
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        userEmb = users_emb(user.to(self.device))
        posEmb = items_emb((pos_items - self.num_user).to(self.device))
        negEmb = items_emb((neg_items - self.num_user).to(self.device))
        reg_loss = (1 / 2) * (userEmb.norm(2).pow(2) +
                              posEmb.norm(2).pow(2) +
                              negEmb.norm(2).pow(2)) / float(len(user))

        reg_loss = self.reg_weight * (reg_loss)
        return loss_value + reg_loss, reg_loss

    def gene_ranklist(self,val_data,test_data,step=20000, topk=10):
        user_tensor = self.result_embed[:self.num_user].cpu()
        item_tensor = self.result_embed[self.num_user:self.num_user+self.num_item].cpu()
        # item_tensor_cold = (self.v_gcn.MLP(self.v_feat[self.num_item:]) + self.a_gcn.MLP(self.a_feat[self.num_item:]) + self.t_gcn.MLP(self.t_feat[self.num_item:]))
        # item_tensor = torch.cat([item_tensor,item_tensor_cold.cpu()])
        # pdb.set_trace()

        start_index = 0
        end_index = self.num_user if step == None else step

        all_index_of_rank_list_tra = torch.LongTensor([])
        all_index_of_rank_list_vt = torch.LongTensor([])
        all_index_of_rank_list_tt = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            # pdb.set_trace()
            score_matrix_tra = torch.matmul(temp_user_tensor, item_tensor.t())
            score_matrix_vt =  score_matrix_tra.clone().detach()
            score_matrix_tt =  score_matrix_tra.clone().detach()

            _, index_of_rank_list_tra = torch.topk(score_matrix_tra, topk)
            all_index_of_rank_list_tra = torch.cat((all_index_of_rank_list_tra, index_of_rank_list_tra.cpu() + self.num_user),
                                               dim=0)
            # pdb.set_trace()
            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col))- self.num_user
                    score_matrix_vt[row][col] = 1e-5
                    score_matrix_tt[row][col] = 1e-5
            # pdb.set_trace()
            for i in range(len(val_data)):
                if val_data[i][0] >= start_index and val_data[i][0] < end_index:
                    row = val_data[i][0] - start_index
                    col = torch.LongTensor(list(val_data[i][1:]))- self.num_user
                    score_matrix_tt[row][col] = 1e-5
            # pdb.set_trace()
            for i in range(len(test_data)):
                if test_data[i][0] >= start_index and test_data[i][0] < end_index:
                    row = test_data[i][0] - start_index
                    col = torch.LongTensor(list(test_data[i][1:]))- self.num_user
                    score_matrix_vt[row][col] = 1e-5 
            _, index_of_rank_list_vt = torch.topk(score_matrix_vt, topk)
            all_index_of_rank_list_vt = torch.cat((all_index_of_rank_list_vt, index_of_rank_list_vt.cpu() + self.num_user),
                                               dim=0)
            _, index_of_rank_list_tt = torch.topk(score_matrix_tt, topk)
            all_index_of_rank_list_tt = torch.cat((all_index_of_rank_list_tt, index_of_rank_list_tt.cpu() + self.num_user),
                                               dim=0)

            start_index = end_index


            if end_index + step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user
        
        
        return all_index_of_rank_list_tra, all_index_of_rank_list_vt, all_index_of_rank_list_tt
    def accuracy(self,rank_list, topk=10):
        length = self.num_user
        precision_10 = recall_10 = ndcg_10 = 0.0
        precision_5 = recall_5 = ndcg_5 = 0.0
        precision_1 = recall_1 = ndcg_1 = 0.0
        for row, col in self.user_item_dict.items():
            # col = np.array(list(col))-self.num_user
            user = row
            pos_items = set(col)
            # print(pos_items)
            num_pos = len(pos_items)
            items_list_10 = rank_list[user].tolist()
            items_list_5 = items_list_10[:5]
            items_list_1 = items_list_10[:1]
            items_10 = set(items_list_10)
            items_5 = set(items_list_5)
            items_1 = set(items_list_1)

            num_hit_10 = len(pos_items.intersection(items_10))
            precision_10 += float(num_hit_10 / topk)
            recall_10 += float(num_hit_10 / num_pos)
            ndcg_score_10 = 0.0
            max_ndcg_score_10 = 0.0
            for i in range(min(num_hit_10, topk)):
                max_ndcg_score_10 += 1 / math.log2(i + 2)
            if max_ndcg_score_10 == 0:
                continue
            for i, temp_item in enumerate(items_list_10):
                if temp_item in pos_items:
                    ndcg_score_10 += 1 / math.log2(i + 2)
            ndcg_10 += ndcg_score_10 / max_ndcg_score_10

            num_hit_5 = len(pos_items.intersection(items_5))
            precision_5 += float(num_hit_5 / 5)
            recall_5 += float(num_hit_5 / num_pos)
            ndcg_score_5 = 0.0
            max_ndcg_score_5 = 0.0
            for i in range(min(num_hit_5, 5)):
                max_ndcg_score_5 += 1 / math.log2(i + 2)
            if max_ndcg_score_5 == 0:
                continue
            for i, temp_item in enumerate(items_list_5):
                if temp_item in pos_items:
                    ndcg_score_5 += 1 / math.log2(i + 2)
            ndcg_5 += ndcg_score_5 / max_ndcg_score_5

            num_hit_1 = len(pos_items.intersection(items_1))
            precision_1 += float(num_hit_1 / 1)
            recall_1 += float(num_hit_1 / num_pos)
            ndcg_score_1 = 0.0
            max_ndcg_score_1 = 0.0
            for i in range(min(num_hit_1, 1)):
                max_ndcg_score_1 += 1 / math.log2(i + 2)
            if max_ndcg_score_1 == 0:
                continue
            for i, temp_item in enumerate(items_list_1):
                if temp_item in pos_items:
                    ndcg_score_1 += 1 / math.log2(i + 2)
            ndcg_1 += ndcg_score_1 / max_ndcg_score_1

        return precision_10 / length, recall_10 / length, ndcg_10 / length, precision_5 / length, recall_5 / length, ndcg_5 / length, precision_1 / length, recall_1 / length, ndcg_1 / length

    def full_accuracy(self, val_data,rank_list,cold_start,topk=10):
        length = len(val_data)
        precision_10 = recall_10 = ndcg_10 = 0.0
        precision_5 = recall_5 = ndcg_5 = 0.0
        precision_1 = recall_1 = ndcg_1 = 0.0
        count = 0
        # pdb.set_trace()
        for data in val_data:
            user = data[0]
            pos_i = data[1:]
            pos_temp = []
            # pdb.set_trace()
            if len(pos_i)==0:
                length = length-1
                count+=1
                continue
            else:
                if cold_start == 1:
                    for item in pos_i:
                        # pdb.set_trace()
                        pos_temp.append(item-self.num_item)
                        
                else:
                    for item in pos_i:
                        # pdb.set_trace()
                        pos_temp.append(item)
                # pdb.set_trace()
            # print(pos_items)
            pos_items = set(pos_temp)

            num_pos = len(pos_items)
            items_list_10 = rank_list[user].tolist()
            items_list_5 = items_list_10[:5]
            items_list_1 = items_list_10[:1]
            items_10 = set(items_list_10)
            items_5 = set(items_list_5)
            items_1 = set(items_list_1)

            num_hit_10 = len(pos_items.intersection(items_10))
            precision_10 += float(num_hit_10 / topk)
            recall_10 += float(num_hit_10 / num_pos)
            ndcg_score_10 = 0.0
            max_ndcg_score_10 = 0.0
            for i in range(min(num_hit_10, topk)):
                max_ndcg_score_10 += 1 / math.log2(i + 2)
            if max_ndcg_score_10 == 0:
                continue
            for i, temp_item in enumerate(items_list_10):
                if temp_item in pos_items:
                    ndcg_score_10 += 1 / math.log2(i + 2)
            ndcg_10 += ndcg_score_10 / max_ndcg_score_10

            num_hit_5 = len(pos_items.intersection(items_5))
            precision_5 += float(num_hit_5 / 5)
            recall_5 += float(num_hit_5 / num_pos)
            ndcg_score_5 = 0.0
            max_ndcg_score_5 = 0.0
            for i in range(min(num_hit_5, 5)):
                max_ndcg_score_5 += 1 / math.log2(i + 2)
            if max_ndcg_score_5 == 0:
                continue
            for i, temp_item in enumerate(items_list_5):
                if temp_item in pos_items:
                    ndcg_score_5 += 1 / math.log2(i + 2)
            ndcg_5 += ndcg_score_5 / max_ndcg_score_5

            num_hit_1 = len(pos_items.intersection(items_1))
            precision_1 += float(num_hit_1 / 1)
            recall_1 += float(num_hit_1 / num_pos)
            ndcg_score_1 = 0.0
            max_ndcg_score_1 = 0.0
            for i in range(min(num_hit_1, 1)):
                max_ndcg_score_1 += 1 / math.log2(i + 2)
            if max_ndcg_score_1 == 0:
                continue
            for i, temp_item in enumerate(items_list_1):
                if temp_item in pos_items:
                    ndcg_score_1 += 1 / math.log2(i + 2)
            ndcg_1 += ndcg_score_1 / max_ndcg_score_1
        return precision_10 / length, recall_10 / length, ndcg_10 / length, precision_5 / length, recall_5 / length, ndcg_5 / length, precision_1 / length, recall_1 / length, ndcg_1 / length

