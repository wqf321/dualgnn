import math
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import pdb
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter_
from torch_geometric.utils import add_self_loops, dropout_adj
##########################################################################

class VBPR_net(torch.nn.Module):
    def __init__(self, num_user, num_item,reg_weight, dim_E, feature,user_item_dict,dataset,device=None):
        super(VBPR_net, self).__init__()
        self.num_user = num_user
        self.user_item_dict = user_item_dict
        self.dataset = dataset
        self.v_feat,self.a_feat,self.t_feat  = feature
        self.dim_tik = 128
        self.num_item = num_item
        self.device = device
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)
        self.reg_weight = reg_weight
        dim_E = 64
        if self.dataset =='Movielens':
            self.user_pref = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, dim_E*2))))
            self.item_cf = nn.Parameter(nn.init.xavier_normal_(torch.rand((self.num_item, dim_E))))
            self.feature = np.hstack(feature)
            self.feature = torch.tensor(self.feature,dtype=torch.float).to(self.device)
            self.MLP = nn.Linear(self.feature.shape[1], dim_E*4)
            self.MLP_1 = nn.Linear(dim_E*4, dim_E)
        elif self.dataset =='Tiktok':
            self.user_pref = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user, self.dim_tik*2))))
            self.item_cf = nn.Parameter(nn.init.xavier_normal_(torch.rand((self.num_item, self.dim_tik))))
            self.word_tensor = self.t_feat.long().to(self.device)
            self.v_feat = self.v_feat.float().to(self.device)
            self.a_feat = self.a_feat.float().to(self.device)
            self.MLP = nn.Linear(self.dim_tik*3, self.dim_tik*2)
            self.MLP_1 = nn.Linear(self.dim_tik*2, self.dim_tik)
            self.word_embedding = nn.Embedding(torch.max(self.word_tensor[1])+1,self.dim_tik)
            nn.Parameter(nn.init.xavier_normal_(self.word_embedding.weight))
        self.result = nn.init.xavier_normal_(torch.rand((num_user+num_item, self.dim_tik))).to(self.device)
        # pdb.set_trace()
    def forward(self):
        # pdb.set_trace()
        if self.dataset =='Tiktok':
            self.t_feat = scatter_('mean',self.word_embedding(self.word_tensor[1]),self.word_tensor[0]).to(self.device)
            self.feature = torch.cat([self.v_feat,self.a_feat,self.t_feat],dim=1)
            temp = self.MLP_1(F.leaky_relu(self.MLP(self.feature)))
        else:    
            temp = self.MLP_1(F.leaky_relu(self.MLP(self.feature)))
        return self.user_pref, torch.cat((self.item_cf,temp),dim=1)

    def loss(self, data):
        # pdb.set_trace()
        user_tensor,item_tensor = data
        user_tensor = user_tensor.view(-1)
        item_tensor = item_tensor.view(-1)
        u_out, i_out = self.forward()
        user_score = u_out[user_tensor]
        item_score = i_out[item_tensor-self.num_user]
        # pdb.set_trace()
        score = torch.sum(user_score*item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        if self.dataset == 'Tiktok':
            reg_embedding_loss = (self.user_pref[user_tensor]**2).mean()+(self.item_cf[item_tensor-self.num_user]**2).mean()+(self.MLP.weight**2).mean()+(self.MLP_1.weight**2).mean()+(self.word_embedding.weight**2).mean() #(self.id_embedding[user_tensor]**2 + self.id_embedding[item_tensor]**2).mean()
        else:
            reg_embedding_loss = (self.user_pref[user_tensor]**2).mean()+(self.item_cf[item_tensor-self.num_user]**2).mean()+(self.MLP.weight**2).mean()+(self.MLP_1.weight**2).mean()
        reg_loss = self.reg_weight * (reg_embedding_loss)

        self.result = torch.cat((u_out, i_out), 0)
        # return loss+reg_loss, loss, reg_loss
        return loss+reg_loss,reg_loss
    def gene_ranklist(self,val_data,test_data,step=20000, topk=10):
        user_tensor = self.user_pref.cpu()
        if self.dataset =='Tiktok':
            self.t_feat = scatter_('mean',self.word_embedding(self.word_tensor[1]),self.word_tensor[0]).to(self.device)
            self.feature = torch.cat([self.v_feat,self.a_feat,self.t_feat],dim=1)
            temp = self.MLP_1(F.leaky_relu(self.MLP(self.feature))).cpu()
            item_tensor = torch.cat((self.item_cf.cpu(), temp), dim=1).cpu()
        else:
            temp = self.MLP_1(F.leaky_relu(self.MLP(self.feature))).cpu()
            item_tensor = torch.cat((self.item_cf.cpu(),temp),dim=1).cpu()
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
    # @decorator
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
                        pos_temp.append(item-self.num_non_cold)
                        
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
        print(count)
        return precision_10 / length, recall_10 / length, ndcg_10 / length, precision_5 / length, recall_5 / length, ndcg_5 / length, precision_1 / length, recall_1 / length, ndcg_1 / length

