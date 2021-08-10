import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch.nn import Parameter
import pdb
class star_gcn(torch.nn.Module):
    def __init__(self,graph, num_user, num_item, num_layer,
                 dim_latent=None,device=None):
        super(star_gcn, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.device = device
        self.graph = graph
        self.dim_latent = dim_latent
        # self.features = features
        self.num_layer = num_layer

        self.W_h1 = nn.Linear(self.dim_latent,self.dim_latent,bias = False)
        self.W_3 = nn.Linear(self.dim_latent,self.dim_latent,bias = False)
        self.W_4 = nn.Linear(self.dim_latent, self.dim_latent, bias=False)

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

        else:
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(self.device))
            # self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, self.dropout,aggr=self.aggr_mode)
            # self.conv_embed_1 = BaseModel_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            # nn.init.xavier_normal_(self.conv_embed_1.weight)

    def forward(self,):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        # temp_features = self.MLP(features) if self.dim_latent else features
        x_0 = torch.cat([users_emb, items_emb])
        # x = F.normalize(x).to(device)
        # pdb.set_trace()


        all_emb = torch.sparse.mm(self.graph, x_0)
        h_1 = F.leaky_relu(all_emb, 0.1)
        h_1 = self.W_h1(h_1)
        x_1 = F.leaky_relu(self.W_3(h_1),0.1)
        x_1 = self.W_4(x_1)
        h_2 = torch.sparse.mm(self.graph, x_1)
        h_2 = F.leaky_relu(h_2, 0.1)
        h_2 = self.W_h1(h_2)
        x_2 = F.leaky_relu(self.W_3(h_2),0.1)
        x_2 = self.W_4(x_2)

        # x_hat =h + x +h_1
        return x_0,x_1,x_2, self.embedding_user,self.embedding_item,self.W_h1.weight,self.W_3.weight,self.W_4.weight


class Stargcn(torch.nn.Module):
    def __init__(self, features,sparse_graph, batch_size, num_user, num_item, aggr_mode, concate,
                 num_layer, reg_weight, user_item_dict,dim_latent,device=None):
        super(Stargcn, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.reg_weight = reg_weight
        self.user_item_dict = user_item_dict
        self.dim_latent = dim_latent
        # pdb.set_trace()
        self.sparse_graph = sparse_graph.to(self.device)
        self.stargcn = star_gcn(self.sparse_graph, num_user, num_item,
                           num_layer=3, dim_latent=self.dim_latent,device=self.device)

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user + num_item, self.dim_latent), requires_grad=True)).to(
            self.device)
        self.result_embed = nn.Parameter(nn.init.xavier_normal_(torch.rand((num_user + num_item, self.dim_latent)))).to(self.device)

    def forward(self, user_nodes, pos_item_nodes, neg_item_nodes):

        representation0,representation1,representation,users_emb,items_emb,self.W_h1,self.W_3,self.W_4= self.stargcn()
        # loss_r = ((representation0[user_nodes] - representation1[user_nodes]).norm(2).pow(2)+
        #           (representation0[pos_item_nodes] - representation1[pos_item_nodes]).norm(2).pow(2)+
        #           (representation0[neg_item_nodes] - representation1[neg_item_nodes]).norm(2).pow(2))/float(len(user_nodes))
        # loss_r_1 = ((representation0[user_nodes] - representation[user_nodes]).norm(2).pow(2) +
        #           (representation0[pos_item_nodes] - representation[pos_item_nodes]).norm(2).pow(2) +
        #           (representation0[neg_item_nodes] - representation[neg_item_nodes]).norm(2).pow(2)) / float(
        #     len(user_nodes))

        item_rep = representation[self.num_user:]
        user_rep = representation[:self.num_user]
        # h_u1 = self.user_graph(user_rep)
        # h_u2 = self.user_graph(h_u1)
        # user_rep = user_rep + h_u1 + h_u2
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        # self.result_embed = representation
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)

        item_rep_1 = representation1[self.num_user:]
        user_rep_1 = representation1[:self.num_user]
        # h_u1 = self.user_graph(user_rep)
        # h_u2 = self.user_graph(h_u1)
        # user_rep = user_rep + h_u1 + h_u2
        self.result_embed_1 = torch.cat((user_rep_1, item_rep_1), dim=0)
        # self.result_embed = representation
        user_tensor_1 = self.result_embed_1[user_nodes]
        pos_item_tensor_1 = self.result_embed_1[pos_item_nodes]
        neg_item_tensor_1 = self.result_embed_1[neg_item_nodes]
        pos_scores_1 = torch.sum(user_tensor_1 * pos_item_tensor_1, dim=1)
        neg_scores_1 = torch.sum(user_tensor_1 * neg_item_tensor_1, dim=1)
        return pos_scores, neg_scores,pos_scores_1,neg_scores_1,users_emb,items_emb

    def loss(self, data):
        user, pos_items, neg_items = data
        pos_scores, neg_scores,pos_scores_1,neg_scores_1,users_emb,items_emb = self.forward(user.to(self.device), pos_items.to(self.device), neg_items.to(self.device))
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        loss_value_1 = -torch.mean(torch.log2(torch.sigmoid(pos_scores_1 - neg_scores_1)))

        loss_value = (loss_value+loss_value_1)/2

        userEmb0 = users_emb(user.to(self.device))
        posEmb0 = items_emb((pos_items-self.num_user).to(self.device))
        negEmb0 = items_emb((neg_items-self.num_user).to(self.device))
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(user))
        reg_weight = (self.W_h1 ** 2).mean()+(self.W_3 ** 2).mean()+(self.W_3 ** 2).mean()
        reg_loss = self.reg_weight * (reg_loss + reg_weight)
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
