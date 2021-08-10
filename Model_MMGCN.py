import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from BaseModel import BaseModel
from torch_geometric.utils import scatter_
import pdb
import random
class GCN(torch.nn.Module):
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat, dim_id, aggr_mode, concate, num_layer, has_id, dim_latent=None,device=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.device = device
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id

        if self.dim_latent:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).to(self.device)
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).to(self.device)
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_feat, self.dim_id)     
            nn.init.xavier_normal_(self.g_layer1.weight)              
          
        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)

        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)  

    def forward(self, features, id_embedding):
        # pdb.set_trace()
        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features),dim=0)
        x = F.normalize(x).to(self.device)

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))#equation 1
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer1(x))#equation 5 
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer1(h)+x_hat)

        h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))#equation 1
        x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer2(x))#equation 5
        x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer2(h)+x_hat)

        h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))#equation 1
        x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer3(x))#equation 5
        x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer3(h)+x_hat)

        return x


class MMGCN(torch.nn.Module):
    def __init__(self, v_feat, a_feat, t_feat, words_tensor, edge_index, batch_size, num_user, num_item, aggr_mode, concate, num_layer, has_id, user_item_dict, reg_weight, dim_x, dataset,drop_rate,device=None):
        super(MMGCN, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dataset = dataset
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.user_item_dict = user_item_dict
        self.weight = torch.tensor([[1.0],[-1.0]]).to(self.device)
        self.reg_weight = reg_weight
        
        self.edge_index = torch.tensor(edge_index).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        self.num_modal = 0

        
        
        self.item_index= torch.zeros([self.num_item],dtype=torch.long)
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = drop_rate
        self.single_percent = 1
        self.double_percent = 0
        # pdb.set_trace()
        drop_item = torch.tensor(np.random.choice(self.item_index, int(self.num_item*self.drop_percent), replace=False))
        drop_item_single = drop_item[:int(self.single_percent*len(drop_item))]
        drop_item_double = drop_item[int(self.single_percent*len(drop_item)):]
        # pdb.set_trace()
        # self.drop_node_num = int(self.init_percent * self.num_item)
        # drop_node_idx = torch.randint(0, self.num_item, size=[self.drop_node_num])


        # self.dropv_node_idx = drop_node_idx[:int(self.drop_node_num/3)]
        # self.dropa_node_idx = drop_node_idx[int(self.drop_node_num/3):int(self.drop_node_num*2/3)]
        # self.dropt_node_idx = drop_node_idx[int(self.drop_node_num*2/3):]
        # random.shuffle(index)
        # self.item_index = self.item_index[index]
        self.dropv_node_idx_single = drop_item_single[:int(len(drop_item_single) * 1/ 3)]
        self.dropa_node_idx_single = drop_item_single[int(len(drop_item_single) * 1/ 3):int(len(drop_item_single) * 2/ 3)]
        self.dropt_node_idx_single = drop_item_single[int(len(drop_item_single) * 2 / 3):]

        self.dropv_node_idx_double = drop_item_double[:int(len(drop_item_double) * 2/ 3)]
        self.dropa_node_idx_double = torch.cat([drop_item_double[:int(len(drop_item_double) / 3)],drop_item_double[int(len(drop_item_double) * 2 / 3):]])
        self.dropt_node_idx_double = drop_item_double[int(len(drop_item_double) * 1 / 3):]

        self.dropv_node_idx = torch.cat([self.dropv_node_idx_single,self.dropv_node_idx_double])
        self.dropa_node_idx = torch.cat([self.dropa_node_idx_single,self.dropa_node_idx_double])
        self.dropt_node_idx = torch.cat([self.dropt_node_idx_single,self.dropt_node_idx_double])
        # pdb.set_trace()
        mask_cnt = torch.zeros(self.num_item, dtype=int).tolist()
        for edge in edge_index:
            mask_cnt[edge[1] - self.num_user] += 1
        mask_dropv = []
        mask_dropa = []
        mask_dropt = []
        for idx, num in enumerate(mask_cnt):
            temp_false = [False] * num
            temp_true = [True] * num
            mask_dropv.extend(temp_false) if idx in self.dropv_node_idx else mask_dropv.extend(temp_true)
            mask_dropa.extend(temp_false) if idx in self.dropa_node_idx else mask_dropa.extend(temp_true)
            mask_dropt.extend(temp_false) if idx in self.dropt_node_idx else mask_dropt.extend(temp_true)

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]
        # pdb.set_trace()
        

        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropa = edge_index[mask_dropa]
        edge_index_dropt = edge_index[mask_dropt]

        # union = set(tuple(t) for t in asdsa+asdsd)
        # pdb.set_trace()
        # [[   14 55485]
        #  [   15 55485]
        #  [   18 55485]
        #  ...
        #  [26180 61470]
        #  [50143 61470]
        #  [52399 61470]]
        # ============================================================================
        # 避免排序报错，在后面在进行这个类型转换
        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropa = torch.tensor(edge_index_dropa).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)


        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropa = torch.cat((self.edge_index_dropa, self.edge_index_dropa[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)


        self.v_feat = torch.tensor(v_feat,dtype=torch.float).to(self.device)
        self.a_feat = torch.tensor(a_feat,dtype=torch.float).to(self.device)
        if self.dataset =='Movielens':
            self.t_feat = torch.tensor(t_feat,dtype=torch.float).to(self.device)
            # pdb.set_trace()
            # self.v_drop_ze = torch.zeros(len(self.dropv_node_idx),self.v_feat.shape[1]).to(self.device)
            # self.a_drop_ze = torch.zeros(len(self.dropa_node_idx),self.a_feat.shape[1]).to(self.device)
            # self.t_drop_ze = torch.zeros(len(self.dropt_node_idx),self.t_feat.shape[1]).to(self.device)
            # self.v_feat[self.dropv_node_idx] =  self.v_drop_ze
            # self.a_feat[self.dropa_node_idx] =  self.a_drop_ze
            # self.t_feat[self.dropt_node_idx] =  self.t_drop_ze

        self.v_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.v_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, dim_latent=256,device=self.device)
        self.num_modal += 1

        
        self.a_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.a_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id,device=self.device)
        self.num_modal += 1
        if self.dataset =='Movielens':
            
            self.t_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.t_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id,device=self.device)
            self.num_modal += 1
        if self.dataset =='Tiktok':
            self.v_drop_ze = torch.zeros(len(self.dropv_node_idx),128).to(self.device)
            self.a_drop_ze = torch.zeros(len(self.dropa_node_idx),128).to(self.device)
            self.t_drop_ze = torch.zeros(len(self.dropt_node_idx),128).to(self.device)
            self.v_feat[self.dropv_node_idx] =  self.v_drop_ze
            self.a_feat[self.dropa_node_idx] =  self.a_drop_ze
            self.words_tensor = torch.tensor(words_tensor, dtype=torch.long).to(self.device)
            self.word_embedding = nn.Embedding(torch.max(self.words_tensor[1])+1, 128)
            nn.init.xavier_normal_(self.word_embedding.weight) 
            self.t_gcn = GCN(self.edge_index_dropt, batch_size, num_user, num_item, 128, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id,device=self.device)
            self.num_modal += 1

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).to(self.device)
        self.result = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).to(self.device)


    def forward(self):
        # pdb.set_trace()
        v_rep = self.v_gcn(self.v_feat, self.id_embedding)
        a_rep = self.a_gcn(self.a_feat, self.id_embedding)
        if self.dataset == 'Tiktok':
            self.t_feat = torch.tensor(scatter_('mean', self.word_embedding(self.words_tensor[1]), self.words_tensor[0])).to(self.device)
            self.t_feat[self.dropt_node_idx] =  self.t_drop_ze
            t_rep = self.t_gcn(self.t_feat, self.id_embedding)
        else:
            t_rep = self.t_gcn(self.t_feat, self.id_embedding)
        representation = (v_rep+a_rep+t_rep)/3
        # representation = v_rep

        self.result = representation
        return representation

    def loss(self,data):
        user_tensor,item_tensor = data
        user_tensor = user_tensor.view(-1)
        item_tensor = item_tensor.view(-1)
        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score*item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        reg_embedding_loss = (self.id_embedding[user_tensor]**2 + self.id_embedding[item_tensor]**2).mean()+(self.v_gcn.preference**2).mean()+(self.a_gcn.preference**2).mean()+(self.t_gcn.preference**2).mean()
        reg_loss = self.reg_weight * (reg_embedding_loss)
        return loss+reg_loss, reg_loss

    def gene_ranklist(self,val_data,test_data,step=20000, topk=10):
        user_tensor = self.result[:self.num_user].cpu()
        item_tensor = self.result[self.num_user:self.num_user+self.num_item].cpu()
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
        print(count)
        return precision_10 / length, recall_10 / length, ndcg_10 / length, precision_5 / length, recall_5 / length, ndcg_5 / length, precision_1 / length, recall_1 / length, ndcg_1 / length

