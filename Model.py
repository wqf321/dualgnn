import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dropout_adj
from torch_geometric.utils import scatter_
from torch.nn import Parameter
from Basicgcn import Base_gcn
import pdb
import random
from matplotlib import cbook
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import random
class GCN(torch.nn.Module):
    def __init__(self,datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None,device = None,features=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        if self.datasets =='tiktok' or self.datasets =='tiktok_new' or self.datasets == 'cold_tiktok':
             self.dim_feat = 128
        elif self.datasets == 'Movielens' or self.datasets == 'cold_movie':
             self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        if self.dim_latent:
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user, self.dim_latent),dtype=torch.float32, requires_grad=True), gain=1).to(self.device))
            self.MLP = nn.Linear(self.dim_feat, 4*self.dim_latent)
            self.MLP_1 = nn.Linear(4*self.dim_latent, self.dim_latent)
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

        else:
            self.preference = nn.Parameter(
                nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user, self.dim_feat),dtype=torch.float32, requires_grad=True), gain=1).to(self.device))
            self.conv_embed_1 = Base_gcn(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)

    def forward(self, edge_index_drop,edge_index,features):
        # pdb.set_trace()
        temp_features = self.MLP_1(F.leaky_relu(self.MLP(features))) if self.dim_latent else features
        # temp_features = F.normalize(temp_features)
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        # pdb.set_trace()
        h = self.conv_embed_1(x, edge_index)  # equation 1
        h_1 = self.conv_embed_1(h, edge_index)

        x_hat =h + x +h_1
        return x_hat, self.preference


class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode,dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features,user_graph,user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix,u_features)
        u_pre = u_pre.squeeze()
        return u_pre

class DualGNN(torch.nn.Module):
    def __init__(self, features, edge_index,batch_size, num_user, num_item, aggr_mode, construction,
                 num_layer, has_id, dim_x, reg_weight, drop_rate,sampling, user_item_dict,dataset,cold_start,device = None):
        super(DualGNN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = sampling
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.cold_start = cold_start
        self.dataset = dataset
        self.construction = construction
        self.reg_weight = reg_weight
        self.user_item_dict = user_item_dict
        self.drop_rate = drop_rate
        self.v_rep = None
        self.a_rep = None
        self.t_rep = None
        self.device = device
        self.v_preference = None
        self.a_preference = None
        self.t_preference = None
        self.dim_latent = 64
        self.dim_feat=128
        self.MLP_v = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_a = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        self.MLP_t = nn.Linear(self.dim_latent, self.dim_latent, bias=False)
        
        v_feat,a_feat,t_feat = features
        # pdb.set_trace()
        self.edge_index = torch.tensor(edge_index,dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)
        # pdb.set_trace()
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_user, 3, 1),dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u.data, dim=1)
        
        self.weight_i = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(self.num_item, 3, 1),dtype=torch.float32, requires_grad=True)))
        self.weight_i.data = F.softmax(self.weight_i.data, dim=1)

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

        # self.dropv_node_idx_double = drop_item_double[:int(len(drop_item_double) * 2/ 3)]
        # self.dropa_node_idx_double = torch.cat([drop_item_double[:int(len(drop_item_double) / 3)],drop_item_double[int(len(drop_item_double) * 2 / 3):]])
        # self.dropt_node_idx_double = drop_item_double[int(len(drop_item_double) * 1 / 3):]

        self.dropv_node_idx = self.dropv_node_idx_single
        self.dropa_node_idx = self.dropa_node_idx_single
        self.dropt_node_idx = self.dropt_node_idx_single
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

        edge_index_dropv = edge_index[mask_dropv]
        edge_index_dropa = edge_index[mask_dropa]
        edge_index_dropt = edge_index[mask_dropt]

        self.edge_index_dropv = torch.tensor(edge_index_dropv).t().contiguous().to(self.device)
        self.edge_index_dropa = torch.tensor(edge_index_dropa).t().contiguous().to(self.device)
        self.edge_index_dropt = torch.tensor(edge_index_dropt).t().contiguous().to(self.device)


        self.edge_index_dropv = torch.cat((self.edge_index_dropv, self.edge_index_dropv[[1, 0]]), dim=1)
        self.edge_index_dropa = torch.cat((self.edge_index_dropa, self.edge_index_dropa[[1, 0]]), dim=1)
        self.edge_index_dropt = torch.cat((self.edge_index_dropt, self.edge_index_dropt[[1, 0]]), dim=1)
        if self.dataset == 'Movielens' or self.dataset == 'cold_movie':
            self.MLP_user = nn.Linear(self.dim_latent*3, self.dim_latent)
            self.v_feat = torch.tensor(v_feat, dtype=torch.float).to(self.device)
            self.a_feat = torch.tensor(a_feat, dtype=torch.float).to(self.device)
            self.t_feat = torch.tensor(t_feat, dtype=torch.float).to(self.device)
            self.v_drop_ze = torch.zeros(len(self.dropv_node_idx),self.v_feat.size(1)).to(self.device)
            self.a_drop_ze = torch.zeros(len(self.dropa_node_idx),self.a_feat.size(1)).to(self.device)
            self.t_drop_ze = torch.zeros(len(self.dropt_node_idx),self.t_feat.size(1)).to(self.device)

            # self.v_feat[self.dropv_node_idx] =  self.v_drop_ze
            # self.a_feat[self.dropa_node_idx] =  self.a_drop_ze
            # self.t_feat[self.dropt_node_idx] =  self.t_drop_ze
            # for node in drop_node_idx:
            #     if node in dropv_node_idx:
            #         self.v_feat[node] = torch.zeros(1, self.v_feat.size(1))
            #     if node in dropa_node_idx:
            #         self.a_feat[node] = torch.zeros(1, self.a_feat.size(1))
            #     if node in dropt_node_idx:
            #         self.t_feat[node] = torch.zeros(1, self.t_feat.size(1))
            self.v_gcn = GCN(self.dataset,batch_size, num_user, num_item, dim_x, self.aggr_mode,
                               num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                               device=self.device, features=self.v_feat)  # 256)
            self.a_gcn = GCN(self.dataset,batch_size, num_user, num_item, dim_x, self.aggr_mode,
                               num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                               device=self.device, features=self.a_feat)
            self.t_gcn = GCN(self.dataset,batch_size, num_user, num_item, dim_x, self.aggr_mode,
                               num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=64,
                               device=self.device, features=self.t_feat)

        elif self.dataset == 'tiktok' or self.dataset == 'tiktok_new':
            self.MLP_user = nn.Linear(self.dim_feat*3, self.dim_feat)
            
            self.word_tensor = t_feat.long().to(self.device)
            self.v_feat = v_feat.float().to(self.device)
            self.a_feat = a_feat.float().to(self.device)

            # self.v_drop_ze = torch.zeros(len(self.dropv_node_idx),128).to(self.device)
            # self.a_drop_ze = torch.zeros(len(self.dropa_node_idx),128).to(self.device)
            # self.t_drop_ze = torch.zeros(len(self.dropt_node_idx),128).to(self.device)

            # self.v_feat[self.dropv_node_idx] =  self.v_drop_ze
            # self.a_feat[self.dropa_node_idx] =  self.a_drop_ze
            # pdb.set_trace()
            self.word_embedding = nn.Embedding(torch.max(self.word_tensor[1])+1,128)
            nn.Parameter(nn.init.xavier_normal_(self.word_embedding.weight))
            # self.t_feat = nn.init.xavier_normal_(torch.rand(num_item,128)).to(device)

            self.v_gcn = GCN(self.dataset,batch_size, num_user, num_item, dim_x, self.aggr_mode,
                               num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=None,device = self.device)  # 128
            self.a_gcn = GCN(self.dataset,batch_size, num_user, num_item, dim_x, self.aggr_mode,
                               num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=None,device = self.device)
            self.t_gcn = GCN(self.dataset,batch_size, num_user, num_item, dim_x, self.aggr_mode,
                               num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=None,device = self.device)



        self.user_graph = User_Graph_sample(num_user, 'add',self.dim_latent)

        # self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user + num_item, dim_x), requires_grad=True)).to(
        #     self.device)
        self.result_embed = nn.Parameter(nn.init.xavier_normal_(torch.tensor(np.random.randn(num_user + num_item, dim_x)))).to(self.device)
    def draw_zhexian(self,index,values,path):
        fig, ax = plt.subplots()
        x = index
        y = values
        ax.plot(x, y, 'ko-')
        plt.savefig(path)
    def forward(self, user_nodes, pos_item_nodes, neg_item_nodes,user_graph,user_weight_matrix,user_cons=None):
        # edge_index, _ = dropout_adj(self.edge_index, edge_attr=None, p=self.dropout)
        if self.dataset == 'tiktok' or self.dataset=='tiktok_new':
            self.t_feat = scatter_('mean',self.word_embedding(self.word_tensor[1]),self.word_tensor[0]).to(self.device)


        self.v_rep, self.v_preference = self.v_gcn(self.edge_index_dropv,self.edge_index,self.v_feat)
        self.a_rep, self.a_preference = self.a_gcn(self.edge_index_dropa,self.edge_index,self.a_feat)
        self.t_rep, self.t_preference = self.t_gcn(self.edge_index_dropt,self.edge_index,self.t_feat)
        # ########################################### multi-modal information construction
        representation = self.v_rep+self.a_rep+self.t_rep
        
        # pdb.set_trace()
        if self.construction == 'weighted_sum':
            self.v_rep = torch.unsqueeze(self.v_rep, 2)
            self.a_rep = torch.unsqueeze(self.a_rep, 2)
            self.t_rep = torch.unsqueeze(self.t_rep, 2)
            user_rep = torch.matmul(
                torch.cat((self.v_rep[:self.num_user], self.a_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2),
                self.weight_u)
            user_rep = torch.squeeze(user_rep)

        if self.construction == 'mean':
            user_rep = (self.v_rep[:self.num_user]+self.a_rep[:self.num_user]+self.t_rep[:self.num_user])/3
        if self.construction == 'max':
            # pdb.set_trace()
            self.v_rep = torch.unsqueeze(self.v_rep, 2)
            self.a_rep = torch.unsqueeze(self.a_rep, 2)
            self.t_rep = torch.unsqueeze(self.t_rep, 2)
            
            user_rep = torch.cat((self.v_rep[:self.num_user], self.a_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
            user_rep = self.weight_u.transpose(1,2)*user_rep
            user_rep = torch.max(user_rep,dim=2).values
        if self.construction == 'cat_mlp':
            # pdb.set_trace()
            self.v_rep = torch.unsqueeze(self.v_rep, 2)
            self.a_rep = torch.unsqueeze(self.a_rep, 2)
            self.t_rep = torch.unsqueeze(self.t_rep, 2)
            user_rep = torch.cat((self.v_rep[:self.num_user], self.a_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
            user_rep = self.weight_u.transpose(1,2)*user_rep

            # user_rep = torch.cat((self.v_rep[:self.num_user], self.a_rep[:self.num_user], self.t_rep[:self.num_user]), dim=1)
            user_rep = torch.cat((user_rep[:,:,0], user_rep[:,:,1], user_rep[:,:,2]), dim=1)
            user_rep = self.MLP_user(user_rep)
        item_rep = representation[self.num_user:]
        ############################################ multi-modal information aggregation
        h_u1 = self.user_graph(user_rep,user_graph,user_weight_matrix)
        user_rep = user_rep + h_u1 
        self.result_embed = torch.cat((user_rep, item_rep), dim=0)
        user_tensor = self.result_embed[user_nodes]
        pos_item_tensor = self.result_embed[pos_item_nodes]
        neg_item_tensor = self.result_embed[neg_item_nodes]
        pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)
        return pos_scores, neg_scores
    def loss_constra(self,user_rep,user_graph,user_weight_matrix,user_cons):
        loss_constra =0 
        # pdb.set_trace()
        neg_scores = torch.exp((user_rep[user_cons]*user_rep[self.user_index]).sum(dim=2)).sum(dim=1)
        pos_scores = torch.exp((user_rep[user_graph]*user_rep[self.user_index]).sum(dim=2)).sum(dim=1)
        loss_constra = -torch.log2(pos_scores/(pos_scores+neg_scores)).mean()
        
        # for i 
        return loss_constra
    def loss(self, data , user_graph,user_weight_matrix,user_cons=None):
        user, pos_items, neg_items = data
        pos_scores, neg_scores = self.forward(user.to(self.device), pos_items.to(self.device), neg_items.to(self.device) ,user_graph,user_weight_matrix.to(self.device),user_cons)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        reg_embedding_loss_v = (self.v_preference[user.to(self.device)] ** 2).mean()
        reg_embedding_loss_a = (self.a_preference[user.to(self.device)] ** 2).mean()
        reg_embedding_loss_t = (self.t_preference[user.to(self.device)] ** 2).mean()
        
        reg_loss = self.reg_weight * (reg_embedding_loss_v+reg_embedding_loss_a+reg_embedding_loss_t)
        if self.construction == 'weighted_sum':
            reg_loss+=self.reg_weight*(self.weight_u ** 2).mean()
            reg_loss+=self.reg_weight*(self.weight_i ** 2).mean()
        elif self.construction == 'cat_mlp':
            reg_loss+=self.reg_weight*(self.MLP_user.weight ** 2).mean()
        return loss_value + reg_loss, reg_loss
    def gene_ranklist(self,val_data,test_data,step=20000, topk=10):
        user_tensor = self.result_embed[:self.num_user].cpu()
        item_tensor = self.result_embed[self.num_user:self.num_user+self.num_item].cpu()
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
    def gene_ranklist_cold(self,val_data,test_data,step=20000, topk=10):
        user_tensor = self.result_embed[:self.num_user].cpu()
        item_tensor = self.result_embed[self.num_user:].cpu()
        item_tensor_cold = item_tensor[self.num_item:]#把冷启动商品找出来
        
        start_index = 0
        end_index = self.num_user if step == None else step

        all_index_of_rank_list_tra = torch.LongTensor([])
        all_index_of_rank_list_vt = torch.LongTensor([])
        all_index_of_rank_list_tt = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            # pdb.set_trace()
            score_matrix_tra = torch.matmul(temp_user_tensor, item_tensor_cold.t())
            score_matrix_vt =  score_matrix_tra.clone().detach()
            score_matrix_tt =  score_matrix_tra.clone().detach()

            _, index_of_rank_list_tra = torch.topk(score_matrix_tra, topk)
            all_index_of_rank_list_tra = torch.cat((all_index_of_rank_list_tra, index_of_rank_list_tra.cpu() + self.num_user),
                                               dim=0)
            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col)) - (self.num_user+self.num_item)
                    if torch.gt(torch.LongTensor(list(col)),-1).all()==False:
                        continue
                    score_matrix_vt[row][col] = 1e-5
                    score_matrix_tt[row][col] = 1e-5
            for i in range(len(val_data)):
                if val_data[i][0] >= start_index and val_data[i][0] < end_index:
                    row = val_data[i][0] - start_index
                    col = torch.LongTensor(list(val_data[i][1:])) - (self.num_user+self.num_item)
                    if torch.gt(torch.LongTensor(list(col)),-1).all()==False:
                        continue
                    score_matrix_tt[row][col] = 1e-5
            for i in range(len(test_data)):
                if test_data[i][0] >= start_index and test_data[i][0] < end_index:
                    row = test_data[i][0] - start_index
                    col = torch.LongTensor(list(test_data[i][1:])) - (self.num_user+self.num_item)
                    if torch.gt(torch.LongTensor(list(col)),-1).all()==False:
                        continue
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

