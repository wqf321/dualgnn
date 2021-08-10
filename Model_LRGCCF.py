# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys
import pdb
import argparse
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
 
import torch.nn.functional as F
import torch.autograd as autograd 

import pdb
from collections import defaultdict
import time


# os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [0]))
# dataset_base_path='../data/movielens'

# parser = argparse.ArgumentParser()
# parser.add_argument('--l_r', type=float, default=0.01, help='Learning rate.')
# parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay.')
# parser.add_argument('--dropout', type=float, default=0 ,help='dropout')
# args = parser.parse_args()

# ##movielens
# user_num=55485
# item_num=5986
# factor_num=64
# batch_size=2048*128
# num_negative_test_val=-1##all 
  
test_recall_list = []
test_ndcg_list = []

class LRGCCF(nn.Module):
    def __init__(self, user_num, item_num, factor_num,user_item_matrix,item_user_matrix,d_i_train,d_j_train,user_item_dict,weight_decay,device=None):
        super(LRGCCF, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """     
        self.user_item_dict = user_item_dict
        self.device = device
        self.num_user = user_num
        self.num_item = item_num
        self.weight_decay = weight_decay
        self.user_item_matrix = user_item_matrix.to(self.device)
        self.item_user_matrix = item_user_matrix.to(self.device)
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num) 

        for i in range(len(d_i_train)):
            d_i_train[i]=[d_i_train[i]]
        for i in range(len(d_j_train)):
            d_j_train[i]=[d_j_train[i]]

        self.d_i_train=torch.FloatTensor(d_i_train).to(self.device)
        self.d_j_train=torch.FloatTensor(d_j_train).to(self.device)
        self.d_i_train=self.d_i_train.expand(-1,factor_num)
        self.d_j_train=self.d_j_train.expand(-1,factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)  

    def loss(self, data):    
        user, item_i, item_j = data
        user = user.to(self.device)
        item_i = item_i.to(self.device)
        item_j = item_j.to(self.device)
        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight  
        # pdb.set_trace()
        gcn1_users_embedding = (torch.sparse.mm(self.user_item_matrix, items_embedding) + users_embedding.mul(self.d_i_train))#*2. #+ users_embedding
        gcn1_items_embedding = (torch.sparse.mm(self.item_user_matrix, users_embedding) + items_embedding.mul(self.d_j_train))#*2. #+ items_embedding
   
        gcn2_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) + gcn1_users_embedding.mul(self.d_i_train))#*2. + users_embedding
        gcn2_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) + gcn1_items_embedding.mul(self.d_j_train))#*2. + items_embedding
          
        gcn3_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) + gcn2_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        gcn3_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) + gcn2_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding
        
        # gcn4_users_embedding = (torch.sparse.mm(self.user_item_matrix, gcn3_items_embedding) + gcn3_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        # gcn4_items_embedding = (torch.sparse.mm(self.item_user_matrix, gcn3_users_embedding) + gcn3_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding
        
        gcn_users_embedding= torch.cat((users_embedding,gcn1_users_embedding,gcn2_users_embedding,gcn3_users_embedding),-1)#+gcn4_users_embedding
        gcn_items_embedding= torch.cat((items_embedding,gcn1_items_embedding,gcn2_items_embedding,gcn3_items_embedding),-1)#+gcn4_items_embedding#
      
        self.user_result = gcn_users_embedding
        self.item_result = gcn_items_embedding

        user_emb = F.embedding(user,gcn_users_embedding)
        item_i_emb = F.embedding(item_i-self.num_user,gcn_items_embedding)
        item_j_emb = F.embedding(item_j-self.num_user,gcn_items_embedding)  
        # # pdb.set_trace() 
        prediction_i = (user_emb * item_i_emb).sum(dim=-1)
        prediction_j = (user_emb * item_j_emb).sum(dim=-1) 
        # loss=-((rediction_i-prediction_j).sigmoid())**2#self.loss(prediction_i,prediction_j)#.sum()
        l2_regulization = self.weight_decay*(user_emb**2+item_i_emb**2+item_j_emb**2).sum(dim=-1)
        # l2_regulization = 0.01*((gcn1_users_embedding**2).sum(dim=-1).mean()+(gcn1_items_embedding**2).sum(dim=-1).mean())
      
        # loss2= -((prediction_i - prediction_j).sigmoid().log().mean())
        # loss= loss2 + l2_regulization
        loss= -((prediction_i - prediction_j)).sigmoid().log().mean() +l2_regulization.mean()
        # pdb.set_trace()
        return loss,l2_regulization.mean()

    def gene_ranklist(self,val_data,test_data,step=20000, topk=10):
        user_tensor = self.user_result.cpu()
        item_tensor = self.item_result.cpu()
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


    

 

 

 


