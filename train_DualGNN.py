import os
import pdb
import argparse
import time
import numpy as np
import math
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DataLoad import MyDataset
from Model import DualGNN
from Model_LightGCN import LightGCN
from Model_LRGCCF import LRGCCF
from Model_MMGCN import MMGCN
from model_VBPR import VBPR_net
from Model_STARGCN import Stargcn
from evaluation import test_eval
from evaluation import train_eval
from parse import parse_args
import networkx as nx
import sys
import random
from collections import Counter
from matplotlib import cbook
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
def readD(set_matrix,num_):
    user_d=[] 
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d
def readTrainSparseMatrix(set_matrix,is_user,u_d,i_d):
    user_items_matrix_i=[]
    user_items_matrix_v=[] 
    if is_user:
        d_i=u_d
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in set_matrix:
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            user_items_matrix_v.append(d_i_j)
    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)
def readTrainSparseMatrix_all(u_i,i_u,u_d,i_d,user_num):
    all_matrix_i=[]
    all_matrix_v=[]

    d_i=u_d
    d_j=i_d
    for i in u_i:
        len_set = len(u_i[i])
        for j in u_i[i]:
            all_matrix_i.append([i, j+user_num])
            d_i_j = np.sqrt(d_i[i] * d_j[j])
            # 1/sqrt((d_i+1)(d_j+1))
            all_matrix_v.append(d_i_j)  # (1./len_set)

    d_i=i_d
    d_j=u_d
    for i in i_u:
        len_set = len(i_u[i])
        for j in i_u[i]:
            all_matrix_i.append([i+user_num, j])
            d_i_j = np.sqrt(d_i[i] * d_j[j])
            # 1/sqrt((d_i+1)(d_j+1))
            all_matrix_v.append(d_i_j)  # (1./len_set)

    all_matrix_i=torch.cuda.LongTensor(all_matrix_i)
    all_matrix_v=torch.cuda.FloatTensor(all_matrix_v)
    return torch.sparse.FloatTensor(all_matrix_i.t(), all_matrix_v)
class Net:
    def __init__(self, args,sparse_graph):
        ##########################################################################################################################################
        self.device = torch.device("cuda:{0}".format(args.device) if torch.cuda.is_available() else "cpu")
        ##########################################################################################################################################
        seed = args.seed
        
        # seed = np.random.randint(0,200000)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.seed = seed
        self.varient = args.varient
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.learning_rate = args.l_r#l_r#
        self.weight_decay = args.weight_decay#weight_decay#
        self.drop_rate = args.dropnode
        self.batch_size = args.batch_size
        self.construction = args.construction
        self.num_traces = args.traces
        self.back_pro = args.backprob
        self.num_workers = args.num_workers
        self.num_epoch = args.num_epoch
        self.K = args.sampling
        self.dataset = args.dataset
        self.cold_start = args.cold_start
        self.dim_latent = args.dim_latent
        self.aggr_mode = args.aggr_mode#aggr_mode#
        self.user_aggr_mode = args.user_aggr_mode
        self.num_layer = args.num_layer
        self.has_id = args.has_id
        self.test_recall = []
        self.test_ndcg = []
        ##########################################################################################################################################
        if self.dataset =='Movielens':
            self.num_user = 55485
            self.num_item = 5986
        elif self.dataset =='tiktok':
            self.num_user = 32309
            self.num_item = 66456
        print('Data loading ...')
        self.train_dataset = MyDataset('../Data/'+self.dataset+'/', self.num_user, self.num_item,self.dataset,self.model_name)
        path = '../Data/'+self.dataset+'/'
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        if args.dataset == 'Movielens':
            self.edge_index = np.load('../Data/Movielens/train.npy')
            self.user_graph_dict = np.load('../Data/Movielens/user_graph_dict.npy', allow_pickle=True).item()
            self.user_graph_dict_rw = np.load('../Data/Movielens/user_graph_dict_randomwalk.npy',allow_pickle=True).item()
            self.val_dataset = np.load('../Data/Movielens/val_full.npy', allow_pickle=True)
            self.test_dataset = np.load('../Data/Movielens/test_full.npy', allow_pickle=True)
            self.sparse_graph = sparse_graph
            self.v_feat = np.load('../Data/Movielens/FeatureVideo_normal.npy',allow_pickle=True)
            self.a_feat = np.load('../Data/Movielens/FeatureAudio_avg_normal.npy',allow_pickle=True)
            self.t_feat = np.load('../Data/Movielens/FeatureText_stl_normal.npy',allow_pickle=True)
            self.user_item_dict = np.load('../Data/Movielens/user_item_dict.npy', allow_pickle=True).item()
            # self.training_user_set = np.load('../Data/Movielens/user_item_dict_lr.npy', allow_pickle=True).item()
            # self.training_item_set = np.load('../Data/Movielens/item_user_dict_lr.npy', allow_pickle=True).item()
            # self.item_user_dict = np.load('./Data/Movielens/item_user_dict.npy', allow_pickle=True).item()
            # self.final_embed = torch.load('./Data/Movielens/movielens_final_emb.pt')
        elif args.dataset == 'Tiktok':
            self.edge_index = np.load('../Data/tiktok_new/train_tik.npy')
            # pdb.set_trace()
            self.sparse_graph = sparse_graph
            self.user_graph_dict = np.load('../Data/tiktok_new/user_graph_dict.npy',allow_pickle=True).item()
            self.val_dataset = np.load('../Data/tiktok_new/val_full_tik.npy',allow_pickle=True)
            self.test_dataset = np.load('../Data/tiktok_new/test_full_tik.npy',allow_pickle=True)
            self.v_feat = torch.load('../Data/tiktok_new/feat_v_tik.pt')
            self.a_feat = torch.load('../Data/tiktok_new/feat_a_tik.pt')
            self.t_feat = torch.load('../Data/tiktok_new/feat_t_tik.pt')
            self.user_item_dict = np.load('../Data/tiktok_new/user_item_dict_tik.npy', allow_pickle=True).item()
            self.training_user_set = np.load('../Data/tiktok_new/user_item_dict_lr.npy', allow_pickle=True).item()
            self.training_item_set = np.load('../Data/tiktok_new/item_user_dict_lr.npy', allow_pickle=True).item()
        elif args.dataset =='cold_movie':
            self.edge_index = np.load(path+'final_cold_movie_train.npy')
            self.user_graph_dict = np.load(path+'user_graph_dict_cold_movie.npy',allow_pickle=True).item()
            self.val_dataset_warm_cold = np.load(path+'val_movie_warm+cold.npy',allow_pickle=True)
            self.test_dataset_warm_cold = np.load(path+'test_movie_warm+cold.npy',allow_pickle=True)
            self.val_dataset_warm = np.load(path+'val_movie_warm.npy',allow_pickle=True)
            self.test_dataset_warm = np.load(path+'test_movie_warm.npy',allow_pickle=True)
            self.val_dataset_cold = np.load(path+'val_movie_cold.npy',allow_pickle=True)
            self.test_dataset_cold = np.load(path+'test_movie_cold.npy',allow_pickle=True)
            self.v_feat = np.load(path+'v_feat_movie_cold.npy')
            self.a_feat = np.load(path+'a_feat_movie_cold.npy')
            self.t_feat = np.load(path+'t_feat_movie_cold.npy')
            self.user_item_dict = np.load(path+'user_item_dict_cold_mov.npy', allow_pickle=True).item()
            self.item_user_dict = np.load(path+'item_user_dict_cold_mov.npy', allow_pickle=True).item()
        print('Data has been loaded.')
        if self.model_name == 'LRGCCF':
            u_d=readD(self.training_user_set,self.num_user)
            i_d=readD(self.training_item_set,self.num_item)
            d_i_train=u_d
            d_j_train=i_d
            sparse_u_i=readTrainSparseMatrix(self.training_user_set,True,u_d,i_d)
            sparse_i_u=readTrainSparseMatrix(self.training_item_set,False,u_d,i_d)

        self.features = [self.v_feat, self.a_feat, self.t_feat]
        if args.model_name =='DualGNN':
            self.model = DualGNN(self.features, self.edge_index,self.batch_size, self.num_user, self.num_item, self.aggr_mode, self.construction, self.num_layer, self.has_id, self.dim_latent,self.weight_decay,self.drop_rate,self.K, self.user_item_dict,self.dataset,self.cold_start,device = self.device).to(self.device)
        if self.model_name == 'LightGCN':
            self.model = LightGCN(self.features, self.sparse_graph,self.batch_size, self.num_user, self.num_item, self.aggr_mode, self.concat, self.num_layer, self.has_id, self.dim_latent,self.weight_decay,self.dropout , self.user_item_dict,device=self.device).to(self.device)
        if self.model_name == 'MMGCN':
            self.model = MMGCN(self.v_feat, self.a_feat, self.t_feat, self.t_feat, self.edge_index, self.batch_size, self.num_user, self.num_item, 'mean', 'False', 2, True, self.user_item_dict, self.weight_decay, self.dim_latent,self.dataset,self.drop_rate,device=self.device).to(self.device)
        if self.model_name == 'VBPR':
            self.model = VBPR_net(self.num_user, self.num_item,self.weight_decay, self.dim_latent, self.features,self.user_item_dict,self.dataset,device=self.device).to(self.device)
        if self.model_name == 'Stargcn':
            self.model = Stargcn(self.features,self.sparse_graph, self.batch_size, self.num_user, self.num_item, self.aggr_mode, self.concat,
                 self.num_layer, self.weight_decay, self.user_item_dict,self.dim_latent,device=self.device).to(self.device)
        if self.model_name == 'LRGCCF':
            self.model = LRGCCF(self.num_user, self.num_item, self.dim_latent,sparse_u_i,sparse_i_u,d_i_train,d_j_train,self.user_item_dict,self.weight_decay,device=self.device).to(self.device)

            
        if args.PATH_weight_load and os.path.exists(args.PATH_weight_load):
            self.model.load_state_dict(torch.load(args.PATH_weight_load))
            print('module weights loaded....')
        ##########################################################################################################################################
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.learning_rate}])
        ##########################################################################################################################################
    def run(self,):
        # self.model.result_embed = torch.load('./results/top40_best_movie/result_emb.pt')
        # pdb.set_trace()
        max_precision = 0.0
        max_recall = 0.0
        max_NDCG = 0.0
        num_decreases = 0
        max_val=0
        while os.path.exists('./results/'+self.varient) == False :
            os.mkdir('./results/'+self.varient)
        result_log_path = './results/'+self.varient+f'/train_log_({args.l_r:.6f}_{args.weight_decay:.6f}_{args.sampling:.6f}_{args.dataset:.9s}_{self.num_traces}_{self.back_pro}_{self.seed}).txt'
        result_path = './results/'+self.varient+'/result_{0}_{1}_{2}.txt'.format(args.dataset,self.num_traces,self.back_pro)
        result_best_path = './results/'+self.varient+'/result_best_{0}_{1}_{2}.txt'.format(args.dataset,self.num_traces,self.back_pro)
        user_cons = 0
        print(args.l_r)
        print(args.weight_decay)
        print(args.dataset)
        
        with open(result_log_path, "a") as f:
            f.write(result_log_path)
            f.write("\n")
        for epoch in range(self.num_epoch):

            self.model.train()
            user_graph, user_weight_matrix = self.topk_sample(self.K)
            print('Now, training start ...')
            pbar = tqdm(total=len(self.train_dataset))
            sum_loss = 0.0
            sum_reg_loss = 0.0

            
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                if self.model_name == 'DualGNN':
                    self.loss,reg_loss = self.model.loss(data ,user_graph,user_weight_matrix,user_cons=user_cons)
                else:
                    self.loss,reg_loss = self.model.loss(data)
                self.loss.backward()
                self.optimizer.step()
                pbar.update(self.batch_size)
                sum_loss += self.loss
                sum_reg_loss += reg_loss
            print('avg_loss:',sum_loss/self.batch_size)
            print('avg_reg_loss:', sum_reg_loss / self.batch_size)
            pbar.close()
            if torch.isnan(sum_loss/self.batch_size):
                with open(result_path,'a') as save_file:
                    save_file.write('lr: {0} \t Weight_decay:{1} \t sampling:{2} \t SEED:{3} is Nan'.format(args.l_r, args.weight_decay,args.sampling,self.seed))
                break

            ranklist_tra, ranklist_vt, ranklist_tt  = self.model.gene_ranklist(self.val_dataset,self.test_dataset)

            train_eval(epoch, self.model, 'Train',ranklist_tra,args,result_log_path)
            _, val_recall_10, _ =test_eval(epoch, self.model, self.val_dataset, 'Val',ranklist_vt,args,result_log_path,0)
            test_precision_10, test_recall_10, test_ndcg_score_10 = test_eval(epoch, self.model, self.test_dataset, 'Test',ranklist_tt,args,result_log_path,0)
            
            if self.model_name == 'DualGNN':
                if self.construction == 'weighted_sum':
                    attn_u = F.softmax(self.model.weight_u,dim=1)
                    attn_u = torch.squeeze(attn_u)


                    attn_u_max = torch.max(attn_u,0)
                    attn_u_max_num = torch.max(attn_u,0).indices[0]
                    attn_u_min = torch.min(attn_u,0)
                    with open(result_log_path, "a") as f:
                        f.write('---------------------------------attn_u_max: {0}-th epoch {1}-th user 0 visual:{2:.4f} acoustic:{3:.4f} text:{4:.4f}---------------------------------'.format(
                                epoch, 10, float(attn_u_max[0][0]),float(attn_u_max[0][1]),float(attn_u_max[0][2])))  # 将字符串写入文件中
                        f.write("\n")
                    with open(result_log_path, "a") as f:
                        f.write('---------------------------------attn_u_num: {0}-th epoch {1}-th user 0 visual:{2:.4f} acoustic:{3:.4f} text:{4:.4f}---------------------------------'.format(
                                epoch, 10, float(attn_u[attn_u_max_num][0]),float(attn_u[attn_u_max_num][1]),float(attn_u[attn_u_max_num][2])))  # 将字符串写入文件中
                        f.write("\n")
                    with open(result_log_path, "a") as f:
                        f.write('---------------------------------attn_u_min: {0}-th epoch {1}-th user 0 visual:{2:.4f} acoustic:{3:.4f} text:{4:.4f}---------------------------------'.format(
                                epoch, 10, float(attn_u_min[0][0]),float(attn_u_min[0][1]),float(attn_u_min[0][2])))  # 将字符串写入文件中
                        f.write("\n")
            self.test_recall.append(test_recall_10)
            self.test_ndcg.append(test_ndcg_score_10)
           # pdb.set_trace()
            if val_recall_10 > max_val:
               max_precision = test_precision_10
               max_recall = test_recall_10
               max_NDCG = test_ndcg_score_10
               max_val = val_recall_10
               num_decreases = 0
               best_embed = self.model.result_embed
            else:   
                if num_decreases >20 and self.model_name != 'Stargcn':
                    with open(result_path, 'a') as save_file:
                        save_file.write(
                            'lr: {0} \t Weight_decay:{1} \t sampling:{2}=====> Precision:{3} \t Recall:{4} \t NDCG:{5} \t SEED:{6}\r\n'.
                            format(args.l_r, args.weight_decay,args.sampling ,max_precision, max_recall, max_NDCG,self.seed))
                    # torch.save(best_attn,'./results/'+self.varient+'/u_prefer_weight.pt')
                    torch.save(best_embed,'./results/'+self.varient+'/result_emb.pt')
                    while os.path.exists(result_best_path) == False :
                        with open(result_best_path, 'a') as save_file:
                            save_file.write(
                                'Recall:{0}\r\n'.
                                format(max_recall)) 
                    # pdb.set_trace()
                    file = open(result_best_path)
                    maxs = file.readline()
                    maxvalue = float(maxs.strip('[Recall:\n]'))
                    break
                else:
                    num_decreases += 1
            if epoch>990:
                    with open(result_path, 'a') as save_file:
                        save_file.write(
                            'lr: {0} \t Weight_decay:{1} \t sampling:{2}=====> Precision:{3} \t Recall:{4} \t NDCG:{5} \t SEED:{6}\r\n'.
                            format(args.l_r, args.weight_decay,args.sampling ,max_precision, max_recall, max_NDCG,self.seed))
                    file = open(result_best_path)
                    maxvalue = float(maxs.strip('[Recall:\n]'))
                    if max_recall >= maxvalue:
                        np.save('./results/'+self.varient+'/recall.npy',self.test_recall)
                        np.save('./results/'+self.varient+'/ndcg.npy',self.test_ndcg)
                        torch.save(self.model.result_embed,'./results/'+self.varient+'/result_emb.pt')
                    break
        return max_recall, max_precision, max_NDCG
    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    # pdb.set_trace()
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)
                
                # user_weight_matrix[i] = torch.tensor(user_graph_weight) / sum(user_graph_weight) #weighted
                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0) #softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k)/k #mean
                # pdb.set_trace()
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            # user_weight_matrix[i] = torch.tensor(user_graph_weight) / sum(user_graph_weight) #weighted
            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0) #softmax
            if self.user_aggr_mode == 'mean':
                # pdb.set_trace()
                user_weight_matrix[i] = torch.ones(k)/k #mean
            # user_weight_list.append(user_weight)
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix
if __name__ == '__main__':
    if args.dataset == 'Tiktok':
        # sparse_graph = torch.load('../Data/tiktok_new/graph.pt')
        sparse_graph=None
    elif args.dataset == 'Movielens':
        # sparse_graph = torch.load('../Data/Movielens/graph.pt')
        sparse_graph=None
    egcn = Net(args,sparse_graph)
    egcn.run()


