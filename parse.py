'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Seed init.')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--model_name', default='DualGNN', help='Model name.')
    parser.add_argument('--data_path', default='amazon-book', help='Dataset path')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--l_r', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay.')
    parser.add_argument('--dropnode', type=float, default=0.1 ,help='dropnode')
    parser.add_argument('--batch_size', type=int, default=8192, help='Batch size.')
    parser.add_argument('--dim_latent', type=int, default=64, help='Latent dimension.')
    parser.add_argument('--num_epoch', type=int, default=1000, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=1, help='Workers number.')
    parser.add_argument('--aggr_mode', default='add', help='Aggregation mode.')
    parser.add_argument('--user_aggr_mode', default='softmax', help='Aggregation mode.')
    parser.add_argument('--construction', default='weighted_sum', help='information construction')
    parser.add_argument('--num_layer', type=int, default=1, help='Layer number.')
    parser.add_argument('--has_id', type=bool, default=True, help='Has id_embedding')
    parser.add_argument('--device', type=int, default=0, help='cuda devices')
    parser.add_argument('--dataset', default='Tiktok', help='Dataset path')
    parser.add_argument('--varient', default='random', help='model varient')
    parser.add_argument('--sampling', type=int, default=40, help='user co-occurance number')
    parser.add_argument('--traces', type=int, default=2, help='number of traces')
    parser.add_argument('--backprob', type=float, default=0.5, help='backward probability')
    parser.add_argument('--blocks',type=int, default=0, help='matrix blocks')
    parser.add_argument('--l_r_1', type=float, default=0.01,help='lr and reg for id embeddings')
    parser.add_argument('--l_r_2', type=float, default=0.01,help='lr and reg for context features')
    parser.add_argument('--l_r_3', type=float, default=0.01,help='lr and reg for wei only')
    parser.add_argument('--wd1', type=float,default=0,help='lr and reg for id embeddings')
    parser.add_argument('--wd2', type=float,default=0.01,help='lr and reg for context features')
    parser.add_argument('--wd3', type=float,default=0.01,help='lr and reg for wei only')
    parser.add_argument('--cold_start', type=int,default=0,help='lr and reg for wei only')
    return parser.parse_args()
