from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import no_grad
import numpy as np
def test_eval(epoch, model, data, prefix, ranklist,args,result_log_path,cold_start,writer=None):
    
    print(prefix+' start...')
    model.eval()

    with no_grad():
        precision_10, recall_10, ndcg_score_10,precision_5, recall_5, ndcg_score_5,precision_1, recall_1, ndcg_score_1 = model.full_accuracy(data,ranklist,cold_start)
        print('---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
            epoch, precision_10, recall_10, ndcg_score_10))
        with open(result_log_path, "a") as f:
                f.write(
                    '---------------------------------'+prefix+': {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                        epoch, 10, precision_10, recall_10, ndcg_score_10))  # 将字符串写入文件中
                f.write("\n")
        with open(result_log_path, "a") as f:
                f.write(
                    '---------------------------------'+prefix+': {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                        epoch, 5, precision_5, recall_5, ndcg_score_5))  # 将字符串写入文件中
                f.write("\n")
        with open(result_log_path, "a") as f:
                f.write(
                    '---------------------------------'+prefix+': {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                        epoch, 1, precision_1, recall_1, ndcg_score_1))  # 将字符串写入文件中
                f.write("\n")
        # writer.add_scalar(prefix+'_Precition', precision, epoch)
        # writer.add_scalar(prefix+'_Recall', recall, epoch)
        # writer.add_scalar(prefix+'_NDCG', ndcg_score, epoch)

        # writer.add_histogram(prefix+'_visual_distribution', model.v_rep, epoch)
        # writer.add_histogram(prefix+'_acoustic_distribution', model.a_rep, epoch)
        # writer.add_histogram(prefix+'_textual_distribution', model.t_rep, epoch)

        return precision_10, recall_10, ndcg_score_10
def train_eval(epoch, model, prefix,ranklist,args,result_log_path,writer=None):
    print(prefix + ' start...')
    model.eval()

    with no_grad():
        precision_10, recall_10, ndcg_score_10,precision_5, recall_5, ndcg_score_5,precision_1, recall_1, ndcg_score_1 = model.accuracy(ranklist)
        print(
            '---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
                epoch, precision_10, recall_10, ndcg_score_10))
        with open(result_log_path, "a") as f:
                f.write(
                    '---------------------------------Tra: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                        epoch, 10, precision_10, recall_10, ndcg_score_10))  # 将字符串写入文件中
                f.write("\n")
        with open(result_log_path, "a") as f:
                f.write(
                    '---------------------------------Tra: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                        epoch, 5, precision_5, recall_5, ndcg_score_5))  # 将字符串写入文件中
                f.write("\n")
        with open(result_log_path, "a") as f:
                f.write(
                    '---------------------------------Tra: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                        epoch, 1, precision_1, recall_1, ndcg_score_1))  # 将字符串写入文件中
                f.write("\n")
        # return precision_10, recall_10, ndcg_score_10,precision_5, recall_5, ndcg_score_5,precision_1, recall_1, ndcg_score_1

