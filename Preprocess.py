import numpy as np
from collections import defaultdict
from tqdm import tqdm
import torch
import pdb
def gen_dense_user_graph(all_edge, num_inter):
    edge_dict = defaultdict(set)

    user_set = set()
    for edge in all_edge:
        user, item = edge
        edge_dict[user].add(item)
        user_set.add(user)

    user_list = list(user_set)
    user_list.sort()
    min_user = user_list[0]
    num_user = user_list[-1]-user_list[0]+1
    print(num_user)
    print(min_user)
    edge_set = set()
    edge_list_i = []
    edge_list_j = []

    edge_adj = np.zeros((num_user, num_user), dtype=np.int)#np.eyes((num_item, num_item), dtype=np.int)#
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    node_set = set()
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            # print(head_key, rear_key)
            item_head = edge_dict[head_key]
            item_rear = edge_dict[rear_key]
            # print(len(user_head.intersection(user_rear)))
            if len(item_head.intersection(item_rear)) > num_inter:
                # edge_list_i.append(head_key-min_item)
                # edge_list_j.append(rear_key-min_item)
                # edge_list_i.append([head_key-min_item, rear_key-min_item])
                # if head_key != rear_key:
                #         edge_list_j.append(head_key-min_item)
                #         edge_list_i.append(rear_key-min_item)
                        # edge_list.append([rear_key-min_item, head_key-min_item])
                edge_set.add((head_key, rear_key))
                node_set.add(head_key)
                node_set.add(rear_key)
                edge_adj[head_key-min_user, rear_key-min_user] = 1
                edge_adj[rear_key-min_user, head_key-min_user] = 1
    # edge_list = [edge_list_i, edge_list_j]
    bar.close()

    return edge_adj, edge_set, node_set
def gen_dense_item_graph(all_edge, num_inter):
    edge_dict = defaultdict(set)

    item_set = set()
    for edge in all_edge:
        user, item = edge
        edge_dict[item].add(user)
        item_set.add(item)

    item_list = list(item_set)
    item_list.sort()
    min_item = item_list[0]
    num_item = item_list[-1]-item_list[0]+1
    print(num_item)
    print(min_item)
    edge_set = set()
    edge_list_i = []
    edge_list_j = []

    edge_adj = np.zeros((num_item, num_item), dtype=np.int)#np.eyes((num_item, num_item), dtype=np.int)#
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    node_set = set()
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            # print(head_key, rear_key)
            user_head = edge_dict[head_key]
            user_rear = edge_dict[rear_key]
            # print(len(user_head.intersection(user_rear)))
            if len(user_head.intersection(user_rear)) > num_inter:
                # edge_list_i.append(head_key-min_item)
                # edge_list_j.append(rear_key-min_item)
                # edge_list_i.append([head_key-min_item, rear_key-min_item])
                # if head_key != rear_key:
                #         edge_list_j.append(head_key-min_item)
                #         edge_list_i.append(rear_key-min_item)       
                        # edge_list.append([rear_key-min_item, head_key-min_item])
                edge_set.add((head_key, rear_key))
                node_set.add(head_key)
                node_set.add(rear_key)
                edge_adj[head_key-min_item, rear_key-min_item] = 1
                edge_adj[rear_key-min_item, head_key-min_item] = 1
    # edge_list = [edge_list_i, edge_list_j]
    bar.close()

    return edge_adj, edge_set, node_set
def gen_user_graph(all_edge):
    edge_dict = defaultdict(set)

    user_set = set()
    for edge in all_edge:
    	user, item = edge
    	edge_dict[user].add(item)
    	user_set.add(user)

    user_list = list(user_set)
    user_list.sort()
    min_user = user_list[0]
    num_user = user_list[-1]-user_list[0]+1
    print(num_user)
    print(min_user)
    edge_set = set()
    edge_list_i = []
    edge_list_j = []
    user_graph_matrix = torch.zeros(num_user,num_user)
    # edge_adj = np.zeros((num_item, num_item), dtype=np.int)#np.eyes((num_item, num_item), dtype=np.int)#
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    node_set = set()
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            # print(head_key, rear_key)
            item_head = edge_dict[head_key]
            item_rear = edge_dict[rear_key]
            # print(len(user_head.intersection(user_rear)))
            if len(item_head.intersection(item_rear)) > 0:
                edge_list_i.append(head_key-min_user)
                edge_list_j.append(rear_key-min_user)
                if head_key != rear_key:
                    user_graph_matrix[head_key-min_user][rear_key-min_user] = len(item_head.intersection(item_rear))
                # edge_list_i.append([head_key-min_item, rear_key-min_item])
                if head_key != rear_key:
                        edge_list_j.append(head_key-min_user)
                        edge_list_i.append(rear_key-min_user)
                        # edge_list.append([rear_key-min_item, head_key-min_item])
                edge_set.add((head_key, rear_key))
                node_set.add(head_key)
                node_set.add(rear_key)
                # edge_adj[head_key-min_item, rear_key-min_item] = 1
                # edge_adj[rear_key-min_item, head_key-min_item] = 1
    edge_list = [edge_list_i, edge_list_j]
    bar.close()

    return edge_list, edge_set, node_set,user_graph_matrix
def gen_item_graph(all_edge):
    edge_dict = defaultdict(set)

    item_set = set()
    for edge in all_edge:
    	user, item = edge
    	edge_dict[item].add(user)
    	item_set.add(item)

    item_list = list(item_set)
    item_list.sort()
    min_item = item_list[0]
    num_item = item_list[-1]-item_list[0]+1
    print(num_item)
    print(min_item)
    edge_set = set()
    edge_list_i = []
    edge_list_j = []

    # edge_adj = np.zeros((num_item, num_item), dtype=np.int)#np.eyes((num_item, num_item), dtype=np.int)#
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    node_set = set()
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            # print(head_key, rear_key)
            user_head = edge_dict[head_key]
            user_rear = edge_dict[rear_key]
            # print(len(user_head.intersection(user_rear)))
            if len(user_head.intersection(user_rear)) > 0:
                edge_list_i.append(head_key-min_item)
                edge_list_j.append(rear_key-min_item)
                # edge_list_i.append([head_key-min_item, rear_key-min_item])
                if head_key != rear_key:
                        edge_list_j.append(head_key-min_item)
                        edge_list_i.append(rear_key-min_item)       
                        # edge_list.append([rear_key-min_item, head_key-min_item])
                edge_set.add((head_key, rear_key))
                node_set.add(head_key)
                node_set.add(rear_key)
                # edge_adj[head_key-min_item, rear_key-min_item] = 1
                # edge_adj[rear_key-min_item, head_key-min_item] = 1
    edge_list = [edge_list_i, edge_list_j]
    bar.close()

    return edge_list, edge_set, node_set
def gen_user_matrix(all_edge):
    edge_dict = defaultdict(set)

    user_set = set()
    for edge in all_edge:
    	user, item = edge
    	edge_dict[user].add(item)
    	user_set.add(user)

    user_list = list(user_set)
    user_list.sort()
    min_user = user_list[0]
    num_user = user_list[-1]-user_list[0]+1
    print(num_user)
    print(min_user)
    edge_set = set()
    edge_list_i = []
    edge_list_j = []
    user_graph_matrix = torch.zeros(num_user,num_user)
    # edge_adj = np.zeros((num_item, num_item), dtype=np.int)#np.eyes((num_item, num_item), dtype=np.int)#
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    node_set = set()
    # pdb.set_trace()
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]
            # print(head_key, rear_key)
            item_head = edge_dict[head_key]
            item_rear = edge_dict[rear_key]
            # print(len(user_head.intersection(user_rear)))
            if len(item_head.intersection(item_rear)) > 0:
                if head_key != rear_key:
                    user_graph_matrix[head_key-min_user][rear_key-min_user] = len(item_head.intersection(item_rear))
                    user_graph_matrix[rear_key - min_user][head_key - min_user] = len(item_head.intersection(item_rear))
    bar.close()

    return user_graph_matrix
def gen_item_matrix(all_edge):
    edge_dict = defaultdict(set)

    item_set = set()
    for edge in all_edge:
    	user, item = edge
    	edge_dict[item].add(user)
    	item_set.add(item)

    item_list = list(item_set)
    item_list.sort()
    min_item = item_list[0]
    num_item = item_list[-1]-item_list[0]+1
    print(num_item)
    print(min_item)
    edge_set = set()
    edge_list_i = []
    edge_list_j = []
    item_graph_matrix = torch.zeros(num_item, num_item)
    # edge_adj = np.zeros((num_item, num_item), dtype=np.int)#np.eyes((num_item, num_item), dtype=np.int)#
    key_list = list(edge_dict.keys())
    key_list.sort()
    bar = tqdm(total=len(key_list))
    node_set = set()
    for head in range(len(key_list)):
        bar.update(1)
        for rear in range(head, len(key_list)):
            head_key = key_list[head]
            rear_key = key_list[rear]

            user_head = edge_dict[head_key]
            user_rear = edge_dict[rear_key]

            if len(user_head.intersection(user_rear)) > 0:
                if head_key != rear_key:
                    item_graph_matrix[head_key - min_item][rear_key - min_item] = len(user_head.intersection(user_rear))
                    item_graph_matrix[rear_key - min_item][head_key - min_item] = len(user_head.intersection(user_rear))
    bar.close()

    return item_graph_matrix
# def data_load(dir_str):
#     train_edge = np.load(dir_str+'/train.npy', allow_pickle=True)
#     val_edge = np.load(dir_str+'/val.npy', allow_pickle=True)#.item()
#     test_edge = np.load(dir_str+'/test.npy', allow_pickle=True)#.item()
#     item_adj = np.load(dir_str+'edge_adj.npy', allow_pickle=True)

#     user_set = set()
#     item_set = set()
#     for edge in train_edge:
#     	user, item = edge
#     	user_set.add(user)
#     	item_set.add(item)

#     for edge in val_edge:
#     	user = edge[0]
#     	items = edge[1:]
#     	user_set.add(user)
#     	item_set = item_set.union(set(items))

#     for edge in test_edge:
#     	user = edge[0]
#     	items = edge[1:]
#     	user_set.add(user)
#     	item_set = item_set.union(set(items))

#     user_list = list(user_set)
#     item_list = list(item_set)
#     user_list.sort()
#     item_list.sort()

#     print(len(user_list), len(item_list))
#     print(user_list[0], user_list[-1])
#     print(item_list[0], item_list[-1])

def gen_user_dict(dir_str):
    user_item_dict = defaultdict(set)
    train_edge = np.load(dir_str+'/train.npy')
    for edge in train_edge:
    	user, item = edge
    	user_item_dict[user].add(item)
    np.save(dir_str+'/user_item_dict.npy', np.array(user_item_dict))

def full_data_gen(dir_str):
    val_edge = np.load(dir_str+'/val.npy', allow_pickle=True)
    test_edge = np.load(dir_str+'/test.npy', allow_pickle=True)
    val_full_list = list()
    test_full_list = list()

    for edge in val_edge:
    	if len(edge) < 1002:
    		continue
    	user = edge[0]
    	items = edge[1001:]
    	val_full_list.append([user]+items)

    for edge in test_edge:
    	if len(edge) < 1002:
    		continue
    	user = edge[0]
    	items = edge[1001:]
    	test_full_list.append([user]+items)
    np.save(dir_str+'/val_full.npy', np.array(val_full_list))
    np.save(dir_str+'/test_full.npy', np.array(test_full_list))

if __name__ == 	'__main__':    
    train_data = np.load('./train.npy')
    # item_item_pairs =[]
    user_graph_matrix = gen_user_matrix(train_data)
    #####################################################################generate user-user matrix
    # pdb.set_trace()
    user_graph = user_graph_matrix
    num_user = 32309
    num_item = 66456
    # user_num = torch.zeros(num_user)
    user_num = torch.zeros(num_user)

    user_graph_dict = {}
    item_graph_dict = {}
    edge_list_i = []
    edge_list_j = []

    for i in range(num_user):
        user_num[i] = len(torch.nonzero(user_graph[i]))
        print("this is ",i,"num",user_num[i])
    # for i in range(num_item):
    #     item_num[i] = len(torch.nonzero(item_graph[i]))
    #     print("this is ",i,"num",item_num[i])

    for i in range(num_user):
        print ("this is ",i,"user")
        if user_num[i] <= 200:
            user_i = torch.topk(user_graph[i],int(user_num[i]))
            edge_list_i =user_i.indices.numpy().tolist()
            edge_list_j =user_i.values.numpy().tolist()
            edge_list = [edge_list_i, edge_list_j]
            user_graph_dict[i] = edge_list
        else:
            user_i = torch.topk(user_graph[i], 200)
            edge_list_i = user_i.indices.numpy().tolist()
            edge_list_j = user_i.values.numpy().tolist()
            edge_list = [edge_list_i, edge_list_j]
            user_graph_dict[i] = edge_list
    # pdb.set_trace()
    np.save('./user_graph_dict_tik.npy',user_graph_dict,allow_pickle=True)
    # i_edge_adj, i_edge_set, i_node_set = gen_item_graph(train_data)
    # np.save('./Data/u_edge_adj.npy', u_edge_adj)
    # np.save('./Data/u_edge_adj.npy', i_edge_adj)
    # torch.save(item_matrix,'./Data/i_weight.pt')
    # np.save('./Data/Kwai/edge_list.npy', edge_adj)
    # print("u_edge",len(u_edge_set))
    # print("u_node",len(u_node_set))
    # print("i_edge", len(i_edge_set))
    # print("i_node", len(i_node_set))
    # data_load('./Data/movielens/')
    # gen_user_dict('./Data/Kwai')    
    # full_data_gen('./Data/Kwai')