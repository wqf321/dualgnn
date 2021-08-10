import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pdb
class MyDataset(Dataset):
	def __init__(self, path, num_user, num_item, dataset,model_name):
		super(MyDataset, self).__init__()
		
		if dataset == 'Movielens':
			self.data = np.load(path+'train.npy')
			self.adj_lists = np.load(path + 'user_item_dict_mov.npy', allow_pickle=True).item()
		elif dataset =='Tiktok':
			self.data = np.load(path+'train_tik.npy')
			self.adj_lists = np.load(path+'user_item_dict_tik.npy',allow_pickle=True).item()
		# self.user_item_dict = np.load(path+'user_item_dict.npy', allow_pickle=True).item()
		# pdb.set_trace()
		self.num_user = num_user
		self.num_item = num_item
		self.model_name = model_name
		self.all_set = set(range(num_user, num_user+num_item))

	def __getitem__(self, index):
		user, pos_item = self.data[index]
		while True:
			neg_item = np.random.randint(self.num_user, self.num_user + self.num_item)
			# pdb.set_trace()
			if neg_item not in self.adj_lists[user]:
				break
		if self.model_name == 'VBPR' or self.model_name == 'MMGCN':
			return [torch.LongTensor([user,user]),torch.LongTensor([pos_item,neg_item])]
		else:
			return [int(user), int(pos_item), int(neg_item)]

	def __len__(self):
		return len(self.data)



if __name__ == '__main__':
	num_item = 76085
	num_user = 36656
	dataset = MyDataset('../../tiktok/', num_user, num_item)
	dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

	for data in dataloader:
		user, pos_items, neg_items= data
		print(yu_target.shape)


