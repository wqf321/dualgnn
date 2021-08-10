import torch
import pdb
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, degree,dropout_adj
from torch_geometric.nn.inits import uniform

class Base_gcn(MessagePassing):
	def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
		super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
		self.aggr = aggr
		self.in_channels = in_channels
		self.out_channels = out_channels

	def forward(self, x, edge_index, size=None):
		# pdb.set_trace()
		if size is None:
			edge_index, _ = remove_self_loops(edge_index)
			# edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = x.unsqueeze(-1) if x.dim() == 1 else x
		# pdb.set_trace()
		return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

	def message(self, x_j, edge_index, size):
		if self.aggr == 'add':
			# pdb.set_trace()
			row, col = edge_index
			deg = degree(row, size[0], dtype=x_j.dtype)
			deg_inv_sqrt = deg.pow(-0.5)
			norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
			return norm.view(-1, 1) * x_j
		return x_j

	def update(self, aggr_out):
		return aggr_out

	def __repr(self):
		return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)