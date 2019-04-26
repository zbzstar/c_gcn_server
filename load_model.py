# -*- coding:utf-8 -*-
import torch
import sys
sys.path.append('/workspace/curve_gcn_release/code')
from Models.GNN import poly_gnn

def get_model_device():
	device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
	# 加载模型
	model = poly_gnn.PolyGNN(state_dim=128,n_adj=4,cnn_feature_grids=[112, 56, 28, 28],coarse_to_fine_steps=3,get_point_annotation=False).to(device)
	model.reload('/workspace/curve_gcn_release/checkpoints/Spline_GCN_epoch8_step21000.pth',strict=False)
	model.eval()
	return model,device