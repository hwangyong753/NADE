import os
import numpy as np
import torch
import csv

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'LA':
        data = np.load('../data/X_LA.npy')    
    elif dataset == 'SD':
        data = np.load('../data/X_SD.npy')
    elif dataset == 'NOAA':
        data = np.load('../data/X_NOAA.npy')
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data

def get_adjacency_matrix(args):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    '''


    if args.dataset == 'LA' or args.dataset == 'SD' or args.dataset == 'NOAA':
        edge_idx = np.load('../data/edge_index_{}.npy'.format(args.dataset))  

        edge_idx = torch.tensor(edge_idx, dtype=torch.int64)
        weight = torch.ones(edge_idx.shape[1])
        adj = torch.sparse.FloatTensor(edge_idx, weight).to_dense().to(args.device)

        return adj


    else:
        raise ValueError