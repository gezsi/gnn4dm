import numpy as np
import networkx as nx

import torch
from torch_geometric.utils import from_networkx

import time
import os

from sklearn.preprocessing import normalize

from cdlib import NodeClustering

from read_string import *

def compute_centrality_measure_features_old( G : nx.Graph, dataset_name : str ):

    dataset_filename = "./data/" + dataset_name + "_centrality_measure.npy"

    if os.path.exists( dataset_filename ):
        with open(dataset_filename, 'rb') as f:
            embedding = np.load(f)
    else:
        node_index_dict = dict( zip( range(len(G.nodes)), list(G.nodes) ) )

        dc = nx.degree_centrality(G)
        ec = nx.eigenvector_centrality(G)
        cc = nx.closeness_centrality(G)
        bc = nx.betweenness_centrality(G)
        pr = nx.pagerank(G)

        structural_features = [dc, ec, cc, bc, pr]

        embedding = np.zeros( (G.number_of_nodes(), len(structural_features)) )
        for i, structural_feature in enumerate(structural_features):
            for key, value in structural_feature.items():
                embedding[node_index_dict[key],i] = value

        with open(dataset_filename, 'wb') as f:
            np.save(f, embedding)

        aligned_df = pd.DataFrame( embedding, 
                                   columns=['degree_centrality', 'eigenvector_centrality', 'closeness_centrality', 'betweenness_centrality', 'pagerank'], 
                                   index=list(G.nodes) )
        aligned_df.to_csv(f"./data/StringDB/centrality_measures.txt", sep='\t', index=True)

    assert embedding.shape[0] == G.number_of_nodes()

    return embedding

def compute_centrality_measure_features( G : nx.Graph, dataset_name : str ):

    dataset_filename = "./data/" + dataset_name + "_centrality_measure.npy"

    if os.path.exists( dataset_filename ):
        with open(dataset_filename, 'rb') as f:
            embedding = np.load(f)
    else:
        node_dict = dict( zip( list(G.nodes), range(len(G.nodes)) ) )

        dc = nx.degree_centrality(G)
        ec = nx.eigenvector_centrality(G)
        cc = nx.closeness_centrality(G)
        bc = nx.betweenness_centrality(G)
        pr = nx.pagerank(G)

        structural_features = [dc, ec, cc, bc, pr]

        embedding = np.zeros( (G.number_of_nodes(), len(structural_features)) )
        for i, structural_feature in enumerate(structural_features):
            for key, value in structural_feature.items():
                embedding[node_dict[key],i] = value

        with open(dataset_filename, 'wb') as f:
            np.save(f, embedding)

        aligned_df = pd.DataFrame( embedding, 
                                   columns=['degree_centrality', 'eigenvector_centrality', 'closeness_centrality', 'betweenness_centrality', 'pagerank'], 
                                   index=list(G.nodes) )
        aligned_df.to_csv(f"./data/StringDB/centrality_measures.txt", 
                          sep='\t', 
                          index=True, 
                          header=False,
                          float_format='%.6f')

    assert embedding.shape[0] == G.number_of_nodes()

    return embedding


def get_STRING_dataset( inputFeatures : list = None, outputFeatures : list = None ):
    
    # Record start time
    start_time = time.time()
    # Create network from STRING DB
    STRING = getSTRING()
    # Record end time
    end_time = time.time()
    # Calculate elapsed time
    elapsed_time = end_time - start_time
    print(f"Network created. Elapsed time: {elapsed_time:.6f} seconds.")

    node_dict = dict( zip( list(STRING.nodes), range(len(STRING.nodes)) ) )
    node_index_dict = dict( zip( range(len(STRING.nodes)), list(STRING.nodes) ) )

    G = nx.relabel_nodes( STRING, node_dict )

    from sklearn.preprocessing import StandardScaler
    
    features = []
    for method in inputFeatures:
        # Record start time
        start_time = time.time()

        if method == 'GTEx':
            features.append( getGTExFeatures( STRING=STRING ) )
        elif method == 'GWASAtlas_128':
            features.append( getGWASAtlasFeatures( STRING=STRING, filter = 'all', performPCA = True, num_components=128 ) )
        elif method == 'GWASAtlas_256':
            features.append( getGWASAtlasFeatures( STRING=STRING, filter = 'all', performPCA = True, num_components=256 ) )
        elif method == 'GWASAtlas_512':
            features.append( getGWASAtlasFeatures( STRING=STRING, filter = 'all', performPCA = True, num_components=512 ) )
        elif method == 'GWASAtlas_1024':
            features.append( getGWASAtlasFeatures( STRING=STRING, filter = 'all', performPCA = True, num_components=1024 ) )
        elif method == 'GWASAtlas_nonukb_128':
            features.append( getGWASAtlasFeatures( STRING=STRING, filter = 'non-ukb', performPCA = True, num_components=128 ) )
        elif method == 'GWASAtlas_nonukb_256':
            features.append( getGWASAtlasFeatures( STRING=STRING, filter = 'non-ukb', performPCA = True, num_components=256 ) )
        elif method == 'GWASAtlas_nonukb_512':
            features.append( getGWASAtlasFeatures( STRING=STRING, filter = 'non-ukb', performPCA = True, num_components=512 ) )
        elif method == 'GWASAtlas_nonukb_1024':
            features.append( getGWASAtlasFeatures( STRING=STRING, filter = 'non-ukb', performPCA = True, num_components=1024 ) )
        elif method == 'GWASAtlas_ukb':
            features.append( getGWASAtlasFeatures( STRING=STRING, filter = 'ukb', performPCA = False ) )
        elif method == 'centrality_measures':
            features.append( compute_centrality_measure_features( STRING, "STRING" ) )

        # Record end time
        end_time = time.time()

        # Calculate elapsed time
        elapsed_time = end_time - start_time

        print(f"Computed features [{method} {features[len(features)-1].shape}]. Elapsed time: {elapsed_time:.6f} seconds.")

    all_features = np.concatenate( features, axis = 1 )

    x = torch.from_numpy( StandardScaler().fit_transform( all_features ) ).float()

    # convert to pyg data format
    data = from_networkx( STRING )
    data.x = x
    
    # computing MSigDB outputs
    start_time = time.time()
    msigdb_ground_truth, msigdb_train_indices, msigdb_valid_indices, msigdb_term_ids = getMSigDBFeatures( STRING=STRING, dbs=outputFeatures )
    end_time = time.time()
    if len(msigdb_ground_truth) > 0:
        print(f"Computed output features [MSigDB]. Elapsed time: {(end_time-start_time):.6f} seconds.")
   
    data.ys = {**msigdb_ground_truth}
    data.train_indices = {**msigdb_train_indices}
    data.valid_indices = {**msigdb_valid_indices}

    return { 'data': data, 
             'graph': G, 
             'node_dict': node_dict, 
             'node_index_dict': node_index_dict,
             'msigdb_term_ids': msigdb_term_ids }


def show_dataset( dataset ):
    # name = "Dataset: " + dataset['graph'].name
    # print('=' * len(name))
    # print(name)
    # print('=' * len(name))
    # print()
    # print(dataset['data'])
    # print()

    # Gather some statistics about the graph.
    print(f"Some statistics about the graph:")
    print(f"-Number of nodes: {dataset['data'].num_nodes}")
    print(f"-Number of edges: {dataset['data'].num_edges}")
    print(f"-Number of node features: {dataset['data'].num_node_features}")
    if hasattr(dataset['data'], 'x_structural'):
        print(f"-Number of structural node features: {dataset['data'].x_structural.shape[1]}")
    print(f"-Average node degree: {dataset['data'].num_edges / dataset['data'].num_nodes:.2f}")
    print(f"-Has isolated nodes: {dataset['data'].has_isolated_nodes()}")
    print(f"-Has self-loops: {dataset['data'].has_self_loops()}")
    print(f"-Is undirected: {dataset['data'].is_undirected()}")
    if dataset['data'].is_undirected():
        print(f"-Is connected: {nx.is_connected(dataset['graph'])}")
        print(f"-Number of connected components: {nx.number_connected_components(dataset['graph'])}")

    if 'groundtruth_communities' in dataset:
        print()
        print('Ground-truth communities')
        print('========================')
        print(f"number of communities: {len(dataset['groundtruth_communities'].communities)}")
        print(f"sizes: {dataset['groundtruth_communities'].size(summary=False)}")
        print(f"average_internal_degree: {dataset['groundtruth_communities'].average_internal_degree().score:.4f}")
        print(f"conductance: {dataset['groundtruth_communities'].conductance().score:.4f}")
        print(f"internal_edge_density: {dataset['groundtruth_communities'].internal_edge_density().score:.4f}")
    
