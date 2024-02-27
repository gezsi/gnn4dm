import argparse

import torch
import torch_geometric
from torch_geometric.utils import from_networkx
import networkx as nx
import pandas as pd
import numpy as np
import scipy as sp

from sklearn.preprocessing import StandardScaler

from models import *
from data import *
from utils import *

from tqdm import tqdm

import statistics
import random as rd

from cdlib import evaluation

import warnings

# Custom function to handle warnings
def warn_handler(message, category, filename, lineno, file=None, line=None):
    print(f"Warning: {message}")
    print(f"Category: {category.__name__}")
    print(f"File: {filename}, Line: {lineno}")

def print_version_info():
    print( "Version info:" )
    print( f"-PyTorch version: {torch.__version__}" )
    print( f"-CUDA version: {torch.version.cuda}" )
    print( f"-PyG version: {torch_geometric.__version__}" )
    print( f"-NetworkX version: {nx.__version__}" )

def loadGraph( filename ):
    print( f"Read graph from file: {filename}" )
    # create networkx graph from the data frame
    G = nx.read_edgelist(filename, nodetype=str)

    # select the largest connected component
    STRGING = nx.subgraph(G, max(nx.connected_components(G), key=len))

    return G

def loadInputFeatures( G : nx.Graph, filenames : list = None ):
    
    merged_df = None
    for filename in filenames:
        print( f"Read input features from file: {filename}" )
        # Read the current file with the first column as the index
        df = pd.read_csv(filename, sep='\s', index_col=0, header=None, engine='python')
        # If merged_df is None, it's the first iteration, so assign df to merged_df
        if merged_df is None:
            merged_df = df
        else:
            # Merge the current df with the merged_df on the index
            merged_df = pd.merge(merged_df, df, left_index=True, right_index=True, how='outer')

    # Step 1: Get the list of nodes from the NetworkX graph "G"
    nodes_list = list(G.nodes())

    # Step 2: Calculate the column-wise mean of the numpy matrix
    columnwise_mean = np.mean(merged_df.to_numpy(), axis=0)

    # Step 3: Align the rows of the DataFrame with the nodes in the NetworkX graph "G"
    # Create a new DataFrame with the same columns as the original DataFrame and index as nodes_list
    aligned_df = pd.DataFrame(columns=merged_df.columns, index=nodes_list)

    n_missing = 0

    # Iterate through the nodes and copy the corresponding row from the original DataFrame
    for node in nodes_list:
        if node in merged_df.index:
            aligned_df.loc[node] = merged_df.loc[node]
        else:
            # If the node is not in "m," fill the row with column-wise mean values
            aligned_df.loc[node] = columnwise_mean
            n_missing += 1

    # Step 4: Convert the aligned DataFrame to a numpy matrix (if needed)
    feature_matrix = aligned_df.to_numpy()

    print( "Number of ids with missing information: " + str(n_missing) )
    print( "Input feature matrix shape: " + str(feature_matrix.shape) )
    
    x = torch.from_numpy( StandardScaler().fit_transform( feature_matrix ) ).float()

    return x

def import_gmt_file( G : nx.Graph, gmt_file_path : str, minSize = 10, maxSize = 500 ):
    """
    Imports a GMT file and returns a dictionary where keys are gene set names
    and values are lists of genes in that set.

    :param gmt_file_path: Path to the GMT file.
    :return: A dictionary with gene set names as keys and lists of genes as values.
    """
    gene_sets = {}
    setOfGraphNodes = set(G.nodes())
    with open(gmt_file_path, 'r') as file:
        print( f"Read pathway annotations from file: {gmt_file_path}" )
        for line in file:
            parts = line.strip().split('\t')
            # The first part is the gene set name, the second part is usually a description
            gene_set_name = parts[0]
            genes = set(parts[2:]).intersection(setOfGraphNodes)
            if len(genes) >= minSize and len(genes) <= maxSize:
                gene_sets[gene_set_name] = genes
    return gene_sets

def loadPathwayDatabases( G : nx.Graph, filenames : list = None, train_ratio = 0.8 ):

    list_nodes = list(G.nodes())

    # Create a dictionary to store tensors for each DB
    db_tensors = {}
    train_indices = {}
    valid_indices = {}
    term_ids = {}    

    for filename in filenames:
        db_name = get_filename_without_extension(filename)
        pathways = import_gmt_file( G, filename )

        # Randomly select pathways for training (and the others for validation)
        selected_pathways_for_training = set(rd.sample(list(pathways.keys()), k = round(len(pathways) * train_ratio)))

        unique_genes = set().union(*pathways.values())
        num_unique_genes = len(unique_genes)

        # Create an empty tensor with the correct number of columns and rows
        tensor_shape = (len(list_nodes), len(pathways))
        db_tensor = torch.zeros(*tensor_shape, dtype=torch.float32)
        train_mask = torch.zeros_like(db_tensor, dtype=torch.bool)
        valid_mask = torch.zeros_like(db_tensor, dtype=torch.bool)

        # find those node indices on which we have info in this db (i.e. those geneids that participate in at least one pathway in this db)
        # this will be the union of the training and validation indices
        known_node_indices_in_db = {list_nodes.index(id) for id in unique_genes}

        name_index = 0
        names = list()
        for pathway_name, pathway_genes in pathways.items():
            # Check if this pathway is selected for training
            if pathway_name in selected_pathways_for_training:
                # Get the row indices for Ids from the list_nodes
                positive_train_indices = {list_nodes.index(id) for id in pathway_genes}                
                # Set the corresponding values to 1 in the tensor
                db_tensor[list(positive_train_indices), name_index] = 1.0
                # Set all known indices to True in the training mask tensor
                train_mask[list(known_node_indices_in_db), name_index] = True
            else:
                # Randomly select pathways for training (and the others for validation)
                selected_genes_for_training = set(rd.sample(list(pathway_genes), k = round(len(pathway_genes) * train_ratio)))
                selected_genes_for_validation = pathway_genes - selected_genes_for_training

                # Get the row indices for Ids from the list_nodes
                positive_train_indices = {list_nodes.index(id) for id in selected_genes_for_training}
                positive_valid_indices = {list_nodes.index(id) for id in selected_genes_for_validation}

                potential_negative_indices = known_node_indices_in_db - positive_train_indices - positive_valid_indices

                num_elements_to_select = round(len(potential_negative_indices) * train_ratio)
                negative_train_indices = set(rd.sample(list(potential_negative_indices), k = num_elements_to_select))

                negative_valid_indices = potential_negative_indices - negative_train_indices

                # Set the corresponding values to 1 in the tensor
                db_tensor[list(positive_train_indices | positive_valid_indices), name_index] = 1.0

                train_mask[list(positive_train_indices | negative_train_indices), name_index] = True
                valid_mask[list(positive_valid_indices | negative_valid_indices), name_index] = True

                assert positive_train_indices.isdisjoint(positive_valid_indices), "Positive indices are not disjoint"
                assert negative_train_indices.isdisjoint(negative_valid_indices), "Negative indices are not disjoint"
                assert positive_train_indices.isdisjoint(negative_train_indices), "Train indices (pos vs neg) are not disjoint"
                assert positive_valid_indices.isdisjoint(negative_valid_indices), "Train indices (pos vs neg) are not disjoint"
                assert known_node_indices_in_db == ( positive_train_indices | negative_train_indices | positive_valid_indices | negative_valid_indices ), "Train + valid indices not equal to all potential indices"

            names.append( pathway_name )
            name_index += 1

        # Store the tensor in the dictionary
        db_tensors[db_name] = db_tensor
        train_indices[db_name] = train_mask
        valid_indices[db_name] = valid_mask
        term_ids[db_name] = names

    return db_tensors, train_indices, valid_indices, term_ids

def train( model, data, source_edge_index, pos_edge_index, neg_edge_index, lambda_bce_loss, lambda_l1_positives_loss, lambda_l2_positives_loss, optimizer ):
    model.train()
    optimizer.zero_grad()

    # compute node embeddings with the encoder       
    F, y_preds = model.encode( data.x, data.edge_index )
    # compute link strength with the decoder for the positive and the negative edges, respectively
    H_positive = model.decode( F, source_edge_index, pos_edge_index )
    H_negative = model.decode( F, source_edge_index, neg_edge_index )
    # compute loss
    ## Bernoulli-Poisson loss
    bp_loss = model.nll_BernoulliPoisson_loss( H_positive, H_negative )
    ## Binary cross-entropy losses for all datasets
    bce_losses = dict()
    for key in y_preds:
        if data.train_indices[key].dim() == 1:
            bce_losses[key] = model.bce_loss( y_preds[key][data.train_indices[key],:], data.ys[key][data.train_indices[key],:] )
        else:
            bce_losses[key] = model.bce_loss( y_preds[key][data.train_indices[key]], data.ys[key][data.train_indices[key]] )
    ## L1 and L2 losses
    l1_positives_loss, l2_positives_loss = model.output_models_l1_l2_losses()
    ## compute final loss
    loss = bp_loss
    for bce_loss in bce_losses.values():
        loss += lambda_bce_loss * bce_loss
    loss += lambda_l1_positives_loss * l1_positives_loss
    loss += lambda_l2_positives_loss * l2_positives_loss

    loss.backward(retain_graph=True)
    optimizer.step()

    return

@torch.no_grad()
def test(model, data, G : nx.Graph):
    model.eval()

    # compute node embeddings with the encoder       
    F, y_preds = model.encode( data.x, data.edge_index )

    internal_evaluation_scores = {}

    # compute BCE loss in validation sets of pathway DBs
    bce_losses = dict()
    for key in y_preds:
        if data.valid_indices[key].dim() == 1:
            bce_losses[key] = model.bce_loss( y_preds[key][data.valid_indices[key],:], data.ys[key][data.valid_indices[key],:] )
        else:
            bce_losses[key] = model.bce_loss( y_preds[key][data.valid_indices[key]], data.ys[key][data.valid_indices[key]] )
    # sum losses, and write them into the output
    valid_bce_loss = 0.0
    for key, bce_loss in bce_losses.items():
        internal_evaluation_scores[f"bce_loss_{key}"] = bce_loss.item()
        valid_bce_loss += bce_loss.item()
    internal_evaluation_scores["sum_bce_loss"] = valid_bce_loss

    # compute accuracy and F1 score in validation sets of pathway DBs
    accuracies, sensitivities, specificities, precisions, f1_scores, auprcs = calculateMetrics( y_preds, data.ys, data.valid_indices )
    for key in accuracies:
        internal_evaluation_scores[f"accuracy_{key}"] = accuracies[key]
        internal_evaluation_scores[f"sensitivity_{key}"] = sensitivities[key]
        internal_evaluation_scores[f"specificity_{key}"] = specificities[key]
        internal_evaluation_scores[f"precision_{key}"] = precisions[key]
        internal_evaluation_scores[f"f1_score_{key}"] = f1_scores[key]
        internal_evaluation_scores[f"auprc_{key}"] = auprcs[key]
    internal_evaluation_scores["mean_accuracy"] = statistics.mean(accuracies.values())
    internal_evaluation_scores["mean_sensitivity"] = statistics.mean(sensitivities.values())
    internal_evaluation_scores["mean_specificity"] = statistics.mean(specificities.values())
    internal_evaluation_scores["mean_precision"] = statistics.mean(precisions.values())
    internal_evaluation_scores["mean_f1_score"] = statistics.mean(f1_scores.values())
    internal_evaluation_scores["mean_auprc"] = statistics.mean(auprcs.values())

    cosine_self_similarity_loss = model.cosine_similarity_loss( F, type = 'l2', min_threshold = 0.1 )
    internal_evaluation_scores["cosine_self_similarity_loss"] = cosine_self_similarity_loss.item()

    internal_evaluation_scores["thresholdOfCommunities"] = model.getThresholdOfCommunities().item()

    # get predicted communities
    pred_communities, original_communities = getPredictecCommunities3( F, G, threshold = model.getThresholdOfCommunities(), normalize=False )

    # and evaluate them
    if pred_communities.communities != []:
        internal_evaluation_scores['count'] = len(pred_communities.communities)
        internal_evaluation_scores['average_size'] = pred_communities.size().score
        internal_evaluation_scores['max_size'] = np.max(pred_communities.size(summary=False))
        internal_evaluation_scores['num_smaller_5'] = np.sum(np.array(pred_communities.size(summary=False)) <= 5)
        internal_evaluation_scores['num_smaller_10'] = np.sum(np.array(pred_communities.size(summary=False)) <= 10)
        internal_evaluation_scores['num_smaller_50'] = np.sum(np.array(pred_communities.size(summary=False)) <= 50)
        internal_evaluation_scores['num_smaller_100'] = np.sum(np.array(pred_communities.size(summary=False)) <= 100)
        internal_evaluation_scores['num_smaller_200'] = np.sum(np.array(pred_communities.size(summary=False)) <= 200)
        internal_evaluation_scores['node_coverage'] = pred_communities.node_coverage
        internal_evaluation_scores['mean_module_per_node'] = (F >= model.getThresholdOfCommunities()).sum(dim=1).float().mean().item()
        internal_evaluation_scores['average_internal_degree'] = pred_communities.average_internal_degree().score
        internal_evaluation_scores['conductance'] = pred_communities.conductance().score
        internal_evaluation_scores['internal_edge_density'] = pred_communities.internal_edge_density().score
        internal_evaluation_scores['fraction_over_median_degree'] = pred_communities.fraction_over_median_degree().score # it drops warning
        # internal_evaluation_scores['avg_embeddedness'] = pred_communities.avg_embeddedness().score # it takes ages to compute
        # internal_evaluation_scores['modularity_overlap'] = pred_communities.modularity_overlap().score # it takes ages to compute

        if 'groundtruth_communities' in dataset:
            internal_evaluation_scores['onmi'] = evaluation.overlapping_normalized_mutual_information_MGH( dataset['groundtruth_communities'], pred_communities ).score
                
    return internal_evaluation_scores, pred_communities, original_communities

def getDevice():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print( f"Using device: {str(device)}" )
    device = torch.device(device)
    return device

def runModel( dataset, args ):

    show_dataset( dataset )

    data = dataset['data']
    G = dataset['graph']
    
    # Get available device
    device = getDevice()
    # Send data to device
    data = data.to(device)

    epochs = 5000
    eval_steps = 50

    output_models = {}
    for db, ids in dataset['term_ids'].items():
        output_models[db] = PositiveLinear( in_features = args.module_count, 
                                            out_features = len(ids), 
                                            ids = ids ).to(device)

    model = GAEL( encoder = GNN( in_channels = data.num_features, 
                                    hidden_channels_before_module_representation = [args.hidden_channels_before_module_representation], 
                                    module_representation_channels = args.module_count, 
                                    out_models = output_models, 
                                    dropout = args.dropout, 
                                    batchnorm = args.batchnorm, 
                                    transform_probability_method = 'tanh',
                                    threshold = args.threshold,
                                    type = args.model_type ),
                    decoder = InnerProductDecoder() ).to(device)

    # print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total number of model parameters:", total_params)
        
    optimizer, scheduler = model.configure_optimizers( args )
    model.reset_parameters()

    print( "Started training..." )
    final_results = list()
    # start training
    for epoch in tqdm(range(1, 1 + epochs)):

        source_edge_index, pos_edge_index, neg_edge_index = torch_geometric.utils.structured_negative_sampling( edge_index = data.edge_index )

        # Generate a random permutation index
        num_samples = len(source_edge_index)
        perm = torch.randperm(num_samples)

        # Apply the permutation index to all three lists
        source_edge_index = source_edge_index[perm]
        pos_edge_index = pos_edge_index[perm]
        neg_edge_index = neg_edge_index[perm]

        # Divide each shuffled list into ten (approximately) equal parts
        num_parts = 10
        part_size = num_samples // num_parts

        for i in range(num_parts):
            start_idx = i * part_size
            end_idx = (i + 1) * part_size

            # Adjust the end index for the last part
            if i == num_parts - 1:
                end_idx = num_samples

            source_part = source_edge_index[start_idx:end_idx]
            pos_part = pos_edge_index[start_idx:end_idx]
            neg_part = neg_edge_index[start_idx:end_idx]

            # training step
            train( model, data, source_part, pos_part, neg_part, 
                lambda_bce_loss = args.lambda_bce_loss, 
                lambda_l1_positives_loss = args.lambda_l1_positives_loss, lambda_l2_positives_loss = args.lambda_l2_positives_loss, 
                optimizer = optimizer )

        # validation
        if epoch % eval_steps == 0:
            valid_results, pred_communities, original_communities = test( model, data, G )

            results = {'epoch': epoch, **valid_results}

            # print( "\n", results )

            # export results
            filename = f"{args.output_dir}/metrics_{epoch:05d}.json"
            saveResultsToJSON2( valid_results, filename )

            filename = f"{args.output_dir}/modules_{epoch:05d}.gmt"
            module_names = exportCommunitiesToGMT( original_communities, filename, dataset['node_index_dict'] )

            # save model
            # modelfilename = f"{args.output_dir}/model_{epoch:05d}.pt"
            # torch.save(model.state_dict(), modelfilename)

            # export weights of final model layers
            ws, ids = [], []
            for name in model.encoder.output_models:
                ws.append( model.encoder.output_models[name].weight.cpu().detach().numpy().T )
                ids.extend( model.encoder.output_models[name].id_list )

            weightfilename = f"{args.output_dir}/weights_{epoch:05d}.csv"
            df_w = pd.DataFrame( data = np.hstack(ws), columns=ids, index=module_names )
            df_w.to_csv( weightfilename )

            final_results.append( { 'model': str(model), 'model-name': model.name, **vars(args), 'epoch': epoch, **valid_results } )

            df = pd.DataFrame( final_results )
            df.to_csv( f"{args.output_dir}/results.csv" )
            
        # apply learning rate scheduler
        scheduler.step()

    df = pd.DataFrame( final_results )
    df.to_csv( f"{args.output_dir}/results.csv" )

    print( "End of training." )

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="""GNN4DM: a graph neural network-based structured model that automates the discovery of 
                                                    overlapping functional disease modules. GNN4DM integrates network topology with
                                                    genomic data to learn the representations of the genes corresponding to functional 
                                                    modules and align these with known biological pathways for enhanced interpretability.""",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--graph', type=str, required=True, metavar="FILE_PATH", help='Path to the TXT file containing graph edges (two columns containing node identifiers separated by a whitespace).')
    parser.add_argument('--input_features', type=str, required=True, metavar="FILE_PATH", nargs='+', help='Paths to one or more TXT files containing node features (first column: node ids; no headers). All files will be merged to form the initial input features for the nodes in the graph. Missing values will be imputed using feature-wise mean values.')
    parser.add_argument('--pathway_databases', type=str, required=True, metavar="FILE_PATH", nargs='+', help='Paths to one or more GMT files containing pathway annotations (using the same ids as in the graph).')
    # Add other hyperparameters as optional arguments
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory.')
    parser.add_argument('--module_count', type=int, default=500, help='Maximum number of modules to detect.')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--learning_rate_decay_step_size', type=int, default=250, help='Learning rate decay step size.')
    parser.add_argument('--hidden_channels_before_module_representation', type=int, default=128, help='Number of hidden channels before module representation.')
    parser.add_argument('--threshold', type=str, default='auto', help='Threshold for edge weights. Can be a float or "auto".')
    parser.add_argument('--batchnorm', type=bool, default=True, help='Whether to use batch normalization.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 penalty).')
    parser.add_argument('--lambda_bce_loss', type=float, default=10.0, help='Lambda for BCE loss.')
    parser.add_argument('--lambda_l1_positives_loss', type=float, default=0.0, help='Lambda for L1 loss on positives.')
    parser.add_argument('--lambda_l2_positives_loss', type=float, default=0.0, help='Lambda for L2 loss on positives.')
    parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN','MLP','GAT','SAGE'], help='Type of model to use.')

    args = parser.parse_args()

    print_version_info()

    # Load and process the graph data
    G = loadGraph(args.graph)

    # Load and process the input features
    input_features = loadInputFeatures( G, args.input_features )

    # Load and process the pathway databases
    ground_truth, train_indices, valid_indices, term_ids = loadPathwayDatabases( G=G, filenames=args.pathway_databases )

    node_dict = dict( zip( list(G.nodes), range(len(G.nodes)) ) )
    node_index_dict = dict( zip( range(len(G.nodes)), list(G.nodes) ) )

    # relabel nodes (into numbers)
    GPrime = nx.relabel_nodes( G, node_dict )
    
    # convert to pyg data format
    data = from_networkx( GPrime )
    data.x = input_features

    data.ys = {**ground_truth}
    data.train_indices = {**train_indices}
    data.valid_indices = {**valid_indices}

    dataset = { 'data': data, 
                'graph': GPrime, 
                'node_dict': node_dict, 
                'node_index_dict': node_index_dict,
                'term_ids': term_ids }
    
    # Set the warnings to be captured with a custom handler
    #warnings.showwarning = warn_handler
    #warnings.filterwarnings('always', category=RuntimeWarning)
    #warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    runModel( dataset, args )
