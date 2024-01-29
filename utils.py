import networkx as nx
import numpy as np

from cdlib import NodeClustering

from sklearn.metrics import precision_recall_curve, auc

import json

import torch
import torch.nn.functional as FN


def cosine_similarity(a, b, eps=1e-8):
    """
    Computes cosine similarity of all columns in a matrix with all columns in an other matrix.
    """
    a_n, b_n = a.norm(dim=0)[None, :], b.norm(dim=0)[None, :]
    a_norm = a / torch.clamp(a_n, min=eps)
    b_norm = b / torch.clamp(b_n, min=eps)
    sim_mt = torch.mm(a_norm.transpose(0, 1), b_norm)
    return sim_mt

def cosine_self_similarity(a, eps=1e-8):
    """
    Computes cosine similarity of all columns in a matrix.
    """
    a_n = a.norm(dim=0)[None, :]
    a_norm = a / torch.clamp(a_n, min=eps)
    sim_mt = torch.mm(a_norm.transpose(0, 1), a_norm)
    return sim_mt

def getPredictecCommunities2(F : torch.Tensor, G : nx.Graph, threshold = 0.1, normalize = True):
    # normalize if needed
    H = FN.normalize( F, p=1, dim=1 ) if normalize else F

    # create list of list indices
    comms = [[] for i in range(H.shape[1])]
    
    # if we have to cut by a threshold
    if threshold != None:
        H_filtered = (H >= threshold).float().cpu().detach().numpy()
        
        row,col = np.nonzero(H_filtered)
        for i,r in enumerate(row):
            comms[col[i]].append(r)
    else: # if the row-wise max is needed (i.e. non-overlapping clustering)
        # get indices
        _, indices = torch.max(H, dim=1)
        
        for i in range(H.shape[1]):
            comms[i] = list(torch.nonzero( indices == i ).cpu().detach().numpy().flatten())

    # filter out empty communities
    comms = [x for x in comms if x != []]

    return NodeClustering( communities=comms, graph=G, overlap=True )

def getPredictecCommunities3(F : torch.Tensor, G : nx.Graph, threshold = 0.1, normalize = True):
    # normalize if needed
    H = FN.normalize( F, p=1, dim=1 ) if normalize else F

    # create list of list indices
    comms = [[] for i in range(H.shape[1])]
    
    # if we have to cut by a threshold
    if threshold != None:
        H_filtered = (H >= threshold).float().cpu().detach().numpy()
        
        row,col = np.nonzero(H_filtered)
        for i,r in enumerate(row):
            comms[col[i]].append(r)
    else: # if the row-wise max is needed (i.e. non-overlapping clustering)
        # get indices
        _, indices = torch.max(H, dim=1)
        
        for i in range(H.shape[1]):
            comms[i] = list(torch.nonzero( indices == i ).cpu().detach().numpy().flatten())

    # filter out empty communities while creating NodeClustering
    # return the original clusters as well as a list, as cdlib sorts the communities by length
    # and we loose the mapping between the weights and the modules

    return NodeClustering( communities=[x for x in comms if x != []], graph=G, overlap=True ), comms

def inv_softplus(bias: float | torch.Tensor) -> float | torch.Tensor:
    """Inverse softplus function.

    Args:
        bias (float or tensor): the value to be softplus-inverted.
    """
    is_tensor = True
    if not isinstance(bias, torch.Tensor):
        is_tensor = False
        bias = torch.tensor(bias)
    out = bias.expm1().clamp_min(1e-6).log()
    if not is_tensor and out.numel() == 1:
        return out.item()
    return out

def calculateMetrics( y_preds : dict, ys : dict, masks : dict ):

    accuracies = {}
    sensitivities = {}
    specificities = {}
    precisions = {}
    f1_scores = {}
    auprcs = {}

    for key in y_preds:
        probabilities = torch.sigmoid(y_preds[key]) if masks[key].dim() == 2 else y_preds[key]

        # Step 2: Apply mask
        probabilities_masked = probabilities[masks[key]] if masks[key].dim() == 2 else probabilities[masks[key],:]
        ground_truth_masked = ys[key][masks[key]] if masks[key].dim() == 2 else ys[key][masks[key],:]

        # Step 3: Apply thresholding (e.g., 0.5)
        binary_predictions = (probabilities_masked > 0.5).float()

        # Step 4: Compute confusion matrix
        true_positive = (binary_predictions * ground_truth_masked).sum()
        false_positive = (binary_predictions * (1 - ground_truth_masked)).sum()
        false_negative = ((1 - binary_predictions) * ground_truth_masked).sum()
        true_negative = ((1 - binary_predictions) * (1 - ground_truth_masked)).sum()

        # Step 5: Compute accuracy, sensitivity, specificity, precision, and F1 score
        accuracies[key] = ((true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)).item()
        sensitivities[key] = (true_positive / (true_positive + false_negative)).item()
        specificities[key] = (true_negative / (true_negative + false_positive)).item()
        precisions[key] = (true_positive / (true_positive + false_positive)).item()
        f1_scores[key] = (2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative)).item()

        # Step 6: Compute Area Under the Precision-Recall Curve
        # Initialize a list to store AUC-PR values for each label
        auc_pr_values = []
        # Iterate through each column (label)
        for label_idx in range(probabilities.shape[1]):            
            # Get the evaluation mask for the current label
            label_mask = masks[key][:, label_idx] if masks[key].dim() == 2 else masks[key]
            # Check if there are any true items in the evaluation mask for this label
            if label_mask.any():
                # Get the predicted probabilities, and ground truth labels for the current label
                label_predictions = probabilities[:, label_idx]
                label_ground_truth = ys[key][:, label_idx]
                
                # Filter instances based on the evaluation mask
                label_predictions = label_predictions[label_mask]
                label_ground_truth = label_ground_truth[label_mask]

                # Compute the precision-recall curve
                precision, recall, _ = precision_recall_curve(label_ground_truth.cpu().numpy(), label_predictions.cpu().numpy())

                # Compute the AUC-PR for the current label
                label_auc_pr = auc(recall, precision)

                # Append the AUC-PR value to the list
                auc_pr_values.append(label_auc_pr)
           
        auprcs[key] = np.mean(auc_pr_values)

    return accuracies, sensitivities, specificities, precisions, f1_scores, auprcs

def saveResultsToJSON( internal_evaluation_scores, communities, filename : str, node_index_dict ):
    
    partition = {
        "communities": [[node_index_dict[int(value)] for value in sublist] for sublist in communities]
    }

    # Step 2: Convert NumPy int64 to Python int
    existing_dict_converted = {key: int(value) if isinstance(value, np.int64) else value for key, value in internal_evaluation_scores.items()}

    # Step 2: Merge dictionaries
    merged_dict = {**existing_dict_converted, **partition}

    # Optionally, convert merged dictionary back to JSON-formatted string
    merged_json_string = json.dumps(merged_dict)

    # Open the file in append mode
    with open(filename, "wt") as file:
        file.write(merged_json_string)
