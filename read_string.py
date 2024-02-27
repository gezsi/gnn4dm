import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import random
import gzip

from sklearn.decomposition import PCA

def getSTRING( cutoff = 700 ):
    """
    Create network from STRING using a prespecified cutoff.
    """
    # Read raw STRING PPI data
    string_raw = pd.read_csv( "./data/StringDB/9606.protein.links.detailed.v12.0.txt.gz", sep=" " )
    string_raw = string_raw[ string_raw['combined_score'] >= cutoff ]
    string_raw['protein1'] = string_raw['protein1'].str.replace("9606.", "", regex = False)
    string_raw['protein2'] = string_raw['protein2'].str.replace("9606.", "", regex = False)

    # Read gene info
    ensembl = pd.read_csv( "./data/StringDB/ensemblGene2Protein_GRCh37.txt", sep = "\t" )
    ensembl = ensembl.rename( columns={ 'Gene stable ID': 'GeneID', 'Protein stable ID': 'ProteinID' } )

    string_db = pd.merge( left=pd.merge(left = string_raw, right=ensembl, how='inner', left_on='protein1', right_on='ProteinID'), right=ensembl, how='inner', left_on='protein2', right_on='ProteinID')[['GeneID_x','GeneID_y']]
    string_db = string_db.drop_duplicates()                                 # filter out duplicate edges
    string_db = string_db[ string_db['GeneID_x'] != string_db['GeneID_y'] ] # filter out self edges

    # create networkx graph from the data frame
    STRING = nx.from_pandas_edgelist(string_db,'GeneID_x','GeneID_y')
    STRING.name = 'STRING-DB v12'

    # select the largest connected component
    STRING = nx.subgraph(STRING, max(nx.connected_components(STRING), key=len))

    nx.write_edgelist( STRING, "./data/StringDB/STRING_v12_edgelist.txt", data=False, delimiter='\t' )

    print( f'Network: {STRING.name}' )
    print( f'Number of nodes = {nx.number_of_nodes(STRING)}' )
    print( f'Number of edges = {nx.number_of_edges(STRING)}' )
    print( f'Directed = {STRING.is_directed()}' )

    return STRING

def getGTExFeatures( STRING ):
    gtex_raw = pd.read_csv( "./data/GTEx/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz", sep="\t", skiprows=2 )
    
    # Substring the 'Name' column to keep only the first 15 characters
    gtex_raw['Name'] = gtex_raw['Name'].str[:15]

    # Remove the 'Description' column
    gtex_raw = gtex_raw.drop(columns='Description')

    # Group by 'Name' and calculate the median for each column
    gtex_raw = gtex_raw.groupby('Name').median()

    # Convert the dataframe to a numpy matrix
    gtex = gtex_raw.iloc[:, :].values
    rownames_m = gtex_raw.index.values

    # Apply log2 transformation to the matrix
    gtex = np.log2(gtex + 1)

    # # Plot a histogram of the matrix m
    # plt.hist(gtex.flatten(), bins=50, log=True)
    # plt.xlabel('Log2 Transformed Values')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Log2 Transformed Values')
    # plt.show()

    # Step 0: Get the list of nodes from the NetworkX graph "G"
    nodes_list = list(STRING.nodes())

    # Step 1: Calculate the column-wise mean of the numpy matrix "m"
    columnwise_mean = np.mean(gtex, axis=0)

    # Step 2: Create a DataFrame from the numpy matrix "m" with row and column names
    df = pd.DataFrame(gtex, index=rownames_m, columns=range(0, gtex.shape[1]))

    # Step 3: Align the rows of the DataFrame with the nodes in the NetworkX graph "G"
    # Create a new DataFrame with the same columns as the original DataFrame and index as nodes_list
    aligned_df = pd.DataFrame(columns=df.columns, index=nodes_list)

    n_missing = 0

    # Iterate through the nodes and copy the corresponding row from the original DataFrame
    for node in nodes_list:
        if node in df.index:
            aligned_df.loc[node] = df.loc[node]
        else:
            # If the node is not in "m," fill the row with column-wise mean values
            aligned_df.loc[node] = columnwise_mean
            n_missing += 1

    aligned_df.astype(float).to_csv('./data/GTEx/GTEx_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm_processed.txt', 
                      sep='\t', 
                      index=True, 
                      header=False,
                      float_format='%.4f')

    # Step 4: Convert the aligned DataFrame to a numpy matrix
    gtex_aligned_matrix = aligned_df.to_numpy()

    print( "Number of gene ids with missing information: " + str(n_missing) )
    print( "GTEx feature matrix shape: " + str(gtex_aligned_matrix.shape) )

    return gtex_aligned_matrix

def getGWASAtlasData():
    f = gzip.open("./data/GWASAtlas/gwasATLAS_v20191115.xlsx.gz", 'rb')
    gwasatlas_studies = pd.read_excel( f )
    f.close()
    gwasatlas_raw = pd.read_csv( "./data/GWASAtlas/gwasATLAS_v20191115_magma_P.txt.gz", sep="\t" )

    # Group by 'GENE' and calculate the median for each column
    gwasatlas_raw = gwasatlas_raw.groupby('GENE').median()

    # Convert the dataframe to a numpy matrix
    gwasatlas = gwasatlas_raw.iloc[:,:].values
    rownames_m = gwasatlas_raw.index.values

    # Plot a histogram of the raw matrix gwasatlas
    plt.hist(gwasatlas.flatten(), bins=50, log=True)
    plt.xlabel('Raw P-values')
    plt.ylabel('Frequency')
    plt.title('Histogram of Raw P-values')
    plt.show()

    # Set missing values to 0.5 => not significant (~ global mean)
    gwasatlas[np.isnan(gwasatlas)] = 0.5

    # Apply -log10 transformation to the matrix
    gwasatlas = -1.0 * np.log10(gwasatlas)

    # Plot a histogram of the transformed matrix 
    plt.hist(gwasatlas.flatten(), bins=50, log=True)
    plt.xlabel('-Log10 Transformed Values')
    plt.ylabel('Frequency')
    plt.title('Histogram of -Log10 Transformed Values')
    plt.show()

    ukb_studies = ['UKB1 (EUR)','UKB2 (EUR)','UKB2 (EUR meta)','UKB1 (EUR meta)']
    gwasatlas_ukb = gwasatlas[:,gwasatlas_studies[gwasatlas_studies['Population'].isin(ukb_studies)].index.values]
    gwasatlas_nonukb = gwasatlas[:,gwasatlas_studies[~gwasatlas_studies['Population'].isin(ukb_studies)].index.values]

    return gwasatlas, gwasatlas_ukb, gwasatlas_nonukb, rownames_m

def getGWASAtlasFeatures( STRING, filter : str = None, performPCA : bool = True, num_components : int = 1024 ):
    valid_filter_values = ['all', 'ukb', 'non-ukb']

    if filter not in valid_filter_values:
        raise ValueError(f"Invalid argument. Must be one of: {', '.join(valid_filter_values)}")
    
    dataset_filename = f"./data/STRING_{filter}_{performPCA}{num_components if performPCA else ''}_gwasatlas.npy"

    if os.path.exists( dataset_filename ):
        with open(dataset_filename, 'rb') as f:
            gwasatlas_aligned_matrix = np.load(f, allow_pickle=True)

        # print( "GWASAtlas feature matrix shape: " + str(gwasatlas_aligned_matrix.shape) )

    else:

        gwasatlas_all, gwasatlas_ukb, gwasatlas_nonukb, rownames_m = getGWASAtlasData()

        print( f"gwasatlas_ukb.shape: {gwasatlas_ukb.shape}" )
        print( f"gwasatlas_nonukb.shape: {gwasatlas_nonukb.shape}" )

        if filter == 'all':
            gwasatlas = gwasatlas_all
        elif filter == 'ukb':
            gwasatlas = gwasatlas_ukb
        elif filter == 'non-ukb':
            gwasatlas = gwasatlas_nonukb

        if performPCA:
            # Perform PCA transformation for dimensionality- and noise reduction
            pca = PCA(n_components=num_components) 

            # Step 3: Fit the PCA model
            pca.fit(gwasatlas)

            # Step 4: Transform the data using the learned PCA model
            transformed_data = pca.transform(gwasatlas)

            # Explained variance (percentage of variance explained by each component)
            explained_variance = pca.explained_variance_ratio_

            # Cumulative explained variance
            cumulative_variance = np.cumsum(explained_variance)

            print("Cumulative Explained Variance:")
            print(cumulative_variance[-1])

            # Step 4: Plot the cumulative explained variance
            plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
            plt.xlabel('Number of Principal Components')
            plt.ylabel('Cumulative Explained Variance')
            plt.title('Cumulative Explained Variance Plot')
            plt.grid()
            plt.show()

        else:
            transformed_data = gwasatlas

        # Step 0: Get the list of nodes from the NetworkX graph "G"
        nodes_list = list(STRING.nodes())

        # Step 1: Calculate the column-wise mean of the numpy matrix "m"
        columnwise_mean = np.mean(transformed_data, axis=0)

        # Step 2: Create a DataFrame from the numpy matrix "m" with row and column names
        df = pd.DataFrame(transformed_data, index=rownames_m, columns=range(0, transformed_data.shape[1]))

        # Step 3: Align the rows of the DataFrame with the nodes in the NetworkX graph "G"
        # Create a new DataFrame with the same columns as the original DataFrame and index as nodes_list
        aligned_df = pd.DataFrame(columns=df.columns, index=nodes_list)

        n_missing = 0

        # Iterate through the nodes and copy the corresponding row from the original DataFrame
        for node in nodes_list:
            if node in df.index:
                aligned_df.loc[node] = df.loc[node]
            else:
                # If the node is not in "m," fill the row with column-wise mean values
                aligned_df.loc[node] = columnwise_mean
                n_missing += 1

        aligned_df.astype(float).to_csv(f"./data/GWASAtlas/gwasATLAS_v20191115_magma_P_{filter}_{performPCA}{num_components if performPCA else ''}_processed.txt", 
                          sep='\t', 
                          index=True, 
                          header=False,
                          float_format='%.4f')

        # Step 4: Convert the aligned DataFrame to a numpy matrix (if needed)
        gwasatlas_aligned_matrix = aligned_df.to_numpy()

        # Now, "aligned_matrix" contains the matrix where each row corresponds to a node in the NetworkX graph "G."
        # If a node is not present in the "m" matrix, the corresponding row will be filled with column-wise mean values.
        # You can further use this "aligned_matrix" for analysis or any other computations.
        print( "#Missing gene ids: " + str(n_missing) )
        print( "GWASAtlas feature matrix shape: " + str(gwasatlas_aligned_matrix.shape) )

        with open(dataset_filename, 'wb') as f:
            np.save(f, gwasatlas_aligned_matrix)

    assert gwasatlas_aligned_matrix.shape[0] == STRING.number_of_nodes()

    return gwasatlas_aligned_matrix


def getMSigDBFeatures( STRING, dbs : list ):

    msigdb_dict_df = pd.read_csv( './data/MSigDB/msigdb_folds.csv', sep=',' )

    list_nodes = list(STRING.nodes())

    # Create a dictionary to store tensors for each DB
    db_tensors = {}
    train_indices = {}
    valid_indices = {}
    term_ids = {}

    dbs_filtered = set(dbs).intersection(set(msigdb_dict_df['DB']))

    # Group the DataFrame by DB, then create multi-hot encoded tensors
    grouped = msigdb_dict_df.groupby('DB')
    for (db), group in grouped:
        if db in dbs_filtered:
            # Create an empty tensor with the correct number of columns and rows
            tensor_shape = (len(list_nodes), group.Name.nunique())
            db_tensor = torch.zeros(*tensor_shape, dtype=torch.float32)
            train_mask = torch.zeros_like(db_tensor, dtype=torch.bool)
            valid_mask = torch.zeros_like(db_tensor, dtype=torch.bool)

            # find those node indices on which we have info in this db (i.e. those geneids that participate in at least one pathway in this db)
            # this will be the union of the training and validation indices
            known_node_indices_in_db = {list_nodes.index(ensembl_id) for ensembl_id in group['EnsemblId'].unique()}

            named_grouped = group.groupby(['Name'])
            name_index = 0
            names = list()
            for (name,), named_group in named_grouped:
                assert named_group['Train_Test'].nunique() == 1, "Not a train or test group"
                
                train_group = named_group['Train_Test'].iloc[0] == "TRAIN"

                if train_group:
                    # Get the row indices for EnsemblIds from the list_nodes
                    positive_train_indices = {list_nodes.index(ensembl_id) for ensembl_id in named_group['EnsemblId']}
                    
                    # Set the corresponding values to 1 in the tensor
                    db_tensor[list(positive_train_indices), name_index] = 1.0

                    train_mask[list(known_node_indices_in_db), name_index] = True

                else:
                    # Get the row indices for EnsemblIds from the list_nodes
                    positive_train_indices = {list_nodes.index(ensembl_id) for ensembl_id in named_group[named_group['Fold_STRING'] != 1]['EnsemblId']}
                    positive_valid_indices = {list_nodes.index(ensembl_id) for ensembl_id in named_group[named_group['Fold_STRING'] == 1]['EnsemblId']}

                    potential_negative_indices = known_node_indices_in_db - positive_train_indices - positive_valid_indices

                    num_elements_to_select = int(len(potential_negative_indices) * 0.8)
                    negative_train_indices = set(random.choices(list(potential_negative_indices),k=num_elements_to_select))

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

                names.append( name )
                name_index += 1

            # Store the tensor in the dictionary
            db_tensors[db] = db_tensor
            train_indices[db] = train_mask
            valid_indices[db] = valid_mask
            term_ids[db] = names

    return db_tensors, train_indices, valid_indices, term_ids

