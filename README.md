# gnn4dm
GNN4DM: A Graph Neural Network-based method to identify overlapping functional disease modules

This is the original implementation of GNN4DM, a graph neural network-based structured model that automates the discovery of overlapping functional disease modules. GNN4DM effectively integrates network topology with genomic data to learn the representations of the genes corresponding to functional modules and align these with known biological pathways for enhanced interpretability.

# Usage
GNN4DM is a standalone command-line Python program designed to identify overlapping communities within networks specified by the user.

The program requires input comprising 
- a graph structure specified by unweighted, undirected edges, 
- numeric features of the nodes, 
- and annotations categorizing the nodes into higher-level entities, such as the known biological functions associated with gene nodes.

```
usage: gnn4dm.py [-h] --graph FILE_PATH --input_features FILE_PATH [FILE_PATH ...] --pathway_databases FILE_PATH [FILE_PATH ...] [--output_dir OUTPUT_DIR] [--module_count MODULE_COUNT]
                 [--learning_rate LEARNING_RATE] [--learning_rate_decay_step_size LEARNING_RATE_DECAY_STEP_SIZE]
                 [--hidden_channels_before_module_representation HIDDEN_CHANNELS_BEFORE_MODULE_REPRESENTATION] [--threshold THRESHOLD] [--batchnorm BATCHNORM] [--dropout DROPOUT]
                 [--weight_decay WEIGHT_DECAY] [--lambda_bce_loss LAMBDA_BCE_LOSS] [--lambda_l1_positives_loss LAMBDA_L1_POSITIVES_LOSS] [--lambda_l2_positives_loss LAMBDA_L2_POSITIVES_LOSS]
                 [--model_type {GCN,MLP,GAT,SAGE}]

options:
  -h, --help            show this help message and exit
  --graph FILE_PATH     Path to the TXT file containing graph edges (two columns containing node identifiers separated by a whitespace). (default: None)
  --input_features FILE_PATH [FILE_PATH ...]
                        Paths to one or more TXT files containing node features (first column: node ids; no headers). All files will be merged to form the initial input features for the nodes in
                        the graph. Missing values will be imputed using feature-wise mean values. (default: None)
  --pathway_databases FILE_PATH [FILE_PATH ...]
                        Paths to one or more GMT files containing pathway annotations (using the same ids as in the graph). (default: None)
  --output_dir OUTPUT_DIR
                        Output directory. (default: ./results)
  --module_count MODULE_COUNT
                        Maximum number of modules to detect. (default: 500)
  --learning_rate LEARNING_RATE
                        Learning rate. (default: 0.001)
  --learning_rate_decay_step_size LEARNING_RATE_DECAY_STEP_SIZE
                        Learning rate decay step size. (default: 250)
  --hidden_channels_before_module_representation HIDDEN_CHANNELS_BEFORE_MODULE_REPRESENTATION
                        Number of hidden channels before module representation. (default: 128)
  --threshold THRESHOLD
                        Threshold for edge weights. Can be a float or "auto". (default: auto)
  --batchnorm BATCHNORM
                        Whether to use batch normalization. (default: True)
  --dropout DROPOUT     Dropout rate. (default: 0.0)
  --weight_decay WEIGHT_DECAY
                        Weight decay (L2 penalty). (default: 0.0)
  --lambda_bce_loss LAMBDA_BCE_LOSS
                        Lambda for BCE loss. (default: 10.0)
  --lambda_l1_positives_loss LAMBDA_L1_POSITIVES_LOSS
                        Lambda for L1 loss on positives. (default: 0.0)
  --lambda_l2_positives_loss LAMBDA_L2_POSITIVES_LOSS
                        Lambda for L2 loss on positives. (default: 0.0)
  --model_type {GCN,MLP,GAT,SAGE}
                        Type of model to use. (default: GCN)
```

## Using original training data
To train GNN4DM using the original data from the paper, you should use the following input files from the './data' directory:

--graph:  
[data/StringDB/STRING_v12_edgelist.txt](data/StringDB/STRING_v12_edgelist.txt) (STRING DB v12 mapped to Ensembl gene identifiers.)

--input_features:  
[data/GTEx/GTEx_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm_processed.txt](data/GTEx/GTEx_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm_processed.txt) (Genome-wide gene expression measurement data from the Genotype-Tissue Expression (GTEx) project. Log-transformed, median gene-level transcript per million (TPM) values across 54 human tissues.)  
[data/GWASAtlas/gwasATLAS_v20191115_magma_P_non-ukb_True512_processed.txt](data/GWASAtlas/gwasATLAS_v20191115_magma_P_non-ukb_True512_processed.txt) (Negative
log-transformed gene-level p-values from the GWAS Atlas project, filtered to the non-UKB specific GWAS summary statistics, and transformed with PCA to get the first 512 principal components.)  
[data/StringDB/centrality_measures.txt](data/StringDB/centrality_measures.txt) (Various centrality measures computed on the STRING PPI graph.)  

--pathway_databases:  
[data/MSigDB/biocarta_ensembl.gmt](data/MSigDB/biocarta_ensembl.gmt) (BioCarta pathway annotations mapped to Ensembl gene identifiers.)  
[data/MSigDB/kegg_ensembl.gmt](data/MSigDB/kegg_ensembl.gmt) (KEGG pathway annotations mapped to Ensembl gene identifiers.)  
[data/MSigDB/reactome_ensembl.gmt](data/MSigDB/reactome_ensembl.gmt) (Reactome pathway annotations mapped to Ensembl gene identifiers.)  
[data/MSigDB/wikipathways_ensembl.gmt](data/MSigDB/wikipathways_ensembl.gmt) (WikiPathways pathway annotations mapped to Ensembl gene identifiers.)  

i.e., the following command can be used:

```
python gnn4dm.py --graph ./data/StringDB/STRING_v12_edgelist.txt --input_features ./data/GTEx/GTEx_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm_processed.txt ./data/GWASAtlas/gwasATLAS_v20191115_magma_P_non-ukb_True512_processed.txt ./data/StringDB/centrality_measures.txt --pathway_databases ./data/MSigDB/biocarta_ensembl.gmt ./data/MSigDB/kegg_ensembl.gmt ./data/MSigDB/reactome_ensembl.gmt ./data/MSigDB/wikipathways_ensembl.gmt
```

# Precomputed functional disease modules of the human interactome (based on STRING)
We applied the GNN4DM framework on the human interactome derived from the STRING protein-protein interaction database to identify its overlapping functional modules. The modules can be accessed from the `results` folder in Gene Matrix Transposed file (GMT) format. We identified the modules using [500](results/gnn4dm_500_string.gmt), [600](results/gnn4dm_600_string.gmt), [700](results/gnn4dm_700_string.gmt), [800](results/gnn4dm_800_string.gmt), [900](results/gnn4dm_900_string.gmt), and [1000](results/gnn4dm_1000_string.gmt) maximum module counts.

## Installation of required libraries
The following command can be used to install all required libraries using conda. Replace CUDAVERSION with your cuda version (e.g., 11.8).
```
conda create --name gnn4dm

conda activate gnn4dm

conda install pytorch torchvision torchaudio pytorch-cuda=CUDAVERSION -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c conda-forge ipykernel ipywidgets pandas networkx matplotlib-base tensorboard gzip openpyxl

pip install cdlib
```

## Cite
Please cite our paper if you use the code or the datasets in your own work:
```
@article{
    gezsi2024gnn4dm,
    title={GNN4DM: A Graph Neural Network-based method to identify overlapping functional disease modules},
    author={Andras Gezsi and Peter Antal},
    journal={In preparation},
    year={},
}
```
