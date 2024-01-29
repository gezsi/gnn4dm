# gnn4dm
GNN4DM: A Graph Neural Network-based method to identify overlapping functional disease modules

This is the original implementation of GNN4DM, a graph neural network-based structured model that automates the discovery of overlapping functional disease modules. GNN4DM effectively integrates network topology with genomic data to learn the representations of the genes corresponding to functional modules and align these with known biological pathways for enhanced interpretability.

# Usage
## Download training data
To train GNN4DM using the original data from the paper, various datasets must be downloaded from their public repositories. The following bash commands, executed from the root directory of the gnn4dm repository, can be used for this purpose:
```bash
wget -c -P ./data/StringDB/ https://stringdb-downloads.org/download/protein.links.detailed.v12.0/9606.protein.links.detailed.v12.0.txt.gz
wget -c -P ./data/GTEx/ https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz 
```

Also, please download the following files manually from the GWAS Atlas (`https://atlas.ctglab.nl/`) into the `./data/GWASAtlas` directory:
```
gwasATLAS_v20191115.xlsx.gz
gwasATLAS_v20191115_magma_P.txt.gz
```

## Identifying modules
A Jupyter notebook [identify_modules.ipynb](identify_modules.ipynb) contains the code for training the model and exporting the results.

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
