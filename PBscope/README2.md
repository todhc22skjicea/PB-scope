##PBscope: Drug Screening Based on P-body

This is the official implementation of paper []
(https://).

## Overview
PBscope is a pipeline for drug screening for P-bodies (processing bodies). which aims to identify potential drug candidates that can modulate P-body function, thereby affecting gene expression and cellular processes.
As shown in the paper, PBscope revealed that the potential relationship between JAK/STAT signaling pathway and P-body phenotype, uncovering a previously unrecognized mechanism of cancer progression.

## Installation
To install PBscope, follow these following steps:

```
conda create -n pbscope python=3.8 &&
conda activate pbscope &&
conda config --add channels conda-forge &&
conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch &&
pip install scipy wandb PyYAML scikit-learn termcolor matplotlib opencv-contrib-python &&
```
pip install -r requirements.txt

## Dataset Preparation
The dataset for P-body screening are now available at https://doi.org/10.5281/zenodo.14591158

## Usage

Basic Usage
Data Preparation:
Ensure that your input data (e.g., genomic data, proteomic data, imaging data) is formatted correctly and placed in the appropriate directory.

Running PBscope:

To run experiments
```
 python main.py --preset drug_screen 
## See also
[Original implementation of CC](https://github.com/Yunfan-Li/Contrastive-Clustering)

#Results Visualization:

After the screening process is complete, results can be visualized using the built-in plotting functions or exported for further analysis.

#Documentation

For comprehensive documentation, including usage examples, parameter descriptions, and troubleshooting tips, please refer to the PBscreen Documentation.

## Citation
If you use this repository, please cite:
```
@inproceedings{
}
```

## License

This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgement

We thank the authors of the below for making their code public.
[Original implementation of CC](https://github.com/Yunfan-Li/Contrastive-Clustering)
[Original implementation of Divclust](https://github.com/ManiadisG/DivClust)