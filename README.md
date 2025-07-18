# PBscope: Contrastive learning of dynamic processing body formation reveals undefined mechanisms of approved compounds

## Overview
![Framework of PBscope](https://github.com/todhc22skjicea/PB-scope/blob/main/PBscope/asset/PBscope.png)

This is the official implementation of paper [PB-scope: Contrastive learning of dynamic processing body formation reveals undefined mechanisms of approved compounds](https://www.biorxiv.org/content/10.1101/2025.06.14.659731v1).



PBscope is a pipeline for AI drug screening method for P-bodies (processing bodies), which aims to identify potential drug candidates that can modulate P-body function, thereby affecting gene expression and cellular processes.

As shown in the [paper](https://www.biorxiv.org/content/10.1101/2025.06.14.659731v1), PBscope revealed that the potential relationship between JAK/STAT signaling pathway and P-body phenotype, uncovering a previously unrecognized mechanism of cancer progression.


## Installation
To install PBscope, follow these following steps:

```
conda create -n pbscope python=3.8 &&
conda activate pbscope &&
conda config --add channels conda-forge &&
conda install pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch &&
pip install scipy wandb PyYAML scikit-learn termcolor matplotlib opencv-contrib-python &&
pip install -r requirements.txt
```

## Dataset Preparation
The dataset for P-body screening are now available at https://doi.org/10.5281/zenodo.14591158

## Usage

Basic Usage
Data Preparation:
Ensure that your input data (e.g., genomic data, proteomic data, imaging data) is formatted correctly and placed in the appropriate directory.

Running PBscope:

To train the model for unsupervised cluster on drug screening dataset, run
```
python main.py --preset drug_screen
```
To virsualize the distribution of drug phenotype features in embedding space, run
```
python test.py
python vir.py
```
To virsualize the distribution of drug phenotype features in two duplicate experiments, run
```
python repeat.py
```

## Citation
If you use this repository, please cite:
```
@article {Shen2025.06.14.659731,
	author = {Shen, Dexin and Zhu, Qionghua and Pang, Xiquan and Yang, Xian and Pan, Dongzhen and Zhang, Mengyang and Li, Yanping and Sun, Zhiyuan and Fang, Liang and Chen, Wei and Tsuboi, Tatsuhisa},
	title = {PB-scope: Contrastive learning of dynamic processing body formation reveals undefined mechanisms of approved compounds},
	elocation-id = {2025.06.14.659731},
	year = {2025},
	doi = {10.1101/2025.06.14.659731},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/06/15/2025.06.14.659731},
	eprint = {https://www.biorxiv.org/content/early/2025/06/15/2025.06.14.659731.full.pdf},
	journal = {bioRxiv}
}
```

## License

This project is licensed under the MIT License. See LICENSE for details.

## Acknowledgement

During our research, we drew inspiration from and referenced the following works. 

[Original implementation of CC](https://github.com/Yunfan-Li/Contrastive-Clustering)

[Original implementation of Divclust](https://github.com/ManiadisG/DivClust)

We thank the authors of above for making their codes public.
