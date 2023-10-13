# CBM-AUC
Sample code of Concept Bottleneck Model with Additional Unsuperviead Concepts (CBM-AUC).

These codes are based on [Self-Explaining Neural Networks (SENN)](https://github.com/dmelis/SENN) and partially based on [Concept Bottleneck Models (CBM)](https://github.com/yewsiang/ConceptBottleneck).

## Dependencies
Please check Dockerfile

## Usage
### Preparation

Please download the datasets and pretrained models. 

- CUB-200-2011: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

- BDD-OIA : https://twizwei.github.io/bddoia_project/

- Faster RCNN for BDD100k: https://github.com/JiqianDong/BDD-100-CV

- inception v.3 for CUB-200-2011: https://github.com/yewsiang/ConceptBottleneck

The original label of BDD-OIA is written in json format, while our model uses pkl. So, after downloading, please use `convert_json_to_pkl.ipynb` to convert the json to pkl format and merge the action and reason file.

Then, make `data`, `models`, `out` folders, and store the downloaded data to `data` and the pretrained models to `models`, respectively (`out` is used to store results).


### Examples of Execution (CBM-AUC)
```bash
# CUB-200-2011
python main_cub.py --train --cuda --load_model --h_type fcc --epochs 5 --batch_size 64 --nconcepts 128 --nconcepts_labeled 112 --h_sparsity 7 --opt sgd --lr 0.001 --weight_decay 0.00004 --h_labeled_param 1.0 --theta_reg_lambda 0.001 --info_hypara 0.5
# BDD-OIA
python main_bdd.py --train --cuda --load_model --h_type fcc --epochs 5 --batch_size 16 --nconcepts 30 --nconcepts_labeled 21 --h_sparsity 7 --opt adam --lr 0.001 --weight_decay 0.00004 --h_labeled_param 1.0 --theta_reg_lambda 0.001 --info_hypara 0.5 --obj bce
```
If you want to full train, please change the epoch number according to our paper.

# Citation
Yoshihide Sawada and Keigo Nakamura. "[Concept Bottleneck Model with Additional Unsupervised Concepts](https://ieeexplore.ieee.org/abstract/document/9758745)". IEEE Access (2022).

# Commercial License
The open source license is in [LICENSE](./LICENSE) file. This software is also available for licensing via [AISIN Corp](https://www.aisin.com/).
