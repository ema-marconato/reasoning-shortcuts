# Analysis and Mitigation of Reasoning Shortcuts

Codebase for the paper: 

Not All Neuro-Symbolic Concepts Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts, E. Marconato, S. Teso, A. Vergari, A. Passerini - NeurIPS (2023)

```
@misc{marconato2023neurosymbolic,
      title={Not All Neuro-Symbolic Concepts Are Created Equal: Analysis and Mitigation of Reasoning Shortcuts}, 
      author={Emanuele Marconato and Stefano Teso and Antonio Vergari and Andrea Passerini},
      year={2023},
      eprint={2305.19951},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Installation and use

To run experiments on XOR, MNIST-Addition, and BDD-OIA -- access the linux terminal and use the conda installation followed by pip3:

```
$conda env create -n rs python=3.8
$conda activate rs
$pip install -r requirements.txt
```


## Usage of BDD-OIA (2048)

BDD-OIA is a dataset of dashcams images for autonomous driving predictions, annotated with input-level objects (like bounding boxes of pedestrians, etc.) and concept-level entities (like "road is clear"). The original dataset can be found here: https://twizwei.github.io/bddoia_project/

I preprocessed the dataset with a pretrained Faster-RCNN on BDD-100k and with the first module in CBM-AUC (Sawada and Nakamura, IEEE (2022)), leading to embeddings of dimension 2048. These are reported in the zip ```bdd_2048.zip```. The original repo of CBM-AUC can be found here https://github.com/AISIN-TRC/CBM-AUC.


For usage, consider citing the original dataset creators and Sawada and Nakamura:

```
@InProceedings{xu2020cvpr,
author = {Xu, Yiran and Yang, Xiaoyin and Gong, Lihang and Lin, Hsuan-Chu and Wu, Tz-Ying and Li, Yunsheng and Vasconcelos, Nuno},
title = {Explainable Object-Induced Action Decision for Autonomous Vehicles},
booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}}

@ARTICLE{sawada2022cbm-auc,
  author={Sawada, Yoshihide and Nakamura, Keigo},
  journal={IEEE Access}, 
  title={Concept Bottleneck Model With Additional Unsupervised Concepts}, 
  year={2022},
  volume={10},
  number={},
  pages={41758-41765},
  doi={10.1109/ACCESS.2022.3167702}}
```

## Structure of the code

* I implemented XOR and MNIST-Addition on a single project folder, located at ```XOR_MNIST```. 

    * ``backbones`` contains the architecture of the NNs used.
    * ``datasets`` cointains the various versions of MNIST addition. If you want to add a dataset it has to be located here.
    * ``example`` is an independent folder containing all the experiments and setup for running XOR
    *  ``models`` contains all models used to benchmark the presence of RSs. Here, you can find DPL, SL, and LTN + recunstruction, but also a simple concept extractor (cext.py) and conditional VAEs (cvae.py)
    * ``utils`` contains the training loop, the losses, the metrics and (only wandb) loggers 
    * ``exp_best_args.py`` is where I collected all best hyper-parameters for MNIST-Addition and XOR.
    * you can use ``experiments.py`` to prepare a stack of experiments. If you run on a cluster, you can run ``server.py`` to access submitit and schedule a job array or use ``run_start.sh`` to run a single experiment. 


* ``BDD_OIA`` follows the design of Sawada and can be executed launching ``run_bdd.sh``. Hyperparameters are already set.


* args in ``utils.args.py``:
    * --dataset: choose the dataset
    * --task: addition/product/multiop
    * --model: which model you choose, remember to add rec at end if you want to add reconstruction penalty
    * --c_sup: percentage of concept supervision. If zero, then 0% of examples are supervise, if 1, then 100% of examples have concept supervision
    * --which_c: pass a list to specify which concepts you want to supervise, e.g. [1,2], will activate supervision for only concept 1 and 2
    * --joint: if included it will process both MNIST digits all together
    * --entropy: if included it will add the entropy penalty
    * --w_sl: weight for the Semantic Loss
    * --gamma: general weight for the mitigation strategy (this will multiply with other weights. My advice is to set it to 1)
    * --wrec, --beta, --w_h, --w_c: different weights for penalties (see also args description)

    * others are quite standard, consider using also:
        * --wandb: put here the name of your project, like 'i-dont-like-rss'
        * --checkin, --checkout: specify path were to load and to save checkpoints, respectivey
        * --validate: activate it to use the validation set (this is a switch from val to test)

## Issues report, bug fixes, and pull requests

For all kind of problems do not hesitate to contact me. If you have additional mitigation strategies that you want to include as for others to test, please send me a pull request. 

