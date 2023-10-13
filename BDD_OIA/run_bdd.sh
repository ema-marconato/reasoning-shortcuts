python main_bdd.py --train --h_type fcc --epochs 5 \
--batch_size 512 --nconcepts 30 --nconcepts_labeled 21 --h_sparsity 7 --opt adam --lr 0.005 \
--weight_decay 0.00004 --theta_reg_lambda 0.001 --obj bce \
--model_name dpl_auc --h_labeled_param 0.01 --w_entropy 1 --seed 0
