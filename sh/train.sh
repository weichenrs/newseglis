################################################ DONE ################################################

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa.py 2 --resume --amp \
#                          --work-dir work_dirs/exp_1113/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa_1113

# CUDA_VISIBLE_DEVICES=2,3 PORT=12000 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test.py 2 --resume \
#                         --work-dir work_dirs/exp_1114/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test


################################################ OKOK ################################################

CUDA_VISIBLE_DEVICES=2,3 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa.py 2 --resume --amp \
                        --work-dir work_dirs/exp_1120/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa

################################################ DOIN ################################################

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa.py 2 --resume --amp \
#                         --work-dir work_dirs/exp_1115/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa.py 2 --resume --amp \
#                         --work-dir work_dirs/exp_1115/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa_nopsp

################################################ TODO ################################################

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa.py 2 --resume \
#                         --work-dir work_dirs/exp_1115/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_train.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa.py 2 --resume --amp \
#                          --work-dir work_dirs/exp_1113/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_fa_1113