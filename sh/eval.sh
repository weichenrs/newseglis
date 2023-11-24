# CUDA_VISIBLE_DEVICES=0,1 PORT=22222 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa.py 2 --resume --amp \
#                         --work-dir work_dirs/test/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa

# PORT=22222 CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa_bn.py 2 --resume --amp \
#                         --work-dir work_dirs/exp_1122/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa_bn

# PORT=22222 CUDA_VISIBLE_DEVICES=0,1 tools/dist_train_ds.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa_bn.py 2 --resume --amp \
#                         --work-dir work_dirs/exp_1122/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_fa_bn


# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh configs/vit/vit_vit-b16_mln_upernet_2xb2-80k_fbp-2048x2048_ds_test.py \
#                         work_dirs/exp_1116/vit_vit-b16_mln_upernet_2xb2-80k_fbp-2048x2048_ds_test/best_mIoU_iter_40000.pth 2 \
#                         --out show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-2048x2048_ds_test/iter_40000_conb


# CUDA_VISIBLE_DEVICES=2,3 tools/dist_test.sh work_dirs/exp_1117/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_80k/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test.py \
#                         work_dirs/exp_1117/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_80k/best_mIoU_iter_80000.pth 2 \
#                         --out show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_80k/best_mIoU_iter_80000

# python tools/eval_conb.py   --output_path 'show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_80k/result_best_mIoU_iter_80000.csv' \
#                             --label_path '/media/dell/data1/cw/data/Five-Billion-Pixels/ori/Annotation__index/test' \
#                             --pred_path 'show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_80k/best_mIoU_iter_80000' \
#                             --conb_path 'show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_80k/best_mIoU_iter_80000_conb' \
#                             --suffix '.png' \
#                             --sizeImg 1024 \


# CUDA_VISIBLE_DEVICES=2,3 tools/dist_test.sh \
#                         work_dirs/exp_1114/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test.py \
#                         work_dirs/exp_1114/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test/best_mIoU_iter_40000.pth \
#                         2 \
#                         --out show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_40k/best_mIoU_iter_40000

# python tools/eval_conb.py   --label_path '/media/dell/data1/cw/data/Five-Billion-Pixels/ori/Annotation__index/test' \
#                             --output_path 'show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_40k/result_best_mIoU_iter_40000.csv' \
#                             --pred_path 'show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_40k/best_mIoU_iter_40000' \
#                             --conb_path 'show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ds_test_40k/best_mIoU_iter_40000_conb' \
#                             --suffix '.png' \
#                             --sizeImg 1024 \


# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh \
#                         work_dirs/before_1114/old/vit_vit-b16_mln_upernet_2xb6-80k_fbp-512x512/vit_vit-b16_mln_upernet_2xb6-80k_fbp-512x512.py \
#                         work_dirs/before_1114/old/vit_vit-b16_mln_upernet_2xb6-80k_fbp-512x512/best_mIoU_iter_65000.pth \
#                         2 \
#                         --out show_dirs/test1123/vit_vit-b16_mln_upernet_2xb6-80k_fbp-512x512_80k/best_mIoU_iter_65000

# python tools/eval_conb.py   --label_path /media/dell/data1/cw/data/Five-Billion-Pixels/ori/Annotation__index/test \
#                             --pred_path show_dirs/test1123/vit_vit-b16_mln_upernet_2xb6-80k_fbp-512x512_80k/best_mIoU_iter_65000 \
#                             --sizeImg 512 \



# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh \
#                         work_dirs/before_1114/old/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py \
#                         work_dirs/before_1114/old/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_50000.pth \
#                         2 \
#                         --out show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_50000

# python tools/eval_conb.py   --label_path /media/dell/data1/cw/data/Five-Billion-Pixels/ori/Annotation__index/test \
#                             --pred_path show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_50000 \
#                             --sizeImg 1024 \

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh \
#                         work_dirs/before_1114/old/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py \
#                         work_dirs/before_1114/old/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_50000.pth \
#                         2 \
#                         --out show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_50000

# python tools/eval_conb.py   --label_path /media/dell/data1/cw/data/Five-Billion-Pixels/ori/Annotation__index/test \
#                             --pred_path show_dirs/test1123/vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024/best_mIoU_iter_50000 \
#                             --sizeImg 1024 \

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh \
#                         work_dirs/before_1114/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ok/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py \
#                         work_dirs/before_1114/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ok/best_mIoU_iter_35000.pth \
#                         2 \
#                         --out show_dirs/test1123/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ok/best_mIoU_iter_35000

# python tools/eval_conb.py   --label_path /media/dell/data1/cw/data/Five-Billion-Pixels/ori/Annotation__index/test \
#                             --pred_path show_dirs/test1123/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_ok/best_mIoU_iter_35000 \
#                             --sizeImg 1024 \

# CUDA_VISIBLE_DEVICES=0,1 tools/dist_test.sh \
#                         work_dirs/before_1114/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_withcp/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024.py \
#                         work_dirs/before_1114/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_withcp/best_mIoU_iter_40000.pth \
#                         2 \
#                         --out show_dirs/test1123/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_withcp/best_mIoU_iter_40000

# python tools/eval_conb.py   --label_path /media/dell/data1/cw/data/Five-Billion-Pixels/ori/Annotation__index/test \
#                             --pred_path show_dirs/test1123/my_vit_vit-b16_mln_upernet_2xb2-80k_fbp-1024x1024_withcp/best_mIoU_iter_40000 \
#                             --sizeImg 1024 \

                            