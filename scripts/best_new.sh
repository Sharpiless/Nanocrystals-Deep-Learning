CUDA_VISIBLE_DEVICES=0 python train_new.py \
    --raw_data data/0309_classification_and_reg-raw-v3.csv \
    --model_type transformer --epoch 100 \
    --model_version v4 --cls_token \
    --dim_feedforward 256 --hidden_size 256 \
    --num_layers 8 --num_head 4 --lr 0.001 \
    --feats feats/unioml_feats_20240309_1x_ours_1x_mof.pkl \
    --num_threshold 300 --pretrain_ratio 0.8 \
    --batch_size 1024 --threshold 20 --weak_threshold 0.2 \
    --weak_data --max_threshold 10000 \
    --seed 0 --scaler_m robust --scaler_t robust \
    --save_path results-0415-best \
    --exp default-Cs2NaBiCl6-pretrain-0.1k-10000 --input_size 514 \
    --without_product \
    --object_name Cs2NaBiCl6 --cls_loss_ratio 1.0 --val_epoch 20 --load_failure \
    --local_expand_data --not_trust_extra \
    --size_loss_ratio 0 \
    --scalers_path scalers/scalers-10000-cls-robust.pkl --dropout 0.0 --loss huber

CUDA_VISIBLE_DEVICES=0 python train_new.py \
    --raw_data data/0309_classification_and_reg-raw-v3.csv \
    --model_type transformer --epoch 100 \
    --model_version v4 --cls_token \
    --dim_feedforward 256 --hidden_size 256 \
    --num_layers 8 --num_head 4 --lr 0.001 \
    --feats feats/unioml_feats_20240309_1x_ours_1x_mof.pkl \
    --num_threshold 300 --pretrain_ratio 0.8 \
    --batch_size 256 --threshold 20 --weak_threshold 0.2 \
    --weak_data --max_threshold 20 \
    --seed 0 --scaler_m robust --scaler_t robust \
    --save_path results-0415-best \
    --exp default-Cs2NaBiCl6-finetune-0.2k-F-default-Cs2NaBiCl6-pretrain-0.1k-10000 --input_size 514 \
    --without_product \
    --object_name Cs2NaBiCl6 --cls_loss_ratio 1.0 --load_failure \
    --pretrained results-0415-best/results-default-Cs2NaBiCl6-pretrain-0.1k-10000/latest.pth \
    --scalers_path scalers/scalers-10000-cls-robust.pkl --val_epoch 1

CUDA_VISIBLE_DEVICES=0 python train_new.py \
    --raw_data data/0309_classification_and_reg-raw-v3.csv \
    --model_type transformer --epoch 300 \
    --model_version v4 --cls_token \
    --dim_feedforward 256 --hidden_size 256 \
    --num_layers 8 --num_head 4 --lr 0.001 \
    --feats feats/unioml_feats_20240309_1x_ours_1x_mof.pkl \
    --num_threshold 300 --pretrain_ratio 0.8 \
    --batch_size 1024 --threshold 20 --weak_threshold 0.15 \
    --weak_data --max_threshold 10000 \
    --seed 0 --scaler_m robust --scaler_t robust \
    --save_path results-0415-best \
    --exp pretrain-Ni-300 --input_size 514 \
    --without_product \
    --object_name Ni --cls_loss_ratio 5.0 --val_epoch 20 --load_failure \
    --local_expand_data --not_trust_extra \
    --size_loss_ratio 0 \
    --scalers_path scalers/scalers-10000-cls-robust.pkl --loss huber

CUDA_VISIBLE_DEVICES=0 python train_new.py \
    --raw_data data/0309_classification_and_reg-raw-v3.csv \
    --model_type transformer --epoch 300 \
    --model_version v4 --cls_token \
    --dim_feedforward 256 --hidden_size 256 \
    --num_layers 8 --num_head 4 --lr 0.001 \
    --feats feats/unioml_feats_20240309_1x_ours_1x_mof.pkl \
    --num_threshold 300 --pretrain_ratio 0.8 \
    --batch_size 256 --threshold 20 --weak_threshold 0.15 \
    --weak_data --max_threshold 20 \
    --seed 0 --scaler_m robust --scaler_t robust \
    --save_path results-0415-best \
    --exp finetune-300-pretrain-Ni-300 --input_size 514 \
    --without_product \
    --object_name Ni --cls_loss_ratio 5.0 --load_failure \
    --pretrained results-0415-best/results-pretrain-Ni-300/latest.pth \
    --scalers_path scalers/scalers-10000-cls-robust.pkl --val_epoch 1

CUDA_VISIBLE_DEVICES=0 python train_new.py \
    --raw_data data/0309_classification_and_reg-raw-v3.csv \
    --model_type transformer --epoch 100 \
    --model_version v4 --cls_token \
    --dim_feedforward 256 --hidden_size 256 \
    --num_layers 8 --num_head 4 --lr 0.001 \
    --feats feats/unioml_feats_20240309_1x_ours_1x_mof.pkl \
    --num_threshold 300 --pretrain_ratio 0.8 \
    --batch_size 1024 --threshold 20 --weak_threshold 0.2 \
    --weak_data --max_threshold 10000 \
    --seed 0 --scaler_m robust --scaler_t robust \
    --save_path results-0415-best \
    --exp pretrain-d0.3-PbSe-100 --input_size 514 \
    --without_product \
    --object_name PbSe --cls_loss_ratio 1.0 --val_epoch 20 --load_failure \
    --local_expand_data --not_trust_extra \
    --size_loss_ratio 0 \
    --scalers_path scalers/scalers-10000-cls-robust.pkl --dropout 0.3 --loss huber

CUDA_VISIBLE_DEVICES=0 python train_new.py \
    --raw_data data/0309_classification_and_reg-raw-v3.csv \
    --model_type transformer --epoch 500 \
    --model_version v4 --cls_token \
    --dim_feedforward 256 --hidden_size 256 \
    --num_layers 8 --num_head 4 --lr 0.001 \
    --feats feats/unioml_feats_20240309_1x_ours_1x_mof.pkl \
    --num_threshold 300 --pretrain_ratio 0.8 \
    --batch_size 256 --threshold 20 --weak_threshold 0.2 \
    --weak_data --max_threshold 20 \
    --seed 0 --scaler_m robust --scaler_t robust \
    --save_path results-0415-best \
    --exp finetune-500-pretrain-d0.3-PbSe-100 --input_size 514 \
    --without_product \
    --object_name PbSe --cls_loss_ratio 1.0 --load_failure \
    --pretrained results-0415-best/results-pretrain-d0.3-PbSe-100/latest.pth \
    --scalers_path scalers/scalers-10000-cls-robust.pkl --val_epoch 1


CUDA_VISIBLE_DEVICES=0 python train_new.py \
    --raw_data data/0309_classification_and_reg-raw-v3.csv \
    --model_type transformer --epoch 300 \
    --model_version v4 --cls_token \
    --dim_feedforward 256 --hidden_size 256 \
    --num_layers 8 --num_head 4 --lr 0.001 \
    --feats feats/unioml_feats_20240309_1x_ours_1x_mof.pkl \
    --num_threshold 300 --pretrain_ratio 0.8 \
    --batch_size 1024 --threshold 20 --weak_threshold 0.15 \
    --weak_data --max_threshold 10000 \
    --seed 0 --scaler_m robust --scaler_t robust \
    --save_path results-0415-best \
    --exp pretrain-Ag2S-s0.1-300 --input_size 514 \
    --without_product \
    --object_name Ag2S --cls_loss_ratio 5.0 --val_epoch 20 --load_failure \
    --local_expand_data --not_trust_extra \
    --size_loss_ratio 0.1 \
    --scalers_path scalers/scalers-10000-cls-robust.pkl --loss huber

CUDA_VISIBLE_DEVICES=0 python train_new.py \
    --raw_data data/0309_classification_and_reg-raw-v3.csv \
    --model_type transformer --epoch 200 \
    --model_version v4 --cls_token \
    --dim_feedforward 256 --hidden_size 256 \
    --num_layers 8 --num_head 4 --lr 0.001 \
    --feats feats/unioml_feats_20240309_1x_ours_1x_mof.pkl \
    --num_threshold 300 --pretrain_ratio 0.8 \
    --batch_size 256 --threshold 20 --weak_threshold 0.15 \
    --weak_data --max_threshold 20 \
    --seed 0 --scaler_m robust --scaler_t robust \
    --save_path results-0415-best \
    --exp finetune-200-pretrain-Ag2S-s0.1-300 --input_size 514 \
    --without_product \
    --object_name Ag2S --cls_loss_ratio 5.0 --load_failure \
    --pretrained results-0415-best/results-pretrain-Ag2S-s0.1-300/latest.pth \
    --scalers_path scalers/scalers-10000-cls-robust.pkl --val_epoch 1