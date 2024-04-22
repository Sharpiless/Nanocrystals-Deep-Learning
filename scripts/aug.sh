CUDA_VISIBLE_DEVICES=0 python data_aug.py \
    --raw_data data/0309_classification_and_reg-raw-v3.csv \
    --feats feats/unimol_feats_20240309_5x.pkl \
    --num_threshold 300 \
    --threshold 20 \
    --weak_data --max_threshold 10000 \
    --seed 0 --scaler_m robust --scaler_t robust \
    --expand_data v2 --save_path extra_data_0313 \