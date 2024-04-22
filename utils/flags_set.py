import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Chemical Reaction Model Training")
    parser.add_argument(
        "--scaler_m",
        type=str,
        default="standard",
        help="Type of model to use",
    )
    parser.add_argument(
        "--scaler_t",
        type=str,
        default="standard",
        help="Type of model to use",
    )
    parser.add_argument(
        "--model_version",
        type=str,
        default="v1",
        choices=["v1", "v2", "v3", "v4", "v5"],
        help="Type of model to use",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["lstm", "gat", "gcn", "rgat", "rgcn", "mlp", "transformer", "mamba"],
        help="Type of model to use (lstm, mlp, transformer)",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=128, help="Size of the hidden layers"
    )
    parser.add_argument(
        "--pretrain_ratio", type=float, default=1.0, help="Training epoches"
    )
    parser.add_argument("--epoch", type=int, default=3000, help="Training epoches")
    parser.add_argument(
        "--dim_feedforward", type=int, default=128, help="Size of the FFN layers"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Training epoches")
    parser.add_argument("--num_head", type=int, default=2, help="Training epoches")
    parser.add_argument("--num_layers", type=int, default=4, help="Training epoches")
    parser.add_argument("--threshold", type=int, default=20, help="Training epoches")
    parser.add_argument(
        "--max_threshold", type=int, default=100, help="Training epoches"
    )

    parser.add_argument(
        "--num_threshold", type=int, default=100, help="Training epoches"
    )
    parser.add_argument("--num_classes", type=int, default=11, help="Training epoches")
    parser.add_argument(
        "--weak_threshold", type=float, default=0.1, help="Training epoches"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Training epoches"
    )

    parser.add_argument("--batch_size", type=int, default=64, help="Training epoches")
    parser.add_argument("--pretrain", action="store_true", default=False, help="")
    parser.add_argument("--average_feats", action="store_true", default=False, help="")
    parser.add_argument("--exp", type=str, default="exp01", help="Training epoches")
    parser.add_argument(
        "--feats", type=str, default="feats/matminer_feats.pkl", help="Training epoches"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None, help="Pretrained checkpoint"
    )
    parser.add_argument(
        "--save_path", type=str, default="results-0131", help="save path"
    )
    parser.add_argument("--raw_data", type=str, default="", help="")
    parser.add_argument("--weak_data", action="store_true", default=False, help="")
    parser.add_argument("--ext_data", action="store_true", default=False, help="")
    parser.add_argument("--use_ema", action="store_true", default=False, help="")
    parser.add_argument("--cls_token", action="store_true", default=False, help="")
    parser.add_argument(
        "--balanced_sample", action="store_true", default=False, help=""
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mask_target", action="store_true", default=False, help="")
    parser.add_argument(
        "--mask_target_rate", type=float, default=0.0, help="Training epoches"
    )
    parser.add_argument(
        "--expand_data", type=str, choices=["v1", "v2"], default=None, help=""
    )
    parser.add_argument("--debug", action="store_true", default=False, help="")
    parser.add_argument("--with_bn", action="store_true", default=False, help="")
    parser.add_argument(
        "--local_expand_data", action="store_true", default=False, help=""
    )
    parser.add_argument(
        "--scalers_path", type=str, default=None, help="Training epoches"
    )
    parser.add_argument("--input_size", type=int, default=2562, help="Random seed")
    parser.add_argument("--clip_input_size", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--save_scalers", type=str, default=None, help="Training epoches"
    )
    parser.add_argument(
        "--loss", type=str, choices=["mse", "rmse", "mae", "huber"], default="mse", help=""
    )
    parser.add_argument("--weak_dist", action="store_true", default=False, help="")
    parser.add_argument("--repeat_data", type=int, default=1, help="Random seed")
    parser.add_argument(
        "--product", type=str, default=None, help="Training epoches"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="Training epoches"
    )
    parser.add_argument("--frozen", type=int, default=0, help="")
    parser.add_argument("--load_failure", action="store_true", default=False, help="")
    parser.add_argument("--size_loss_ratio", type=float, default=1.0, help="")
    parser.add_argument("--dist_loss_ratio", type=float, default=0.0, help="")
    parser.add_argument("--cls_loss_ratio", type=float, default=0.0, help="")
    parser.add_argument("--binary_loss_ratio", type=float, default=0.0, help="")
    parser.add_argument("--without_kaiming_normal", action="store_true", default=False, help="")
    parser.add_argument("--not_trust_extra", action="store_true", default=False, help="")
    parser.add_argument("--without_product", action="store_true", default=False, help="")
    parser.add_argument(
        "--object_name", type=str, default=None, help="Training epoches"
    )
    parser.add_argument(
        "--train_target", type=str, choices=["v1", "v2"], default=None, help=""
    )
    parser.add_argument("--val_epoch", type=int, default=100, help="")
    parser.add_argument("--eval_only", action="store_true", default=False, help="")
    parser.add_argument(
        "--ignore_param", type=str, default="nothing", help="Training epoches"
    )
    parser.add_argument("--size_as_input", action="store_true", default=False, help="")
    parser.add_argument("--focal_loss", action="store_true", default=False, help="")
    
    # Parse arguments
    args = parser.parse_args()
    return args
