import argparse

def pretrain_arg_parser():
    parser = argparse.ArgumentParser(description="Argument parser for PiCME pretraining.")
    
    # Modality selection
    parser.add_argument(
        "--modalities", 
        type=str, 
        nargs="+", 
        help="List of modalities to use (required for all_pairs and ovo)"
    )
    parser.add_argument(
        "--modality_pairs", 
        type=str, 
        nargs=2, 
        help="Pair of modalities to use for single_pair training"
    )

    parser.add_argument('--anchor_weight', type=float, default=0.5, 
                    help='Weight for the Anchor Alignment loss (aligning singletons to paired neighbors).')
    
    # Training hyperparameters
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100, 
        help="Number of epochs"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4, 
        help="Learning rate"
    )
    parser.add_argument(
        "--optimizer", 
        type=str, 
        default="adam", 
        choices=["adam", "sgd"], 
        help="Optimizer"
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.01, 
        help="Weight decay"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.07, 
        help="Temperature for InfoNCE loss"
    )
    parser.add_argument(
        "--projection_dim", 
        type=int, 
        default=128, 
        help="Projection dimension"
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    
    # Wandb config
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        default="picme_pretraining", 
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_entity", 
        type=str, 
        default=None, 
        help="Wandb entity name"
    )
    parser.add_argument(
        "--wandb_name", 
        type=str, 
        default=None, 
        help="Wandb run name"
    )
    parser.add_argument(
        "--wandb_notes", 
        type=str, 
        default=None, 
        help="Wandb notes"
    )
    parser.add_argument(
        "--wandb_tags", 
        type=str, 
        nargs="+", 
        default=None, 
        help="Wandb tags"
    )
    
    parser.add_argument("--dataset_tag", type=str, default="", help="Tag for dataset type (e.g., 'missing' for missing modality datasets)")
    
    # Output paths
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./checkpoints", 
        help="Output directory for checkpoints"
    )

    parser.add_argument(
        "--pretraining_type", 
        type=str, 
        choices=["single_pair", "all_pairs", "ovo", "masked_global", "focal"], 
        required=True, 
        help="Type of pretraining to use"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=2.0,
        help="Gamma (focusing parameter) for Focal Loss. Higher values focus more on hard examples. Default is 2.0."
    )
    
    parser.add_argument(
    "--modality_to_drop", 
    type=str, 
    default=None, 
    help="Specify a modality to randomly drop during training for augmentation."
    )
    
    parser.add_argument(
        "--drop_probability", 
        type=float, 
        default=0.5, 
        help="The probability of dropping the specified modality for any given sample."
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on pretraining type
    if args.pretraining_type == "single_pair":
        if not args.modality_pairs or len(args.modality_pairs) != 2:
            parser.error("Must specify exactly two modalities for single_pair pretraining")
    else:  # all_pairs or ovo
        if not args.modalities or len(args.modalities) < 2:
            parser.error("Must specify at least two modalities for all_pairs or ovo pretraining")
    
    return args

def finetune_arg_parser():
    parser = argparse.ArgumentParser(
        description="Argument parser for multi-modal contrastive learning model."
    )

    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        required=True,
        help="List of data types (all strings)",
    )
    parser.add_argument(
        "--learning_rate", type=float, nargs="+", required=True, help="Learning rate(s)"
    )
    parser.add_argument(
        "--seed_category",
        type=str,
        choices=["single", "sweep"],
        required=True,
        help="Seed category (single or sweep)",
    )
    parser.add_argument(
        "--seed_number",
        type=int,
        required=False,
        help="Seed number (required if seed_category is single)",
    )
    parser.add_argument(
        "--batch_size", type=int, nargs="+", required=True, help="Batch size"
    )
    parser.add_argument(
        "--modality_lambdas",
        type=float,
        nargs="+",
        required=False,
        help="Lambda values computed from modality finetuning.",
    )
    parser.add_argument(
        "--fusion_method", type=str, help="Embedding fusion method for prediction"
    )
    parser.add_argument("--epochs", type=int, required=True, help="Batch size")
    parser.add_argument(
        "--objective",
        type=str,
        required=True,
        help="Metric to optimize (must be in metrics)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="List of metrics (all strings)",
    )
    parser.add_argument(
        "--wandb_project", type=str, required=True, help="Wandb project name"
    )

    parser.add_argument(
        "--wandb_name", type=str, required=True, help="Wandb run name"
    )
    
    parser.add_argument(
        "--task",
        type=str,
        choices=["mortality", "phenotyping"],
        required=True,
        help="Task (must be mortality or phenotyping)",
    )
    parser.add_argument(
        "--state_dict", type=str, required=True, help="Pretrained state dict"
    )
    parser.add_argument("--save_prefix", type=str, required=True, help="Save location")
    parser.add_argument("--freeze", action="store_true", help="Flag to freeze weights")
    parser.add_argument(
        "--weigh_loss", action="store_true", help="Flag to weight Cross Entropy Loss"
    )

    
    parser.add_argument(
        "--sequential_model", 
        action="store_true", 
        help="Use the sequential Belief Update Transformer (BUT) model."
    )

    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before optimizer update."
    )
    
    parser.add_argument(
        "--llm_model_name", 
        type=str, 
        default="mistralai/Mistral-7B-v0.1", 
        help="Hugging Face model ID to use as the BUT backbone."
    )
    parser.add_argument(
        "--input_embedding_dim", 
        type=int, 
        default=256, 
        help="Dimension of the pre-trained modality embeddings from Phase 1."
    )
    
    # --- LoRA (PEFT) Arguments ---
    parser.add_argument(
        "--use_lora", 
        action="store_true", 
        help="Enable LoRA (PEFT) for fine-tuning."
    )
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=8, 
        help="LoRA r parameter (rank)."
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=16, 
        help="LoRA alpha parameter."
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.1, 
        help="LoRA dropout."
    )

    parser.add_argument(
        "--use_imputation", 
        action="store_true", 
        help="Enable KNN imputation for missing modalities in static models."
    )
    parser.add_argument(
        "--embedding_bank_path", 
        type=str, 
        default=None, 
        help="Path to the pre-computed .pkl embedding bank."
    )

    # --- Uncertainty Head Argument ---
    parser.add_argument(
        "--use_evidential_head", 
        action="store_true", 
        help="Use an Evidential Deep Learning head for uncertainty."
    )
    
    # --- NEW ARGUMENT ---
    parser.add_argument(
        "--annealing_steps",
        type=int,
        default=10, 
        help="Number of annealing steps for evidential loss."
    )

    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')

    parser.add_argument(
        "--compute_embeddings_on_fly", 
        action="store_true", 
        help="If set, loads raw data and encodes it during training loop instead of loading precomputed tensors."
    )
    
    parser.add_argument(
        "--filter_incomplete", 
        action="store_true", 
        help="If set (and computing on fly), skips patients who do not have ALL 5 modalities present."
    )
    
    # Validate that the optimize metric is in metrics
    args = parser.parse_args()
    if args.objective not in args.metrics:
        parser.error("The objective metric must be in the list of metrics")

    # Validate seed number if seed_category is single
    if args.seed_category == "single" and args.seed_number is None:
        parser.error("Seed number must be specified if seed category is single")

    if args.model_name not in ["ovo", "pairs", "single_pair", "baseline"]:
        parser.error(
            "The model name must be one of ['ovo', 'pairs', 'single_pair']."
        )

    supported_metrics = ["hamming", "auroc", "auprc", "f1"]
    for metric in args.metrics:
        if metric not in supported_metrics:
            parser.error(
                f"The model name must be one of {supported_metrics}. Loss and accuracy are automatically calculated."
            )

    supported_fusion = ["concatenation", "vanilla_lstm", "modality_lstm"]
    if args.fusion_method not in supported_fusion:
        parser.error(f"The fusion method must be one of {supported_fusion}. ")

    if args.fusion_method == "modality_lstm":
        if args.modality_lambdas is None:
            parser.error(
                f"Modality lambdas must be provided for modality_lstm fusion method."
            )
        elif len(args.modalities) != len(args.modality_lambdas):
            parser.error(
                f"Number of modality lambdas != number of modalities provided."
            )
        elif sum(args.modality_lambdas) - 1 >= 0.0005:
            parser.error(f"Modality lambda values must sum to around 1.")

    if "hamming" in args.metrics and args.task == "mortality":
        parser.error("Hamming loss only supported for phenotyping.")

    return args
    

def evaluation_arg_parser():
    parser = argparse.ArgumentParser(
        description="Argument parser for multi-modal contrastive learning model."
    )

    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        required=True,
        help="List of data types (all strings)",
    )
    parser.add_argument(
        "--seed_category",
        type=str,
        choices=["single", "sweep"],
        required=True,
        help="Seed category (single or sweep)",
    )
    parser.add_argument(
        "--seed_number",
        type=int,
        required=False,
        help="Seed number (required if seed_category is single)",
    )
    parser.add_argument(
        "--batch_size", type=int, required=True, help="Batch size"
    )
    parser.add_argument(
        "--modality_lambdas",
        type=float,
        nargs="+",
        required=False,
        help="Lambda values computed from modality finetuning.",
    )
    parser.add_argument(
        "--fusion_method", type=str, help="Embedding fusion method for prediction"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help="List of metrics (all strings)",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["mortality", "phenotyping"],
        required=True,
        help="Task (must be mortality or phenotyping)",
    )
    parser.add_argument(
        "--state_dict", type=str, required=True, help="Pretrained state dict"
    )
    parser.add_argument(
        "--eval_name",
        type=str,
        required=True,
        help="Name to save the evaluation.",
    )
    parser.add_argument(
        "--classifier_state_dict",
        type=str,
        required=True,
        help="Path to the classifier state dict",
    )
    parser.add_argument("--use_test", type=str, default="val", help="Whether to use test data (`test`) or validation data (`val`)")
    parser.add_argument("--save_prefix", type=str, required=True, help="Save location")

    parser.add_argument("--evaluate_with_missing", action="store_true", help="Enable evaluation on datasets with missing modalities.")
    parser.add_argument("--dataset_tag", type=str, default="", help="Tag for dataset type (e.g., 'missing' for missing modality datasets)")
    parser.add_argument("--embedding_bank_path", type=str, help="Path to pre-computed embedding bank for imputation.")


    # Validate that the optimize metric is in metrics
    args = parser.parse_args()
    
    if args.evaluate_with_missing and not args.embedding_bank_path:
        parser.error("--embedding_bank_path is required when --evaluate_with_missing is set.")
        
    # Validate seed number if seed_category is single
    if args.seed_category == "single" and args.seed_number is None:
        parser.error("Seed number must be specified if seed category is single")

    if args.model_name not in ["ovo", "pairs", "single_pair", "baseline"]:
        parser.error(
            "The model name must be one of ['ovo', 'pairs', 'single_pair']."
        )

    supported_metrics = ["hamming", "auroc", "auprc", "f1"]
    for metric in args.metrics:
        if metric not in supported_metrics:
            parser.error(
                f"The model name must be one of {supported_metrics}. Loss and accuracy are automatically calculated."
            )

    supported_fusion = ["concatenation", "vanilla_lstm", "modality_lstm"]
    if args.fusion_method not in supported_fusion:
        parser.error(f"The fusion method must be one of {supported_fusion}. ")

    if args.fusion_method == "modality_lstm":
        if args.modality_lambdas is None:
            parser.error(
                f"Modality lambdas must be provided for modality_lstm fusion method."
            )
        elif len(args.modalities) != len(args.modality_lambdas):
            parser.error(
                f"Number of modality lambdas != number of modalities provided."
            )
        elif sum(args.modality_lambdas) - 1 >= 0.0005:
            parser.error(f"Modality lambda values must sum to around 1.")

    if "hamming" in args.metrics and args.task == "mortality":
        parser.error("Hamming loss only supported for phenotyping.")

    return args
