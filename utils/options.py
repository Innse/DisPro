import argparse


def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description="configurations for response prediction")
    parser.add_argument("--excel_file", type=str, help="path to csv file")
    parser.add_argument("--study", type=str, help="which study")
    parser.add_argument("--modal", type=str, default="WSI", help="required modality")
    parser.add_argument("--signatures", type=str, default=None, help="path to signatures file (signatures.csv)")
    parser.add_argument("--sampling", type=int, default=0, help="sampling for mini-batch")

    # Checkpoint + Misc. Pathing Parameters
    parser.add_argument("--seed", type=int, default=1, help="random seed for reproducible experiment (default: 1)")
    parser.add_argument("--log_data", action="store_true", default=True, help="log data using tensorboard")
    parser.add_argument("--evaluate", action="store_true", dest="evaluate", help="evaluate model on test set")
    parser.add_argument("--resume", type=str, default="", metavar="PATH", help="path to latest checkpoint (default: none)")
    parser.add_argument("--tqdm", action="store_true", dest="tqdm", help="whether use tqdm")
    parser.add_argument("--OOM", type=int, default=0, help="Ramdomly sampling some patches to avoid OOM error")

    # General Model Parameters.
    parser.add_argument("--model", type=str, default="meanmil", help="type of model (default: meanmil)")

    # Optimizer Parameters + Survival Loss Function
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW", "RAdam", "PlainRAdam", "Lookahead"], default="Adam")
    parser.add_argument("--scheduler", type=str, choices=["None", "exp", "step", "plateau", "cosine"], default="cosine")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--num_epoch", type=int, default=20, help="maximum number of epochs to train (default: 20)")
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate (default: 0.0002)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--loss", type=str, default="nll_surv", help="slide-level classification loss function (default: ce)")
    parser.add_argument("--multi_lr", action="store_true", help="use different learning rate for different parts of the model")
    
    
    # Results Parameters
    parser.add_argument("--result_dir", default='/path/to/results/',
                        help='results directory')
    
    # Data Loading Parameters
    parser.add_argument("--data_root_wsi", type=str, help="path to wsi directory")
    parser.add_argument("--data_root_omics", type=str, help="path to omics directory")
    parser.add_argument("--data_root_report", type=str, help="path to report directory")
    parser.add_argument("--folds", type=int, default=5, help="number of folds for cross-validation")
    parser.add_argument("--k_start", type=int, default=-1, help="start fold for cross-validation")
    parser.add_argument("--k_end", type=int, default=-1, help="end fold for cross-validation")
    
    # Missing Data Parameters
    parser.add_argument("--unipro", action="store_true", help="flag for setting up unipro")
    parser.add_argument("--multipro", action="store_true", help="flag for setting up multipro")
    
    parser.add_argument("--missing_config_train", type=str, default=None, help="missing data configuration (dict) for training")
    parser.add_argument("--eval_settings", type=str, default="", help="evaluation settings")
    
    args = parser.parse_args()
    return args



def get_gene_config(args):
    gene_config = argparse.Namespace()
    gene_config.signatures = args.signatures
    gene_config.drop_rate = 0.25
    gene_config.in_embed_dim = 768
    gene_config.out_embed_dim = 768
    gene_config.pooler = "mean"
    gene_config.pool_method = "topj" # for MI-CLIP
    gene_config.topj = 256
    
    return gene_config

def get_wsi_config():
    wsi_config = argparse.Namespace()
    wsi_config.modal_enc_name = 'fc'
    wsi_config.in_embed_dim = 1024
    wsi_config.out_embed_dim = 768
    wsi_config.pool_method = 'topj'
    wsi_config.topj = 256
    
    return wsi_config

def get_prompt_config(modal=None):
    prompt_config = argparse.Namespace()
    if modal == 'WSI':
        prompt_config.max_len_ctx = 256
        prompt_config.ctx_init = "This is a pathology slide image from the patient with overall survival of"
    elif modal == 'Omics':
        prompt_config.max_len_ctx = 257
        prompt_config.ctx_init = "These are gene expression profiles from the patient with overall survival of"
    else:
        prompt_config.max_ctx = 255
        prompt_config.ctx_init = ""
        
    prompt_config.class_token_position = "end"
    classnames = ['dead: high risk',
                  'dead: mid-high risk',
                  'dead: mid-low risk',
                  'dead: low risk',
                  'alive: short observation',
                  'alive: mid-short observation',
                  'alive: mid-long observation',
                  'alive: long observation']
    prompt_config.classnames = classnames
    
    return prompt_config


def get_missing_config(missing_config_train):
    missing_config = argparse.Namespace()
    
    if missing_config_train is not None:
        missing_config.missing_config_train = {}
        modals_rates = missing_config_train.split("_")
        for modal_rate in modals_rates:
            modal, rate = modal_rate.split(":")
            missing_config.missing_config_train[modal] = float(rate)
    else:
        raise NotImplementedError("missing_config_train is not implemented.")
    
    suffix_map = {"WSI": "W", "Omics": "O"}
    suffix = "_".join([f"{suffix_map[modal]}{int(rate*100)}" for modal, rate in missing_config.missing_config_train.items()])
    missing_config.suffix = suffix
    
    return missing_config