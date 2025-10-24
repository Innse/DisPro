import os
import time
os.environ["TOKENIZERS_PARALLELISM"] = "false"



from utils.options import parse_args
from utils.util import set_seed
from utils.loss import define_loss
from utils.optimizer import define_optimizer
from utils.scheduler import define_scheduler
from utils.util import CV_Meter

from torch.utils.data import DataLoader, SubsetRandomSampler

def main(args):
    # set random seed for reproduction
    set_seed(args.seed)
    # create results directory
    if args.evaluate:
        results_dir = args.resume
    else:
        # ********************************************************************************************************************
        missing_config = args.missing_config
        args.miss_suffix = missing_config.suffix
        result_file = "[{model}]-[{suffix}]-[{time}]".format(model=args.model,
                                                                suffix=args.miss_suffix,
                                                                time=time.strftime("%Y-%m-%d]-[%H-%M-%S"))
        results_dir = os.path.join(args.result_dir,
                args.modal,
                args.study,
                result_file
            )
    print("[checkpoint] results directory: ", results_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
            
    
    args.num_classes = 4
    # 5-fold cross validation
    meter = CV_Meter(fold=args.folds)
    if args.k_start == -1:
        args.k_start = 0
    if args.k_end == -1:
        args.k_end = args.folds
    # start 5-fold CV evaluation.
    for fold in range(args.k_start, args.k_end):
        # define dataset
        dataset = None
        excel_file = os.path.join(args.excel_file, f"{args.study}_fold_{fold}_{args.miss_suffix}.csv")
        if args.unipro:
            from datasets.TCGA_Dataset_Uni import TCGA_Dataset
            dataset = TCGA_Dataset(excel_file=excel_file, modal=args.modal, signatures="./datasets/pathway_signatures.csv",
                                    data_root_wsi=args.data_root_wsi, data_root_omics=args.data_root_omics)
        elif args.multipro:
            from datasets.TCGA_Dataset_Multi import TCGA_Dataset
            dataset = TCGA_Dataset(excel_file=excel_file, modal=args.modal, signatures="./datasets/pathway_signatures.csv",
                                    data_root_wsi=args.data_root_wsi, data_root_omics=args.data_root_omics)
        else:
            raise NotImplementedError("unipro or multipro is not set up.")
        # get split
        splits = dataset.splits
        dataloaders = {split: DataLoader(dataset, batch_size=1, sampler=SubsetRandomSampler(splits[split]), num_workers=4, pin_memory=True) for split in splits.keys()}
    
        # build model, criterion, optimizer, schedular
        #################################################
        # Unimodal: Gene
        if args.model == "Coop_PathTrans_BioBert":
            from models.Omics.Coop_PathTrans_BioBert.network import CoOp
            from models.Omics.Coop_PathTrans_BioBert.engine import Engine
            from utils.options import get_gene_config, get_prompt_config

            prompt_config = get_prompt_config(modal='Omics')
            gene_config = get_gene_config(args)
            gene_config.omics_size = dataset.omics_size
            gene_config.num_classes = args.num_classes

            args.lr = 1e-5
            args.num_epoch = 50
            model_dict = {"clsStrEnc_name": "dmis-lab/biobert-base-cased-v1.2",
                          "modal_enc_name": "PathTransMean", "prompt_config": prompt_config, "gene_config": gene_config}
            model = CoOp(**model_dict)
            engine = Engine(args, results_dir, fold)
        elif args.model == "Coop_WSI_BioBert":
            from models.WSI.Coop_WSI_BioBert.network import CoOp
            from models.WSI.Coop_WSI_BioBert.engine import Engine
            from utils.options import get_wsi_config, get_prompt_config

            prompt_config = get_prompt_config(modal='WSI')
            modal_config = get_wsi_config()

            model_dict = {"clsStrEnc_name": "dmis-lab/biobert-base-cased-v1.2",
                          "prompt_config": prompt_config, "modal_config": modal_config}
            model = CoOp(**model_dict)
            engine = Engine(args, results_dir, fold)
        elif args.model == "DisPro":
            from models.Incomplete.DisPro.network import Transformer
            from models.Incomplete.DisPro.engine import Engine
            from utils.options import get_gene_config, get_prompt_config, get_wsi_config
            from utils.util import load_uni_models_for_missing, loading_unipro_config

            
            missing_modal_config = loading_unipro_config()
            path_model_wsi, path_model_omics = load_uni_models_for_missing(fold, missing_modal_config, args)
            prompt_config_wsi = get_prompt_config('WSI')
            prompt_config_omics = get_prompt_config('Omics')
            # prompt_config = [prompt_config_wsi, prompt_config_omics]
            prompt_config = {'WSI': prompt_config_wsi, 
                             'Omics': prompt_config_omics}
            gene_config = get_gene_config(args)
            wsi_config = get_wsi_config()
            unis_config = {"path_model_wsi": path_model_wsi, 
                           "path_model_omics": path_model_omics,
                           "prompt_config": prompt_config,
                           "gene_config": gene_config,
                           "wsi_config": wsi_config}
            model_dict = {"unis_config": unis_config, "omic_sizes": dataset.omics_size, "encoder": "BioBERT",
                            "num_classes": args.num_classes, "max_length": 512, 
                            "n_WSI": dataset.path_size, "dim_token": 768, "fine_tune": False}
            model = Transformer(**model_dict)
            engine = Engine(args, results_dir, fold)
        
        else:
            raise NotImplementedError("model [{}] is not implemented".format(args.model))
        print("[model] trained model: ", args.model)
        criterion = define_loss(args)
        print("[model] loss function: ", args.loss)
        optimizer = define_optimizer(args, model)
        print("[model] optimizer: ", args.optimizer, "\t lr: ", args.lr, "\t weight_decay: ", args.weight_decay)
        scheduler = define_scheduler(args, optimizer)
        print("[model] scheduler: ", args.scheduler)
        # start training
        results = engine.learning(model, dataloaders, criterion, optimizer, scheduler)
        meter.updata(results)

    csv_path = os.path.join(results_dir, "results_{}.csv".format(args.model))
    meter.save(csv_path)


if __name__ == "__main__":
    start_time = time.strftime("[%Y-%m-%d]-[%H-%M-%S]")
    print(f"======================================= Start Training at {start_time} =======================================")

    args = parse_args()

    from utils.options import get_missing_config
    missing_config = get_missing_config(args.missing_config_train)
    args.missing_config = missing_config

    results = main(args)
    print("finished!")
