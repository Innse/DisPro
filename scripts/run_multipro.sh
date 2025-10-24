data_root_wsi="/path/to/feature/Patches/TCGA__"
pretrained_model="uni"
data_root_omics="/path/to/omics/TCGA-Omics/"
result_dir="/path/to/results/DisPro"

models="DisPro"
studies="BLCA"

############################################################################################################
signature="./datasets/pathway_signatures.csv"

############################################################################################################
extra_suffix_org=""


############################################################################################################
multi_lr=true
# multi_lr=false

epochs=50
############################################################################################################
# rates_wsi=(0 0.2 0.3 0.4 0.6)
# rates_omics=(0.6 0.4 0.3 0.2 0)
# rates_wsi=(0 0.2 0.3 0.6)
# rates_omics=(0.6 0.4 0.3 0)
rates_wsi=(0.2)
rates_omics=(0.4)

LEN_RATES=${#rates_wsi[@]}

############################################################################################################
EVAL_SET="wsi_omics_wsi-omics"
# EVAL_SET="omics_wsi-omics"

############################################################################################################
modals_setup="WSI_Omics"
gpu_id=7


if [ ! -d "$result_dir" ]; then
    mkdir -p $result_dir
fi
log_dir=$result_dir/logs
if [ ! -d "$log_dir" ]; then
    mkdir -p $log_dir
fi

echo "$(date): All studies to be run: $studies"
echo "$(date): All models to be run: $models"

for model in $models
do  
    for i in $(seq 0 $(($LEN_RATES - 1))); do
        rate1=${rates_wsi[$i]}
        rate2=${rates_omics[$i]}
        echo "$(date): [Missing combination $i]: [$rate1 - $rate2]"


        MISSING_CONFIG_TRAIN="WSI:${rate1}_Omics:${rate2}"
        suffix="_${MISSING_CONFIG_TRAIN}"
        extra_suffix="${extra_suffix_org}${suffix}"

        
        for study in $studies
        do  
            log_file="${log_dir}/${study}_${modals_setup}_${model}${extra_suffix}_log.txt"
            if [ -f "$log_file" ]; then
                echo "$(date): ${study}_${modals_setup}_${model}${extra_suffix} already exists."
            fi
            echo "$(date): The log file will be saved at $log_file"
            echo "$(date): Checking the GPU memory utilization in $GPUs ..."

            if [[ $signature == *"pathway"* ]]; then
                echo "$(date): Use ${signature} as signatures"
                ulimit -n 4096
                
                CUDA_VISIBLE_DEVICES=$gpu_id nohup python main.py  --multipro --multi_lr \
                                                                        --model $model \
                                                                        --study $study \
                                                                        --missing_config_train $MISSING_CONFIG_TRAIN \
                                                                        --excel_file ./splits/csv_missing_cleaned/${study}/ \
                                                                        --data_root_wsi ${data_root_wsi}${study}"/pt_files/"${pretrained_model} \
                                                                        --data_root_omics ${data_root_omics}"TCGA-"${study}"/" \
                                                                        --modal ${modals_setup} --eval_settings $EVAL_SET \
                                                                        --result_dir ${result_dir} \
                                                                        --signatures $signature \
                                                                        --num_epoch $epochs >> $log_file 2>&1 &
            
            fi
            echo "$(date): ======================================================"
            sleep 60
        done
    done
done

echo "All jobs have been submitted."