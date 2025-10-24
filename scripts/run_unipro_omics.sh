# data_root_wsi="/data2/xyx/Pathology/Patches/TCGA__"
data_root_wsi="/path/to/feature/Patches/TCGA__"
pretrained_model="uni"
data_root_omics="/path/to/omics/TCGA-Omics/"
result_dir="/path/to/results/DisPro"

models="Coop_PathTrans_BioBert"
modal_setup="Omics"
studies="BLCA"
# studies="BLCA BRCA COADREAD GBMLGG STAD"
extra_suffix_org=""
# rates_wsi=(0 0.2 0.3)
# rates_omics=(0.6 0.4 0.3)
rates_wsi=(0.2)
rates_omics=(0.4)
LEN_RATES=${#rates_wsi[@]}

gpu_id=7
memory_thre=70

if [ ! -d "$result_dir" ]; then
    mkdir -p $result_dir
fi
log_dir=$result_dir/logs
if [ ! -d "$log_dir" ]; then
    mkdir -p $log_dir
fi


for model in $models
do  
    for i in $(seq 0 $(($LEN_RATES - 1))); do
        rate1=${rates_wsi[$i]}
        rate2=${rates_omics[$i]}
        echo "$(date): [Missing combination $i]: [$rate1 - $rate2]"


        MISSING_CONFIG_TRAIN="WSI:${rate1}_Omics:${rate2}"

        

        for study in $studies
        do  
            
            suffix="_${MISSING_CONFIG_TRAIN}_WSI"
            extra_suffix="${extra_suffix_org}${suffix}"
            log_file="${log_dir}/${study}_${modal_setup}_${model}${extra_suffix}_log.txt"
            if [ -f "$log_file" ]; then
                echo "$(date): ${study}_${modal_setup}_${model}${extra_suffix} already exists."
            fi
            echo "$(date): The log file will be saved at $log_file"
            echo "$(date): ${study}-${modal_setup}-${model}"
            ulimit -n 4096
            CUDA_VISIBLE_DEVICES=$gpu_id nohup python main.py --unipro \
                                                                --model $model \
                                                                --study $study \
                                                                --missing_config_train $MISSING_CONFIG_TRAIN \
                                                                --excel_file ./splits/csv_missing_cleaned/${study}/ \
                                                                --data_root_wsi ${data_root_wsi}${study}"/pt_files/"${pretrained_model} \
                                                                --data_root_omics ${data_root_omics}"TCGA-"${study}"/" \
                                                                --modal ${modal_setup} \
                                                                --result_dir ${result_dir} \
                                                                --num_epoch 30 \
                                                                --batch_size 1 >> $log_file 2>&1 &
            echo "=============================================================================================="
            sleep 50
        done
    done
done

echo "All jobs have been submitted."