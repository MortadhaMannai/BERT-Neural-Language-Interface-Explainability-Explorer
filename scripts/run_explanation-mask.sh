function run_explanation(){
    model_name=$1
    explainer=$2
    interaction_order=$3
    mask_n=$4
    mask_p=$5
    buildup_p=$6
    no_correction=$7
    baseline_token=$8
    short=$9
    do_buildup=${10}
    script="python explainers/save_explanations.py --batch_size 256 --model_name $model_name --mode test \
            --explainer $explainer --baseline_token $baseline_token \
            --interaction_order $interaction_order --mask_n $mask_n --mask_p $mask_p --buildup_p $buildup_p"
    if [ $no_correction == 1 ]; then
        script="$script --no_correction"
    fi
    if [ $do_buildup == 1 ]; then
        script="$script --do_buildup"
    fi
    if [ $short == 1 ]; then
        script="$script --data_root data/e-SNLI/esnli_test_processed_1k.csv"
    fi
    # echo "the script is:"
    echo $script

    # eval $script
}
############################ FREE AREA #########################################
N=5000
P=0.5
SHORT=1
(
CUDA_VISIBLE_DEVICES=0 run_explanation bert-base mask_explain 1 $N $P 0.3 1 'attention+[MASK]' $SHORT 0 &
CUDA_VISIBLE_DEVICES=1 run_explanation bert-base mask_explain 2 $N $P 0.3 1 'attention+[MASK]' $SHORT 0 &
CUDA_VISIBLE_DEVICES=2 run_explanation bert-base mask_explain 3 $N $P 0.3 1 'attention+[MASK]' $SHORT 0 &
CUDA_VISIBLE_DEVICES=2 run_explanation bert-base mask_explain 4 $N $P 0.3 1 'attention+[MASK]' $SHORT 1 &
CUDA_VISIBLE_DEVICES=0 run_explanation bert-base mask_explain 5 $N $P 0.3 0 'attention+[MASK]' $SHORT 1 &
CUDA_VISIBLE_DEVICES=1 run_explanation bert-base mask_explain 6 $N $P 0.3 0 'attention+[MASK]' $SHORT 1 &
) | xargs --max-procs=4 -L 1 -I {} sh -c "{}"
