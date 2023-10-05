function run_explanation(){
    model_name=$1
    explainer=$2
    mask_n=$3
    baseline_token=$4
    short=$5
    script="python explainers/save_explanations.py --batch_size 64 --model_name $model_name --mode test \
            --explainer $explainer --baseline_token $baseline_token --mask_n $mask_n"
    if [ $short == 1 ]; then
        script="$script --data_root data/e-SNLI/esnli_test_processed_1k.csv"
    fi
    # echo "the script is:"
    # echo $script

    eval $script
}
############################ FREE AREA #########################################
N=5000
SHORT=1

CUDA_VISIBLE_DEVICES=1 run_explanation bert-base lime $N '[MASK]' $SHORT
# CUDA_VISIBLE_DEVICES=1 run_explanation bert-base select_all $N '[MASK]' $SHORT
