function run_explanation(){
    model_name=$1
    explainer=$2
    baseline_token=$3
    arch_int_topk=$4
    cross_merge=$5
    short=$6
    script="python explainers/save_explanations.py --batch_size 32 --model_name $model_name --mode test \
            --explainer $explainer --baseline_token $baseline_token --arch_int_topk $arch_int_topk"
    if [ $cross_merge == 1 ]; then
        script="$script --do_cross_merge"
    fi
    if [ $short == 1 ]; then
        script="$script --data_root data/e-SNLI/esnli_test_processed_1k.csv"
    fi
    echo the script is:
    echo $script

    eval $script
}
############################ FREE AREA #########################################
(trap 'kill 0' SIGINT;
run_explanation bert-base arch_pair '[MASK]' 5 0 1 &
run_explanation bert-base cross_arch_pair '[MASK]' 5 0 1 &

run_explanation bert-base arch '[MASK]' 5 1 1 &
run_explanation bert-base arch '[MASK]' 10 1 1

run_explanation bert-base cross_arch '[MASK]' 5 1 1 &
run_explanation bert-base cross_arch '[MASK]' 10 1 1 &

run_explanation bert-base arch '[MASK]' 5 0 1 &
run_explanation bert-base cross_arch '[MASK]' 5 0 1
)