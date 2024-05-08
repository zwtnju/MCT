model=codet5
data_num=-1
for mask_type in "c" "cs" "s"
do
    python run.py \
        --do_adapter \
        --mask_type $mask_type \
        --cache_path=./cache_data/${data_num} \
        --data_num=${data_num} \
        --output_dir=./saved_models/${model}/${mask_type}/${data_num}/adapter \
        --model_type=codet5 \
        --model_name_or_path=../models/${model}/adapter/${mask_type} \
        --do_train \
        --do_test \
        --train_data_file=../dataset/train.jsonl \
        --eval_data_file=../dataset/valid.jsonl \
        --test_data_file=../dataset/test.jsonl \
        --epoch 10 \
        --block_size 400 \
        --train_batch_size 64 \
        --eval_batch_size 64 \
        --learning_rate 2e-5 \
        --max_grad_norm 1.0 \
        --evaluate_during_training \
        --warmup_steps 1000 \
        --seed 123456  2>&1 | tee ${model}_${mask_type}_${data_num}_adapter.log
    echo "mask_type:" $mask_type
    python ../evaluator/evaluator.py -a ../dataset/test.jsonl -p saved_models/${model}/${mask_type}/${data_num}/adapter/predictions.txt
done
