model=codet5
data_num=-1
for mask_type in "c" "cs" "s"
do
    python run.py \
        --do_adapter \
        --mask_type $mask_type \
        --cache_path=./cache_data/${data_num} \
        --data_num=${data_num} \
        --output_dir=./saved_models/${model}/${mask_type} \
        --model_type=codet5 \
        --model_name_or_path=../models/${model}/${mask_type} \
        --do_train \
        --do_test \
        --train_data_file=../dataset/train.txt \
        --eval_data_file=../dataset/valid.txt \
        --test_data_file=../dataset/test.txt \
        --epoch 2 \
        --block_size 400 \
        --train_batch_size 16 \
        --eval_batch_size 32 \
        --learning_rate 5e-5 \
        --max_grad_norm 1.0 \
        --evaluate_during_training \
        --warmup_steps 100 \
        --seed 123456  2>&1 | tee run_${model}_${mask_type}.log

    echo "mask_type:" $mask_type
    python ../evaluator/evaluator.py -a ../dataset/test.txt -p saved_models/${model}/${mask_type}/predictions.txt
done
