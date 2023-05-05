model=codebert
for mask_type in "base" "i" "ic" "is" "ics" \
                 "c" "cv" "cf" "cs" "cvf" \
                 "cvs" "cfs" "cvfs" "v" "vf" \
                 "vs" "vfs" "f" "fs" "s"
do
    python run.py \
        --output_dir=./saved_models/${model}/${mask_type} \
        --model_type=roberta \
        --model_name_or_path=../models/${model}/${mask_type} \
        --do_train \
        --do_test \
        --train_data_file=../dataset/train.jsonl \
        --eval_data_file=../dataset/valid.jsonl \
        --test_data_file=../dataset/test.jsonl \
        --epoch 5 \
        --block_size 400 \
        --train_batch_size 32 \
        --eval_batch_size 64 \
        --learning_rate 2e-5 \
        --max_grad_norm 1.0 \
        --evaluate_during_training \
        --warmup_steps 1000 \
        --seed 123456  2>&1 | tee ${model}_${mask_type}.log

    python ../evaluator/evaluator.py -a ../dataset/test.jsonl -p saved_models/${model}/${mask_type}/predictions.txt
done