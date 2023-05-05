model=codebert
for mask_type in "base" "i" "ic" "is" "ics" \
                 "c" "cv" "cf" "cs" "cvf" "cvs" "cfs" "cvfs" \
                 "v" "vf" "vs" "vfs" \
                 "f" "fs" "s"
do
    export CUDA_VISIBLE_DEVICES=1
    python run.py \
        --output_dir=./saved_models/${model}/${mask_type} \
        --model_type=roberta \
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