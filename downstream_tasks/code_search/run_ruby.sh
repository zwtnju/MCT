lang=ruby
model_type=codebert
for mask_type in "base" "i" "ic" "is" "ics" \
                 "c" "cv" "cf" "cs" "cvf" \
                 "cvs" "cfs" "cvfs" "v" "vf" \
                 "vs" "vfs" "f" "fs" "s"
do
    python run.py \
    --output_dir results/${model_type}/$mask_type/$lang \
    --model_name_or_path models/$model_type/$mask_type  \
    --do_train \
    --do_test \
    --train_data_file dataset/CSN/$lang/train.jsonl \
    --eval_data_file dataset/CSN/$lang/valid.jsonl \
    --test_data_file dataset/CSN/$lang/test.jsonl \
    --codebase_file dataset/CSN/$lang/codebase.jsonl \
    --num_train_epochs 10 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 \
    --model_type $model_type
done
