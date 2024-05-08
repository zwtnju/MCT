lang=go
model_type=codet5
for mask_type in c cs s
do
    python run.py \
    --do_adapter \
    --output_dir results/${model_type}/adapter/$mask_type/$lang \
    --model_name_or_path models/$model_type/adapter/$mask_type  \
    --do_train \
    --do_test \
    --train_data_file dataset/CSN/$lang/train.jsonl \
    --eval_data_file dataset/CSN/$lang/valid.jsonl \
    --test_data_file dataset/CSN/$lang/test.jsonl \
    --codebase_file dataset/CSN/$lang/codebase.jsonl \
    --num_train_epochs 2 \
    --code_length 256 \
    --nl_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --seed 123456 \
    --model_type $model_type \
    --mask_type $mask_type
done
