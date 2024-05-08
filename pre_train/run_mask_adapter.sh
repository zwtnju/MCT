model_type=unixcoder
pretrained_model=microsoft/unixcoder-base
load_model_path=microsoft/unixcoder-base
mask_type=iden
total_mask_type=i
output_dir=./results/${model_type}/$total_mask_type
batch_size=32
gradient_accumulation_steps=1
train_epochs=1
train_log=${model_type}_${total_mask_type}_train.log
python3 run_mask.py --do_adapter --adapter_name ${total_mask_type}_adapter --adapter_type parallel \
--mask_type $mask_type --total_mask_type $total_mask_type --model_type $model_type \
--model_name_or_path $pretrained_model  --load_model_path $load_model_path --output_dir $output_dir --do_train \
--train_batch_size $batch_size --gradient_accumulation_steps $gradient_accumulation_steps --train_epochs $train_epochs \
>> $train_log 2>&1 &
