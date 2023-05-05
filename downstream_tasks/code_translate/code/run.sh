model=codebert
for mask_type in "base" "i" "ic" "is" "ics" \
                 "c" "cv" "cf" "cs" "cvf" \
                 "cvs" "cfs" "cvfs" "v" "vf" \
                 "vs" "vfs" "f" "fs" "s"
do
    python run.py \
      --do_train \
      --do_eval \
      --do_test \
      --model_type roberta \
      --model_name_or_path ../models/${model}/${mask_type} \
      --train_filename ../data/train.java-cs.txt.java,../data/train.java-cs.txt.cs \
      --dev_filename ../data/valid.java-cs.txt.java,../data/valid.java-cs.txt.cs \
      --test_filename ../data/test.java-cs.txt.java,../data/test.java-cs.txt.cs \
      --output_dir ./saved_models/${model}/${mask_type} \
      --max_source_length 512 \
      --max_target_length 512 \
      --beam_size 5 \
      --train_batch_size 16 \
      --eval_batch_size 16 \
      --learning_rate 5e-5 \
      --train_steps 100000 \
      --eval_steps 5000
done