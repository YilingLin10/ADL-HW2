# ADL-HW2
## Kaggle link
https://www.kaggle.com/competitions/ntu-adl-hw2-spring-2021/leaderboard
## Environment setup (Python 3.8):
```
pip install -r requirements.txt
```

## Context Selection model:
* /path/to/train_file: path to your train file
* /path/to/validation_file: path to your validation file
* /path/to/context_file: path to your context file
* /path/to/output_dir/ : path to your output directory
```
!python3.8 train_mc.py  \
  --model_name_or_path bert-base-chinese  \
  --do_train  \
  --train_file /path/to/train_file \
  --do_eval   \
  --validation_file /path/to/validation_file  \
  --context_file /path/to/context_file \
  --output_dir /path/to/output_dir/  \
  --overwrite_output_dir \
  --overwrite_output \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --max_seq_length 512  \
  --per_device_eval_batch_size 16 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing
```
## Question Answering model:
* /path/to/train_file: Path to your train file
* /path/to/validation_file: path to your validation file
* /path/to/context_file: path to your context file
* /path/to/output_dir/ : path to your output directory
```
!python3.8 train_qa.py  \
  --model_name_or_path hfl/chinese-roberta-wwm-ext  \
  --do_train  \
  --train_file /path/to/train_file \
  --do_eval   \
  --validation_file /path/to/validation_file  \
  --context_file /path/to/context_file \
  --output_dir /path/to/output_dir/  \
  --overwrite_output_dir \
  --overwrite_output \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 512  \
  --doc_stride 128 \
  --per_device_eval_batch_size 8 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --gradient_checkpointing  \
  --load_best_model_at_end True \
  --evaluation_strategy steps \
  --eval_steps 1000 \
  --save_steps 1000 \
  --label_names ["start_positions","end_positions"] \
  --metric_for_best_model eval_exact_match  \
  --greater_is_better True
```
