python3.8 /content/drive/MyDrive/r10922165/predict_context.py  \
  --model_name_or_path /content/drive/MyDrive/r10922165/cs/ \
  --test_file "${2}" \
  --output_dir /content/drive/MyDrive/r10922165/cs/  \
  --context_file "${1}" \
  --pred_file /content/drive/MyDrive/r10922165/pred_relevant.csv  \
  --do_train False  \
  --do_predict  True  \
  --per_device_eval_batch_size 16 \
  --max_seq_length 512  \
  --dataloader_drop_last False 

python3.8 /content/drive/MyDrive/r10922165/predict_answer.py \
  --model_name_or_path /content/drive/MyDrive/r10922165/qa/ \
  --do_predict  \
  --test_file "${2}" \
  --relevant_file /content/drive/MyDrive/r10922165/pred_relevant.csv  \
  --pred_file "${3}" \
  --context_file "${1}" \
  --output_dir /content/drive/MyDrive/r10922165/qa/  \
  --max_seq_length 512  \
  --doc_stride 128 \
  --per_device_eval_batch_size 16 \
  --dataloader_drop_last False