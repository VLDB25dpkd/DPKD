export TASK_NAME=sst2
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export HF_HUB_OFFLINE=1


python run_glue_no_trainer.py \
  --model_name_or_path "sst2_eps1_training_epoch10" \
  --task_name $TASK_NAME \
  --max_length 128 \
  --per_device_train_batch_size 2048 \
  --learning_rate 5e-4 \
  --num_train_epochs 20 \
  --output_dir "./eps1/sst2_5e-4" \
  --sigma 3.4543945312499993
  # --overwrite_output_dir True