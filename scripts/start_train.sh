PYTHONPATH=$(pwd) python -m torch.distributed.run \
  --nproc_per_node=4 \
  train/pretraining.py \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 2e-4 \
  --max_steps 95000 \
  --gradient_accumulation_steps 8 \
  --warmup_steps_ratio 0.03 \
  --eval_steps_ratio 0.05 \
  --logging_steps 100 \
  --save_steps 10000 \
  --max_grad_norm 1.0 \
  --model_path artifacts \
  --data_path ... \
  --output_path ...