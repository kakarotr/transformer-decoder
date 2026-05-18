PYTHONPATH=$(pwd) uv run -m torch.distributed.run \
  --nproc_per_node=8 \
  train/pretraining.py \
  --per_device_train_batch_size 7 \
  --per_device_eval_batch_size 7 \
  --learning_rate 2e-4 \
  --max_steps 36330 \
  --gradient_accumulation_steps 6 \
  --warmup_steps_ratio 0.03 \
  --eval_steps_ratio 0.02 \
  --logging_steps 100 \
  --save_steps 2000 \
  --max_grad_norm 1.0 \
  --model_path artifacts \
  --data_path /root/autodl-tmp/pretraining \
  --output_path /root/autodl-tmp/checkpoints