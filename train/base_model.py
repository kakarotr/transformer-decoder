import math
from collections import deque
from typing import Annotated, Deque

from pydantic import BaseModel, Field, model_validator


class TrainingArguments(BaseModel):
    output_dir: Annotated[str, Field(description="模型权重输出目录")] = "weight"
    learning_rate: Annotated[float, Field(description="优化器的基础学习率")] = 5e-4
    per_device_train_batch_size: Annotated[int, Field(description="单卡训练时每个 step 的 batch size")] = 8
    per_device_eval_batch_size: Annotated[int, Field(description="单卡评估时的 batch size")] = 8
    max_steps: Annotated[int, Field(description="最大训练步数")] = -1
    num_train_epochs: Annotated[int, Field(description="训练轮数")] = -1
    gradient_accumulation_steps: Annotated[int, Field(description="梯度累积步数")] = 1
    max_grad_norm: Annotated[float, Field(description="梯度裁剪阈值")] = 1.0
    warmup_steps_ratio: Annotated[float, Field(description="warmup 阶段占总训练步数的比例")] = 0.03
    warmup_start_factor: Annotated[float, Field(description="warmup 起始学习率系数")] = 0.1
    eval_steps_ratio: Annotated[float, Field(description="评估间隔占总训练步数的比例")] = 0.02
    logging_steps: Annotated[int, Field(description="训练日志打印间隔")] = 100
    save_steps: Annotated[int, Field(description="模型保存间隔")] = 1000
    log_dir: Annotated[str, Field(description="TensorBoard 日志输出目录")] = "/workspace"
    flush_secs: Annotated[int, Field(description="TensorBoard 写入磁盘的刷新间隔（秒）")] = 30
    use_torch_complie: Annotated[bool, Field(description="是否使用 torch.complie 优化")] = True
    early_stopping_patience: int = 4
    early_stopping_min_delta: float = 0.002

    # 派生字段：初始化后自动计算
    warmup_steps: Annotated[int, Field(description="warmup 的实际步数")] = 0
    eval_steps: Annotated[int, Field(description="实际评估间隔步数")] = 0

    @model_validator(mode="after")
    def compute_steps(self):
        if self.max_steps > 0:
            self.warmup_steps = max(1, math.ceil(self.max_steps * self.warmup_steps_ratio))
            self.eval_steps = max(1, math.ceil(self.max_steps * self.eval_steps_ratio))
        else:
            self.warmup_steps = 0
            self.eval_steps = 0
        return self

    @model_validator(mode="after")
    def validate_steps_and_epochs(self):
        if self.max_steps == -1 and self.num_train_epochs == -1:
            raise ValueError(
                "Invalid configuration: 'max_steps' and 'num_train_epochs' cannot both be -1. "
                "At least one of them must be set to a positive value."
            )
        return self


class MetricsData(BaseModel):
    last_train_loss: float = 0.0
    last_eval_loss: float = 0.0
    last_eval_ppl: float = 0.0
    last_grad_norm: float = 0.0
    recent_train_losses: Deque[float] = Field(default_factory=deque)
    recent_tokens_per_sec: Deque[float] = Field(default_factory=deque)
    start_time: float = 0.0
    last_update_time: float = 0.0
