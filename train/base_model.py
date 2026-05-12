import math
from collections import deque
from typing import Annotated, Deque

from pydantic import BaseModel, Field, model_validator


class TrainingArguments(BaseModel):
    model_path: str = Field(description="模型、分词器配置路径")
    data_path: str = Field(description="训练数据路径")
    output_path: str = Field(description="模型权重输出路径")
    learning_rate: float = Field(default=2e-4, description="优化器的基础学习率")
    per_device_train_batch_size: int = Field(default=8, description="单卡训练时每个 step 的 batch size")
    per_device_eval_batch_size: int = Field(default=8, description="单卡评估时的 batch size")
    max_steps: int = Field(default=-1, description="最大训练步数")
    num_train_epochs: int = Field(default=-1, description="训练轮数")
    gradient_accumulation_steps: int = Field(default=1, description="梯度累积步数")
    max_grad_norm: float = Field(default=1.0, description="梯度裁剪阈值")
    warmup_steps_ratio: float = Field(default=0.03, description="warmup 阶段占总训练步数的比例")
    warmup_start_factor: float = Field(default=0.1, description="warmup 起始学习率系数")
    eval_steps_ratio: float = Field(default=0.05, description="评估间隔占总训练步数的比例")
    logging_steps: int = Field(default=100, description="训练日志打印间隔")
    save_steps: int = Field(default=1000, description="模型保存间隔")

    # 派生字段：初始化后自动计算
    warmup_steps: int = Field(default=0, description="warmup 的实际步数")
    eval_steps: int = Field(default=0, description="实际评估间隔步数")

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
