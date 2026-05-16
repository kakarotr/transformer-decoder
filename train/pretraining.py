import json
import math
import os
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, no_type_check

import torch
import torch.distributed as dist
import torch.optim as optim
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text
from safetensors.torch import save_file
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

from models.config import TransformerConfig
from models.liger.causal_lm import CausalLanguageModel
from models.liger.loss import compute_loss, eval_compute_loss
from train.base_model import MetricsData, TrainingArguments
from train.dataset import PackedTokenDataset
from train.utils import parse_args


@dataclass
class TrainState:
    micro_steps: int = 0
    optimizer_steps: int = 0
    epoch: int = 0
    accum_loss_sum: float = 0.0
    accum_token_count: float = 0.0
    should_stop: bool = False


class PretrainingTrainer:
    def __init__(
        self,
        *,
        arguments: TrainingArguments,
        training_stage: Literal["pretrain", "continued_pretrain"] = "pretrain",
    ):
        Path("logs").mkdir(exist_ok=True)
        self.arguments = arguments
        self.output_path = self.arguments.output_path

        self.is_distributed, self.world_size, self.local_rank, self.device = self._init_distributed()
        self.is_main_process = (not self.is_distributed) or dist.get_rank() == 0
        if self.is_main_process:
            print(self.arguments.model_dump_json(indent=2))

        self.config, self.tokenizer, self.model = self._get_tokenizer_and_model(self.arguments.model_path)

        self.train_dataset, self.eval_dataset = self._load_dataset(self.arguments.data_path)
        self.train_dataloader, self.eval_dataloader = self._load_dataloader()

        self.compute_loss, self.eval_compute_loss = self._get_loss_fn()

        self.token_per_update = (
            self.arguments.per_device_train_batch_size
            * self.world_size
            * self.config.max_position_embeddings
            * self.arguments.gradient_accumulation_steps
        )

        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

        # 指标
        self.console = Console() if self.is_main_process else None
        self.live: Live | None = None
        self.progress: Progress | None = None
        self.progress_task_id: TaskID | None = None
        self.display_mode = "train"
        metric_window = max(int(self.arguments.logging_steps), 1)
        now = time.perf_counter()
        self.monitor_data = MetricsData(
            recent_train_losses=deque(maxlen=metric_window),
            recent_tokens_per_sec=deque(maxlen=metric_window),
            start_time=now,
            last_update_time=now,
        )

    def __call__(self):
        state = self._init_train_state()

        with self._live_context():
            self._refresh_live(state.optimizer_steps)
            self._run_training_loop(state)

        self._save_checkpoint(state=state)
        self._emit_event(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [done] training finished at step={state.optimizer_steps}"
        )

        if self.is_distributed:
            dist.destroy_process_group()

    def _init_train_state(self):
        self.optimizer.zero_grad(set_to_none=True)
        now = time.perf_counter()
        self.monitor_data.start_time = now
        self.monitor_data.last_update_time = now
        return TrainState()

    def _run_training_loop(self, state: TrainState):
        while state.optimizer_steps < self.arguments.max_steps and not state.should_stop:
            self._run_epoch(state)

    def _run_epoch(self, state: TrainState):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(state.epoch)

        for batch in self.train_dataloader:
            self._run_micro_step(batch, state)
            if state.optimizer_steps >= self.arguments.max_steps or state.should_stop:
                break

        state.epoch += 1

    def _run_micro_step(self, batch: torch.Tensor, state: TrainState):
        input_ids = batch.to(device=self.device, non_blocking=True)
        labels = input_ids

        is_update_step = ((state.micro_steps + 1) % self.arguments.gradient_accumulation_steps) == 0
        sync_context = self.model.no_sync() if self.is_distributed and not is_update_step else nullcontext()

        with sync_context:
            raw_loss = self._compute_raw_loss(input_ids, labels)
            self._accumulate_loss_metrics(raw_loss, labels, state)

            loss = raw_loss / self.arguments.gradient_accumulation_steps
            loss.backward()
            state.micro_steps += 1

            if state.micro_steps % self.arguments.gradient_accumulation_steps != 0:
                return

            self._apply_optimizer_step(state)
            self._handle_step_side_effects(state)

    def _compute_raw_loss(self, input_ids: torch.Tensor, labels: torch.Tensor):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden_states = self.model(input_ids)
            lm_head_weight = self.model.lm_head.weight if not self.is_distributed else self.model.module.lm_head.weight
            return self.compute_loss(hidden_states, lm_head_weight, labels)

    def _accumulate_loss_metrics(self, raw_loss: torch.Tensor, labels: torch.Tensor, state: TrainState):
        valid_token_count = labels[:, 1:].ne(-100).sum().item()
        state.accum_loss_sum += raw_loss.detach().item() * valid_token_count
        state.accum_token_count += valid_token_count

    def _apply_optimizer_step(self, state: TrainState):
        train_loss = state.accum_loss_sum / max(state.accum_token_count, 1)
        self.monitor_data.last_train_loss = train_loss
        self.monitor_data.recent_train_losses.append(train_loss)
        state.accum_loss_sum = 0.0
        state.accum_token_count = 0

        grad_norm = clip_grad_norm_(
            self.model.parameters(),
            max_norm=self.arguments.max_grad_norm if self.arguments.max_grad_norm else float("inf"),
        )
        self.monitor_data.last_grad_norm = grad_norm.item()

        self.optimizer.step()
        self.scheduler.step()
        state.optimizer_steps += 1
        self._update_throughput_metrics()

    def _update_throughput_metrics(self):
        now = time.perf_counter()
        step_elapsed = max(now - self.monitor_data.last_update_time, 1e-6)
        self.monitor_data.last_update_time = now
        self.monitor_data.recent_tokens_per_sec.append(self.token_per_update / step_elapsed)

    def _handle_step_side_effects(self, state: TrainState):
        self._refresh_live(state.optimizer_steps)
        if state.optimizer_steps % self.arguments.logging_steps == 0:
            self._write_metrics_log(state.optimizer_steps)
        self._run_eval(state.optimizer_steps)
        if state.optimizer_steps % self.arguments.save_steps == 0:
            self._save_checkpoint(state=state)
        self.optimizer.zero_grad(set_to_none=True)

    def _write_metrics_log(self, optimizer_steps: int):
        if not self.is_main_process:
            return

        with open("logs/metric.jsonl", mode="a", encoding="utf-8") as f:
            data = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "step": optimizer_steps,
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6g}",
                "loss": f"{self.monitor_data.last_train_loss:.4f}",
                "grad_norm": f"{float(self.monitor_data.last_grad_norm):.4f}",
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _run_eval(self, optimizer_steps: int):
        if not (optimizer_steps % self.arguments.eval_steps == 0 or optimizer_steps == self.arguments.max_steps):
            return

        self.display_mode = "eval"
        self._refresh_live(optimizer_steps)
        self._emit_event(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"[eval] start step={optimizer_steps}/{self.arguments.max_steps}"
        )

        self.monitor_data.last_eval_loss = self._evaluate()
        try:
            self.monitor_data.last_eval_ppl = math.exp(self.monitor_data.last_eval_loss)
        except OverflowError:
            self.monitor_data.last_eval_ppl = float("inf")

        self._emit_event(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"[eval] done step={optimizer_steps} "
            f"loss={self.monitor_data.last_eval_loss:.4f} "
            f"ppl={self.monitor_data.last_eval_ppl:.4f}"
        )
        self.display_mode = "train"
        self._refresh_live(optimizer_steps)

        if not self.is_main_process:
            return
        with open("logs/eval.jsonl", mode="a") as f:
            data = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "step": optimizer_steps,
                "lr": f"{self.optimizer.param_groups[0]['lr']:.6g}",
                "eval_loss": f"{self.monitor_data.last_eval_loss:.4f}",
                "eval_ppl": f"{self.monitor_data.last_eval_ppl:.4f}",
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _save_checkpoint(self, state: TrainState):
        optimizer_steps = state.optimizer_steps if state.optimizer_steps != 0 else None
        if self.is_main_process:
            checkpoint_path = (
                Path(f"{self.output_path}/checkpoint-{optimizer_steps}/model.safetensors")
                if optimizer_steps is not None
                else Path(f"{self.output_path}/model.safetensors")
            )
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            state_dict = self.model.module.state_dict() if self.is_distributed else self.model.state_dict()
            torch.save(
                {
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "train_state": asdict(state),
                    "rng_state": torch.cuda.get_rng_state_all(),
                },
                checkpoint_path.parent / "trainer_state.pt",
            )
            save_file(state_dict, checkpoint_path)

    def _init_distributed(self):
        is_distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
        if is_distributed:
            backend = "nccl" if dist.is_nccl_available() else "gloo"
            dist.init_process_group(backend=backend)
            local_rank = int(os.environ["LOCAL_RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            world_size = 1
            local_rank = 0
            device = torch.device("cuda")

        return is_distributed, world_size, local_rank, device

    def _load_dataset(self, data_dir):
        train_dataset = PackedTokenDataset(Path(f"{data_dir}/train"), seq_len=self.config.max_position_embeddings)
        eval_dataset = PackedTokenDataset(Path(f"{data_dir}/eval"), seq_len=self.config.max_position_embeddings)

        return train_dataset, eval_dataset

    def _load_dataloader(self):
        train_sampler = None
        eval_sampler = None
        self.train_sampler = None

        if self.is_distributed:
            train_sampler = DistributedSampler(dataset=self.train_dataset, shuffle=True, drop_last=True)
            eval_sampler = DistributedSampler(dataset=self.eval_dataset, shuffle=False, drop_last=False)
            self.train_sampler = train_sampler

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.arguments.per_device_train_batch_size,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=4,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            pin_memory=True,
            drop_last=True,
        )
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.arguments.per_device_eval_batch_size,
            num_workers=8,
            persistent_workers=True,
            prefetch_factor=4,
            sampler=eval_sampler,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )

        return train_dataloader, eval_dataloader

    def _get_tokenizer_and_model(self, model_path: str):
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        with open(f"{model_path}/config.json", mode="r", encoding="utf-8") as f:
            config = TransformerConfig.model_validate_json(f.read())

        model = CausalLanguageModel(config=config).to(self.device)

        if self.is_distributed:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=False,
            )
        return config, tokenizer, model

    def _get_loss_fn(self):
        return compute_loss, eval_compute_loss

    def _init_scheduler(self):
        if self.arguments.warmup_steps > 0:
            cosine_steps = self.arguments.max_steps - self.arguments.warmup_steps
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[
                    LinearLR(
                        self.optimizer,
                        start_factor=self.arguments.warmup_start_factor,
                        total_iters=self.arguments.warmup_steps,
                    ),
                    CosineAnnealingLR(self.optimizer, T_max=cosine_steps, eta_min=self.arguments.learning_rate * 0.1),
                ],
                milestones=[self.arguments.warmup_steps],
            )
        else:
            scheduler = CosineAnnealingLR(
                self.optimizer, T_max=self.arguments.max_steps, eta_min=self.arguments.learning_rate * 0.1
            )
        return scheduler

    def _init_optimizer(self):
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if "embed" in name:
                no_decay.append(param)
            elif param.ndim < 2:
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = optim.AdamW(
            params=[
                {"params": decay, "weight_decay": 0.1},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.arguments.learning_rate,
            eps=1e-8,
            betas=(0.9, 0.95),
            fused=True,
        )
        return optimizer

    @no_type_check
    @torch.no_grad()
    def _evaluate(self):
        self.model.eval()

        loss_sum = torch.zeros(1, device=self.device)
        token_count = torch.zeros(1, device=self.device, dtype=torch.long)

        for batch in self.eval_dataloader:
            input_ids = batch.to(device=self.device, non_blocking=True)
            labels = input_ids

            with torch.autocast("cuda", dtype=torch.bfloat16):
                hidden_states = self.model(input_ids)
                lm_head_weight = (
                    self.model.lm_head.weight if not self.is_distributed else self.model.module.lm_head.weight
                )
                loss, valid_token_count = self.eval_compute_loss(hidden_states, lm_head_weight, labels)

            loss_sum += loss
            token_count += valid_token_count

        if self.is_distributed:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(token_count, op=dist.ReduceOp.SUM)

        mean_loss = (loss_sum / token_count.clamp_min(1)).item()
        self.model.train()
        return mean_loss

    def _live_context(self):
        if not self.is_main_process:
            return nullcontext()

        self.progress = Progress(
            TextColumn("[bold cyan]{task.fields[mode]}", justify="right"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TextColumn("[bold]{task.completed:.0f} / {task.total:.0f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )

        self.progress_task_id = self.progress.add_task(
            "train",
            total=self.arguments.max_steps,
            completed=0,
            mode="train",
        )

        self.live = Live(
            self._render_dashboard(),
            console=self.console,
            refresh_per_second=4,
            transient=False,
        )

        return self.live

    def _refresh_live(self, optimizer_steps: int):
        if not self.is_main_process:
            return

        assert self.live is not None and self.progress is not None and self.progress_task_id is not None
        mode_text = "Eval" if self.display_mode == "eval" else "Train"
        self.progress.update(self.progress_task_id, completed=optimizer_steps, mode=mode_text)
        self.live.update(self._render_dashboard(), refresh=True)

    def _render_dashboard(self):
        metrics_text = Text(self._build_metrics_line(), overflow="ellipsis", no_wrap=True)
        assert self.progress is not None
        return Group(self.progress, metrics_text)

    def _build_metrics_line(self):
        loss_window = self.monitor_data.recent_train_losses
        speed_window = self.monitor_data.recent_tokens_per_sec
        avg_loss = sum(loss_window) / len(loss_window) if loss_window else 0.0
        avg_tokens = sum(speed_window) / len(speed_window) if speed_window else 0.0
        lr = self.optimizer.param_groups[0]["lr"]
        eval_display = "running..." if self.display_mode == "eval" else f"{self.monitor_data.last_eval_loss:.4f}"
        ppl_display = "running..." if self.display_mode == "eval" else f"{self.monitor_data.last_eval_ppl:.4f}"

        return (
            f"step_loss: {self.monitor_data.last_train_loss:.4f} | "
            f"avg_loss@{max(int(self.arguments.logging_steps), 1)}: {avg_loss:.4f} | "
            f"grad: {self.monitor_data.last_grad_norm:.4f} | "
            f"lr: {lr:.6g} | "
            f"tok/s(avg): {avg_tokens:,.0f} | "
            f"eval_loss: {eval_display} | ppl: {ppl_display} | "
        )

    def _emit_event(self, message: str):
        if self.is_main_process and self.live is not None:
            self.live.console.print(message)


if __name__ == "__main__":
    arguments = parse_args()
    trainer = PretrainingTrainer(arguments=arguments)
    trainer()
