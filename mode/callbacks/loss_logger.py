"""
Callback to log training metrics to file
"""
import logging
from pytorch_lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


class LossLogger(Callback):
    """
    Logs training loss values to the logging file at regular intervals
    """

    def __init__(self, log_every_n_steps: int = 50):
        """
        Args:
            log_every_n_steps: Log metrics every N training steps
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs,
        batch,
        batch_idx: int
    ) -> None:
        """Called after training batch ends"""

        if (trainer.global_step + 1) % self.log_every_n_steps != 0:
            return

        if trainer.is_global_zero:
            metrics_to_log = {}

            for key, value in trainer.callback_metrics.items():
                if 'train/' in key and '_step' in key:
                    metrics_to_log[key] = value.item() if hasattr(value, 'item') else value

            if metrics_to_log:
                metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics_to_log.items()])
                logger.info(
                    f"Step {trainer.global_step} | Epoch {trainer.current_epoch} | {metric_str}"
                )
