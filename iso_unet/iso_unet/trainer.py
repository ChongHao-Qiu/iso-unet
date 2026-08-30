import os
import lightning as L
from . import utils


class _SimpleEpochLogger(L.pytorch.callbacks.Callback):
    """Print a concise training status line every N epochs, replacing the noisy Lightning progress bar."""
    def __init__(self, every: int = 1):
        super().__init__()
        self.every = every

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch % self.every) != 0:
            return
        m = trainer.callback_metrics
        parts = [f'Epoch {trainer.current_epoch:>4d}/{trainer.max_epochs}']
        for key in ('train_loss_epoch', 'train_loss', 'train_mse',
                    'train_prompt', 'valid_loss', 'film_gamma_norm'):
            if key in m:
                v = m[key]
                v = float(v) if hasattr(v, 'item') else float(v)
                parts.append(f'{key.replace("train_loss_epoch","train_loss")}={v:.4f}')
        print('  ' + '  '.join(parts), flush=True)


class Trainer(L.Trainer):
    def __init__(self, log_dirpath='./logs', name='IsoGen', min_epochs=1, max_epochs=1000, patience=10, min_delta=0.001,
                 gradient_clip_val=None, gradient_clip_algorithm='norm',
                 enable_progress_bar: bool = True,
                 simple_epoch_log: bool = False, simple_epoch_every: int = 1,
                 accelerator: str = 'auto', devices=1):
        utils.p_header(f'Name: {name}')

        early_stop = L.pytorch.callbacks.early_stopping.EarlyStopping(
            monitor='valid_loss',  # Metric to monitor
            patience=patience,
            mode='min',
            verbose=True,
            min_delta=min_delta
        )

        ckpt_callback = L.pytorch.callbacks.ModelCheckpoint(
            dirpath=log_dirpath, filename=name,
        )

        # Remove the checkpoint file if it exists
        self.ckpt_fpath = os.path.join(log_dirpath, f'{name}.ckpt')
        utils.p_header(f'Checkpoint path: {os.path.abspath(self.ckpt_fpath)}')

        if os.path.exists(self.ckpt_fpath):
            os.remove(self.ckpt_fpath)
        else:
            pass

        callbacks = [early_stop, ckpt_callback]
        if simple_epoch_log:
            callbacks.append(_SimpleEpochLogger(every=simple_epoch_every))

        # 'auto' picks CUDA when available and falls back to CPU otherwise, so the
        # pipeline is runnable (if slow) on a machine without a GPU. Pass
        # accelerator='gpu' explicitly to fail loudly when no GPU is present.
        super().__init__(
            accelerator=accelerator, devices=devices, strategy='auto',
            min_epochs=min_epochs, max_epochs=max_epochs,
            default_root_dir=log_dirpath, callbacks=callbacks,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            enable_progress_bar=enable_progress_bar,
        )