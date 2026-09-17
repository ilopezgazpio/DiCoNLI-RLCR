"""Optional timing telemetry; no tracking service is needed for local runs."""

from contextlib import contextmanager
from functools import wraps
from time import perf_counter


@contextmanager
def profiling_context(trainer, name):
    started = perf_counter()
    try:
        yield
    finally:
        if trainer.accelerator.is_main_process:
            values = {
                f"profiling/Time taken: {type(trainer).__name__}.{name}": perf_counter() - started
            }
            if "wandb" in trainer.args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log(values)
            if "mlflow" in trainer.args.report_to:
                import mlflow

                if mlflow.active_run() is not None:
                    mlflow.log_metrics(values, step=trainer.state.global_step)


def profiling_decorator(function):
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        with profiling_context(self, function.__name__):
            return function(self, *args, **kwargs)

    return wrapped
