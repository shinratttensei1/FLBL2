from typing import List, Tuple, Union, Optional, Dict
from flwr.serverapp.strategy import FedAvg
from flwr.app import RecordDict

_LR_DECAY = 0.80   # multiply LR by this factor each round
_LR_FLOOR = 0.05   # never drop below this fraction of the base LR


class MedicalFedAvg(FedAvg):
    """FedAvg with per-round learning-rate decay.

    Clients receive a decaying LR each round:
        lr_round = base_lr * decay^(round-1),  floored at base_lr * floor_frac

    This prevents the model from oscillating in later rounds after the initial
    fast learning phase (typically rounds 1-3).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_lr: Optional[float] = None

    def configure_train(self, server_round, arrays, config, grid):
        if "lr" in config:
            if self._base_lr is None:
                self._base_lr = float(config["lr"])
            round_lr = self._base_lr * (_LR_DECAY ** (server_round - 1))
            round_lr = max(round_lr, self._base_lr * _LR_FLOOR)
            config["lr"] = round_lr
            print(
                f"  [FedAvg] Round {server_round}: "
                f"lr={round_lr:.6f}  (base={self._base_lr:.6f}, decay={_LR_DECAY})"
            )
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        return super().aggregate_fit(server_round, results, failures)
