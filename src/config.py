from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingConfig:
    random_seed: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    optuna_trials: int = 60
    primary_metric: str = "f1"
    primary_metric_name: str = "cv_f1"
    top_k_for_tuning: int = 2
    min_f1_improvement: float = 0.01
    target_f1_improvement: float = 0.03
    max_roc_auc_drop: float = 0.01
    threshold_points: int = 181
    search_space_version: str = "linear_v2"


TRAINING_CONFIG = TrainingConfig()


def config_as_dict() -> dict:
    return asdict(TRAINING_CONFIG)
