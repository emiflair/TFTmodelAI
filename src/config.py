"""Project-level configuration data structures for the EURUSD TFT pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "EURUSD_15M.csv"
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"


@dataclass
class SeedConfig:
    global_seed: int = 42
    numpy_seed: Optional[int] = None
    torch_seed: Optional[int] = None


@dataclass
class DataConfig:
    csv_path: Path = DATA_PATH
    timezone: str = "UTC"
    frequency: str = "15min"
    lookback_bars: int = 256
    horizon_bars: int = 3
    min_history_bars: int = 40_000
    max_forward_fill_bars: int = 3
    drop_threshold_bars: int = 4
    winsorize_pct: float = 0.001


@dataclass
class QuantileConfig:
    quantiles: Sequence[float] = (0.1, 0.5, 0.9)
    default_band: Sequence[float] = (0.1, 0.9)

    def as_list(self) -> list[float]:
        return list(self.quantiles)


@dataclass
class TrainingWindowConfig:
    train_months: int = 1
    val_months: int = 1
    test_months: int = 1
    stride_months: int = 1


@dataclass
class TrainingConfig:
    hidden_size: int = 128
    attention_head_size: int = 2
    dropout: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    gradient_clip_val: float = 1.0
    max_epochs: int = 12
    early_stop_patience: int = 5
    mixed_precision: bool = False
    deterministic: bool = True
    fast_dev_run: bool = True
    fast_max_epochs: int = 3
    fast_max_splits: int = 2
    fast_batch_size: Optional[int] = 128


@dataclass
class EvaluationConfig:
    coverage_band: Sequence[float] = (0.1, 0.9)
    target_coverage: float = 0.80
    coverage_tolerance: float = 0.05
    pf_band_thresholds: Sequence[float] = (0.002,)
    pf_band_norm_multiplier: float = 1.0
    directional_hit_target: float = 0.60
    pf_target: float = 1.2


@dataclass
class ArtifactPaths:
    checkpoint_dir: Path = ARTIFACT_ROOT / "checkpoints"
    scalers_dir: Path = ARTIFACT_ROOT / "scalers"
    manifests_dir: Path = ARTIFACT_ROOT / "manifests"
    metrics_dir: Path = ARTIFACT_ROOT / "metrics"
    model_cards_dir: Path = ARTIFACT_ROOT / "model_cards"
    latest_symlink: Path = ARTIFACT_ROOT / "checkpoints" / "tft_EURUSD_15m_3B_latest.ckpt"


@dataclass
class LoggingConfig:
    enable_tensorboard: bool = True
    log_every_n_steps: int = 25
    version_logging: bool = True


@dataclass
class ProjectConfig:
    seed: SeedConfig = field(default_factory=SeedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    quantiles: QuantileConfig = field(default_factory=QuantileConfig)
    windows: TrainingWindowConfig = field(default_factory=TrainingWindowConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    artifacts: ArtifactPaths = field(default_factory=ArtifactPaths)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


DEFAULT_CONFIG = ProjectConfig()


def ensure_artifact_dirs(paths: ArtifactPaths) -> None:
    """Create expected artifact directories if they do not exist."""
    for path in (
        paths.checkpoint_dir,
        paths.scalers_dir,
        paths.manifests_dir,
        paths.metrics_dir,
        paths.model_cards_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
