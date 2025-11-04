"""Project-level configuration data structures for the XAUUSD TFT pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "XAUUSD_15M.csv"
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
    lookback_bars: int = 256      # Increased from 128 to capture more market context
    horizon_bars: int = 3
    min_history_bars: int = 20_000  # Reduced to match smaller dataset
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
    # OPTIMIZED FOR RECENT DATA: More training data, shorter test windows
    train_months: int = 9   # Increased from 6 for more context
    val_months: int = 1     # 1 month validation
    test_months: int = 1    # 1 month test
    stride_months: int = 1  # 1-month stride for 8 folds (was 2)


@dataclass
class TrainingConfig:
    # OPTIMIZED FOR RECENT DATA & GENERALIZATION
    # Prevent overfitting with stronger regularization
    hidden_size: int = 160           # Balanced capacity
    attention_head_size: int = 4     # Standard attention heads
    dropout: float = 0.2             # Increased from 0.15 to prevent overfitting
    learning_rate: float = 2e-4      # Stable learning rate
    weight_decay: float = 1e-3       # L2 regularization
    batch_size: int = 64             # Memory-efficient batch size
    gradient_clip_val: float = 0.1   # Prevent gradient explosions
    max_epochs: int = 30             # Reduced from 40 to prevent overfitting
    early_stop_patience: int = 6     # Reduced from 8 for earlier stopping
    mixed_precision: bool = True     # Fast training with BF16/FP32
    deterministic: bool = False      # Non-deterministic is faster
    fast_dev_run: bool = False       # PRODUCTION MODE
    fast_max_epochs: int = 30        # Match max_epochs
    fast_max_splits: int = 3         # Train 3 folds for robustness
    fast_batch_size: Optional[int] = 64  # Memory-efficient batch size


@dataclass
class EvaluationConfig:
    coverage_band: Sequence[float] = (0.1, 0.9)
    target_coverage: float = 0.80
    coverage_tolerance: float = 0.05
    pf_band_thresholds: Sequence[float] = (0.001, 0.002, 0.003)  # Multiple thresholds
    pf_band_norm_multiplier: float = 1.0
    directional_hit_target: float = 0.52  # More realistic target
    pf_target: float = 1.15               # More achievable profit target
    # New evaluation parameters
    sharpe_target: float = 1.0            # Target Sharpe ratio
    max_drawdown_threshold: float = 0.05  # Maximum acceptable drawdown
    consistency_window: int = 100         # Rolling window for consistency metrics


@dataclass
class ArtifactPaths:
    checkpoint_dir: Path = ARTIFACT_ROOT / "checkpoints"
    scalers_dir: Path = ARTIFACT_ROOT / "scalers"
    manifests_dir: Path = ARTIFACT_ROOT / "manifests"
    metrics_dir: Path = ARTIFACT_ROOT / "metrics"
    model_cards_dir: Path = ARTIFACT_ROOT / "model_cards"
    latest_symlink: Path = ARTIFACT_ROOT / "checkpoints" / "tft_XAUUSD_15m_3B_latest.ckpt"


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
