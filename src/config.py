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
    lookback_bars: int = 128      # Reduced from 256 for faster processing
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
    # ROBUST TRAINING: Use 6 months training + 1 month val + 1 month test
    # This gives model more data to learn from while still being recent
    train_months: int = 6   # 6 months of recent market data
    val_months: int = 1     # 1 month validation
    test_months: int = 1    # 1 month test
    stride_months: int = 3  # 3-month stride = ~3 folds covering last year


@dataclass
class TrainingConfig:
    # MEMORY-OPTIMIZED SETTINGS FOR GPU TRAINING (T4/L4/A100)
    # Balanced for performance and memory efficiency
    hidden_size: int = 160           # Reduced from 256 to fit GPU memory
    attention_head_size: int = 4     # Standard attention heads
    dropout: float = 0.15            # Increased dropout to prevent overfitting
    learning_rate: float = 2e-4      # Slightly lower for stable learning
    weight_decay: float = 1e-3       # L2 regularization
    batch_size: int = 64             # Reduced from 128 for memory efficiency
    gradient_clip_val: float = 0.1   # Prevent gradient explosions
    max_epochs: int = 40             # More epochs for better convergence
    early_stop_patience: int = 8     # Stop if no improvement for 8 epochs
    mixed_precision: bool = True     # Fast training with BF16/FP32
    deterministic: bool = False      # Non-deterministic is faster
    fast_dev_run: bool = False       # PRODUCTION MODE
    fast_max_epochs: int = 40        # Match max_epochs
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
