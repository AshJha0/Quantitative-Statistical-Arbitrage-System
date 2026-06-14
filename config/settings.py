"""
System Configuration
Centralized settings for the quant system.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import os


@dataclass
class SystemConfig:
    """Main system configuration."""

    # Paths
    data_dir: str = "data"
    model_dir: str = "models"
    output_dir: str = "output"
    log_dir: str = "logs"

    # Capital
    initial_capital: float = 1_000_000
    base_currency: str = "USD"

    # Data
    default_frequency: str = "1min"
    warmup_period: int = 252
    use_synthetic_data: bool = True

    # Trading
    trading_hours: tuple = (0, 24)  # 24/7 for crypto/fx
    max_pairs: int = 10

    # Risk
    max_drawdown: float = 0.15
    max_position_size: float = 0.25
    commission_rate: float = 0.0001

    # RL
    rl_training_steps: int = 50000
    rl_evaluation_episodes: int = 5

    # Dashboard
    dashboard_theme: str = "dark"
    refresh_interval_ms: int = 5000

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> 'SystemConfig':
        """Load configuration from environment variables."""
        return cls(
            data_dir=os.getenv('QUANT_DATA_DIR', 'data'),
            model_dir=os.getenv('QUANT_MODEL_DIR', 'models'),
            initial_capital=float(os.getenv('QUANT_CAPITAL', '1000000')),
            log_level=os.getenv('QUANT_LOG_LEVEL', 'INFO')
        )

    def create_directories(self) -> None:
        """Create required directories."""
        import os
        for dir_path in [self.data_dir, self.model_dir, self.output_dir, self.log_dir]:
            os.makedirs(dir_path, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()


def get_config() -> SystemConfig:
    """Get system configuration."""
    return DEFAULT_CONFIG


def update_config(**kwargs) -> None:
    """Update configuration values."""
    global DEFAULT_CONFIG
    for key, value in kwargs.items():
        if hasattr(DEFAULT_CONFIG, key):
            setattr(DEFAULT_CONFIG, key, value)
