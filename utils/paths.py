from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "datasets"
MODEL_DIR = PROJECT_ROOT / "models"
RESULT_DIR = PROJECT_ROOT / "results"


def project_path(*parts) -> Path:
    """Return an absolute path under the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def as_posix_path(path: Path) -> str:
    """Return a stable string path for config values and logs."""
    return path.as_posix()
