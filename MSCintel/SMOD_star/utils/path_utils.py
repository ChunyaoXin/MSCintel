from pathlib import Path


def repo_root_from(path=None):
    if path is None:
        return Path(__file__).resolve().parents[1]
    path = Path(path).resolve()
    return path if path.is_dir() else path.parent


def resolve_path(repo_root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(repo_root) / path


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
