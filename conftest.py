import os
import warnings

os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")
warnings.filterwarnings(
    "ignore",
    message=r"Jupyter is migrating its paths to use standard platformdirs.*",
    category=DeprecationWarning,
)
