"""resenh: robust low-light image restoration and enhancement.

    >>> import resenh
    >>> res = resenh.enhance("night.jpg")        # path, URL, bytes, PIL image or NumPy array
    >>> res.save("night_bright.png")
    >>> res                                       # in a notebook: before / after
    >>> print(res.summary())                      # what was measured, chosen and why

Modes: ``"auto"`` (default, the hybrid), ``"hybrid"``, ``"dl"`` (network only) and
``"classical"`` (no deep learning, no PyTorch needed).
"""

from resenh.io import ImageInputError, load_image, save_image
from resenh.pipeline import MODES, PRESETS, EnhanceResult, enhance
from resenh.utils import download_weights, enhance_folder, info

__version__ = "0.1.2"

__all__ = [
    "enhance", "enhance_folder", "EnhanceResult", "MODES", "PRESETS",
    "load_image", "save_image", "ImageInputError", "download_weights", "info", "__version__",
]
