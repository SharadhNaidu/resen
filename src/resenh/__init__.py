"""resenh: robust low-light image restoration and enhancement.

>>> import resenh
>>> result = resenh.enhance("night.jpg")          # adaptive hybrid (classical + deep)
>>> resenh.save_image(result.image, "night_enhanced.png")
"""

from resenh.io import ImageInputError, load_image, save_image
from resenh.pipeline import MODES, EnhanceResult, enhance

__version__ = "0.1.0"

__all__ = ["enhance", "EnhanceResult", "MODES", "load_image", "save_image", "ImageInputError",
           "__version__"]
