from .pixel import PixelExtractor
from .bars import BarsExtractor
EXTRACTOR_MAP = {
    "pixel": PixelExtractor,
    "bars":  BarsExtractor,
}
