
import numpy as np
from PIL import Image

from ptpd_calibration.imaging.processor import ImageProcessor, ProcessingResult


def test_grayscale_flat_validation():
    processor = ImageProcessor()

    # Create flat grayscale image (all 128)
    arr = np.full((100, 100), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")

    # Create dummy result
    result = ProcessingResult(
        image=img,
        original_size=(100, 100),
        original_mode="L",
        original_format=None,
        original_dpi=None,
        curve_applied=False,
        inverted=False
    )

    # Validate
    stats = processor.validate_image_channels(result, require_uniform=False)

    print(f"Grayscale Flat Image Stats: {stats['all_channels_processed']}")

    # Check color behavior
    arr_color = np.zeros((100, 100, 3), dtype=np.uint8)
    arr_color[:,:,0] = 128 # Channel 0 is flat
    arr_color[:,:,1] = np.random.randint(0, 255, (100, 100)) # Channel 1 is valid
    arr_color[:,:,2] = np.random.randint(0, 255, (100, 100)) # Channel 2 is valid

    img_color = Image.fromarray(arr_color, mode="RGB")
    result_color = ProcessingResult(
        image=img_color,
        original_size=(100, 100),
        original_mode="RGB",
        original_format=None,
        original_dpi=None,
        curve_applied=False,
        inverted=False
    )
    stats_color = processor.validate_image_channels(result_color, require_uniform=False)

    print(f"Color Image with one flat channel Stats: {stats_color['all_channels_processed']}")

if __name__ == "__main__":
    test_grayscale_flat_validation()
