from pathlib import Path

import numpy as np
from PIL import Image


def create_dummy_step_tablet(output_path: Path):
    """Create a dummy 21-step tablet image."""
    # Create a 21-step gradient
    steps = np.linspace(0, 255, 21, dtype=np.uint8)

    # Create image array (height 100, width 21*50)
    step_width = 50
    height = 100
    width = len(steps) * step_width

    image_data = np.zeros((height, width), dtype=np.uint8)

    for i, value in enumerate(steps):
        start_x = i * step_width
        end_x = (i + 1) * step_width
        image_data[:, start_x:end_x] = value

    # Add some noise
    noise = np.random.normal(0, 2, image_data.shape).astype(np.int8)
    image_data = np.clip(image_data.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Save
    img = Image.fromarray(image_data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"Created dummy step tablet at {output_path}")


if __name__ == "__main__":
    output = Path("tests/fixtures/step_tablet_dummy.png")
    create_dummy_step_tablet(output)
