"""
Training demo script for deep learning curve prediction.

Usage:
    Install the package in editable mode first:
        pip install -e .
    Then run:
        python scripts/train_demo.py
"""

import logging
from pathlib import Path

from ptpd_calibration.config import DeepLearningSettings
from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
from ptpd_calibration.ml.deep.synthetic_data import generate_training_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main() -> None:
    logger.info("Starting training demo...")
    
    # 1. Generate synthetic data
    logger.info("Generating synthetic data...")
    database = generate_training_data(num_records=200, seed=42)
    logger.info(f"Generated {len(database)} records.")

    # 2. Configure settings
    settings = DeepLearningSettings(
        num_epochs=10,  # Short run for demo
        batch_size=16,
        learning_rate=1e-3,
        device="cpu", # Use CPU for safety in demo
        early_stopping_patience=3
    )
    
    # 3. Initialize predictor
    predictor = DeepCurvePredictor(settings)
    
    # 4. Train
    logger.info("Training model...")
    stats = predictor.train(
        database=database,
        val_ratio=0.2,
        num_epochs=settings.num_epochs
    )
    
    logger.info(f"Training complete. Stats: {stats}")
    
    # 5. Evaluate
    logger.info("Evaluating model...")
    # We can use the validation set implicitly handled or do a manual check
    # Let's just predict on a sample record
    sample_record = database.get_all_records()[0]
    result = predictor.predict(sample_record)
    
    logger.info(f"Prediction made. Curve length: {len(result.curve)}")
    logger.info(f"Mean predicted value: {result.curve.mean():.4f}")
    
    # 6. Save
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    save_path = output_dir / "demo_model"
    predictor.save(save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    main()
