from ptpd_calibration.deep_learning.training.pipelines import CurveTrainingPipeline, TrainingConfig


def verify_fix() -> None:
    print("Initializing CurveTrainingPipeline...")
    config = TrainingConfig(device="cpu")
    pipeline = CurveTrainingPipeline(config)

    print("Creating model...")
    try:
        _ = pipeline._create_model()
        print("Model created successfully.")
    except Exception as e:
        print(f"FAIL: Model creation failed: {e}")
        return

    print("Creating data loaders...")
    try:
        loaders = pipeline._create_data_loaders()
        train_loader = loaders[0]
        print("Data loaders created.")
    except Exception as e:
        print(f"FAIL: Data loader creation failed: {e}")
        return

    print("Fetching one batch...")
    try:
        batch = next(iter(train_loader))
        densities, conditions, curves = batch
        print(f"Densities shape: {densities.shape}")
        print(f"Conditions shape: {conditions.shape}")
        print(f"Curves shape: {curves.shape}")

        expected_seq_len = 256
        expected_feats = 8

        if densities.shape[1] != expected_seq_len:
            print(
                f"FAIL: Unexpected sequence length {densities.shape[1]}, expected {expected_seq_len}"
            )
        if densities.shape[2] != expected_feats:
            print(f"FAIL: Unexpected feature dim {densities.shape[2]}, expected {expected_feats}")

    except Exception as e:
        print(f"FAIL: Batch fetching failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    verify_fix()
