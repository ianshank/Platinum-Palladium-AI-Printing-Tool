"""
Training loop for AlphaZero-based calibration optimization.

Implements the self-play training cycle:
1. Self-play games to generate training data
2. Train neural network on collected data
3. Evaluate and save best model
4. Repeat
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from ptpd_calibration.alphazero.config import AlphaZeroConfig
from ptpd_calibration.alphazero.export.acv import export_to_acv
from ptpd_calibration.alphazero.game.platinum_game import PlatinumGame
from ptpd_calibration.alphazero.mcts.alpha_mcts import MCTSPlayer
from ptpd_calibration.alphazero.nn.policy_value_net import PolicyValueNet


class ReplayBuffer:
    """
    Experience replay buffer for training data.

    Stores (state, policy, value) tuples from self-play games.
    """

    def __init__(self, max_size: int = 10000):
        """
        Initialize replay buffer.

        Args:
            max_size: Maximum number of examples to store
        """
        self.max_size = max_size
        self.buffer: list[tuple[np.ndarray, np.ndarray, float]] = []

    def add(self, examples: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
        """Add examples to the buffer."""
        self.buffer.extend(examples)

        # Remove oldest examples if over capacity
        while len(self.buffer) > self.max_size:
            self.buffer.pop(0)

    def sample(self, batch_size: int) -> list[tuple[np.ndarray, np.ndarray, float]]:
        """Sample a batch from the buffer."""
        if len(self.buffer) < batch_size:
            return self.buffer.copy()

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)


class TrainingLogger:
    """
    Logger for training progress.

    Writes metrics to CSV and provides console output.
    """

    def __init__(self, log_dir: Path, verbose: bool = True):
        """
        Initialize logger.

        Args:
            log_dir: Directory for log files
            verbose: Whether to print to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

        # CSV file for metrics
        self.csv_path = self.log_dir / "training_log.csv"
        self.csv_file = None
        self.csv_writer = None

        # History
        self.history: list[dict] = []

    def start(self) -> None:
        """Start logging session."""
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "iteration",
            "timestamp",
            "games_played",
            "avg_reward",
            "best_reward",
            "policy_loss",
            "value_loss",
            "buffer_size",
        ])
        self.csv_file.flush()

    def log(self, metrics: dict) -> None:
        """Log metrics for an iteration."""
        self.history.append(metrics)

        # Write to CSV
        if self.csv_writer:
            self.csv_writer.writerow([
                metrics.get("iteration", 0),
                datetime.now().isoformat(),
                metrics.get("games_played", 0),
                metrics.get("avg_reward", 0),
                metrics.get("best_reward", 0),
                metrics.get("policy_loss", 0),
                metrics.get("value_loss", 0),
                metrics.get("buffer_size", 0),
            ])
            self.csv_file.flush()

        # Console output
        if self.verbose:
            print(f"\n=== Iteration {metrics.get('iteration', 0)} ===")
            print(f"  Games played: {metrics.get('games_played', 0)}")
            print(f"  Avg reward: {metrics.get('avg_reward', 0):.4f}")
            print(f"  Best reward: {metrics.get('best_reward', 0):.4f}")
            print(f"  Policy loss: {metrics.get('policy_loss', 0):.4f}")
            print(f"  Value loss: {metrics.get('value_loss', 0):.4f}")
            print(f"  Buffer size: {metrics.get('buffer_size', 0)}")

    def close(self) -> None:
        """Close log files."""
        if self.csv_file:
            self.csv_file.close()

        # Save full history as JSON
        history_path = self.log_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)


class AlphaZeroTrainer:
    """
    Trainer for AlphaZero-based calibration optimization.

    Coordinates self-play, training, and evaluation.
    """

    def __init__(self, config: AlphaZeroConfig | None = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration
        """
        self.config = config or AlphaZeroConfig()

        # Set random seed
        np.random.seed(self.config.seed)

        # Initialize components
        self.game = PlatinumGame(self.config)
        self.net = PolicyValueNet(self.config)
        self.replay_buffer = ReplayBuffer(self.config.training.buffer_size)
        self.logger = TrainingLogger(self.config.log_dir, self.config.verbose)

        # Best model tracking
        self.best_reward = float("-inf")
        self.best_state = None
        self.iterations_without_improvement = 0

    def self_play_game(self) -> tuple[list, float]:
        """
        Play a single self-play game.

        Returns:
            Tuple of (training_examples, final_reward)
        """
        player = MCTSPlayer(self.net, self.game.clone(), self.config)
        examples = player.self_play(
            temperature=self.config.mcts.temperature,
            temperature_threshold=self.config.mcts.temperature_threshold,
        )

        final_reward = examples[-1][2] if examples else 0.0

        return examples, final_reward

    def self_play_games(self, num_games: int) -> tuple[list, list]:
        """
        Play multiple self-play games.

        Args:
            num_games: Number of games to play

        Returns:
            Tuple of (all_examples, rewards)
        """
        all_examples = []
        rewards = []

        for i in range(num_games):
            examples, reward = self.self_play_game()
            all_examples.extend(examples)
            rewards.append(reward)

            if self.config.verbose and (i + 1) % 5 == 0:
                print(f"  Game {i + 1}/{num_games}, reward: {reward:.3f}")

        return all_examples, rewards

    def train_iteration(self, iteration: int) -> dict:
        """
        Run a single training iteration.

        Args:
            iteration: Current iteration number

        Returns:
            Metrics dictionary
        """
        start_time = time.time()

        # Self-play phase
        if self.config.verbose:
            print("\n--- Self-play phase ---")

        examples, rewards = self.self_play_games(
            self.config.training.games_per_iteration
        )

        # Add to replay buffer
        self.replay_buffer.add(examples)

        # Training phase
        if self.config.verbose:
            print("\n--- Training phase ---")

        if len(self.replay_buffer) >= self.config.training.min_buffer_size:
            train_stats = self.net.train_on_batch(
                self.replay_buffer.buffer,
                batch_size=self.config.training.batch_size,
                num_epochs=self.config.training.num_epochs,
            )
        else:
            train_stats = {"final_policy_loss": 0, "final_value_loss": 0}

        # Calculate metrics
        avg_reward = np.mean(rewards) if rewards else 0
        best_reward = np.max(rewards) if rewards else 0

        metrics = {
            "iteration": iteration,
            "games_played": len(rewards),
            "avg_reward": float(avg_reward),
            "best_reward": float(best_reward),
            "policy_loss": train_stats.get("final_policy_loss", 0),
            "value_loss": train_stats.get("final_value_loss", 0),
            "buffer_size": len(self.replay_buffer),
            "time_seconds": time.time() - start_time,
        }

        # Update best model
        if avg_reward > self.best_reward:
            self.best_reward = avg_reward
            self.iterations_without_improvement = 0
            self._save_best_model()
        else:
            self.iterations_without_improvement += 1

        return metrics

    def train(self) -> dict:
        """
        Run the full training loop.

        Returns:
            Final training statistics
        """
        self.logger.start()

        print(f"\n{'='*50}")
        print("AlchemistZero Training")
        print(f"{'='*50}")
        print(f"Iterations: {self.config.training.num_iterations}")
        print(f"Games per iteration: {self.config.training.games_per_iteration}")
        print(f"MCTS playouts: {self.config.mcts.n_playout}")
        print(f"Network parameters: {self.net.get_num_parameters():,}")
        print(f"{'='*50}\n")

        try:
            for iteration in range(1, self.config.training.num_iterations + 1):
                metrics = self.train_iteration(iteration)
                self.logger.log(metrics)

                # Checkpoint
                if iteration % self.config.training.checkpoint_interval == 0:
                    self._save_checkpoint(iteration)

                # Early stopping
                if metrics["avg_reward"] >= self.config.training.target_reward:
                    print(f"\nTarget reward {self.config.training.target_reward} reached!")
                    break

                if self.iterations_without_improvement >= self.config.training.patience:
                    print(f"\nEarly stopping: no improvement for {self.config.training.patience} iterations")
                    break

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")

        finally:
            self.logger.close()
            self._save_final_results()

        return {
            "best_reward": self.best_reward,
            "iterations_completed": len(self.logger.history),
            "final_buffer_size": len(self.replay_buffer),
        }

    def _save_best_model(self) -> None:
        """Save the best model checkpoint."""
        path = self.config.model_dir / "best_model.pth"
        self.net.save(path)

        if self.config.verbose:
            print(f"  New best model saved! Reward: {self.best_reward:.4f}")

    def _save_checkpoint(self, iteration: int) -> None:
        """Save a training checkpoint."""
        path = self.config.model_dir / f"checkpoint_{iteration:04d}.pth"
        self.net.save(path)

    def _save_final_results(self) -> None:
        """Save final results including the winning curve."""
        print("\n--- Generating final results ---")

        # Play evaluation games
        rewards = []
        best_state = None
        best_reward = float("-inf")

        for _ in range(5):
            game = PlatinumGame(self.config)
            player = MCTSPlayer(self.net, game, self.config)

            state = game.init_board()
            while not game.is_terminal(state):
                action = player.get_action(state, temperature=0.1)
                state, _ = game.get_next_state(state, action)

            reward = game.get_reward(state)
            rewards.append(reward)

            if reward > best_reward:
                best_reward = reward
                best_state = state

        print(f"Evaluation rewards: {[f'{r:.3f}' for r in rewards]}")
        print(f"Best reward: {best_reward:.4f}")

        # Export best curve to ACV
        if best_state is not None:
            from ptpd_calibration.alphazero.bridge.printing_env import PrintState

            print_state = PrintState.from_vector(
                best_state,
                self.config.state.num_density_steps,
            )

            acv_path = self.config.model_dir / "winning_curve.acv"
            export_to_acv(print_state.densities, acv_path)
            print(f"Winning curve exported to: {acv_path}")

            # Save action mapping
            action_mapping = {}
            from ptpd_calibration.alphazero.config import ActionType
            game = PlatinumGame(self.config)
            for i, _action_type in enumerate(ActionType):
                action_mapping[i] = game.get_action_description(i)

            mapping_path = self.config.model_dir / "action_mapping.json"
            with open(mapping_path, "w") as f:
                json.dump(action_mapping, f, indent=2)
            print(f"Action mapping saved to: {mapping_path}")

            # Save winning state
            state_path = self.config.model_dir / "winning_state.json"
            state_summary = game.get_state_summary(best_state)
            with open(state_path, "w") as f:
                json.dump(state_summary, f, indent=2)
            print(f"Winning state saved to: {state_path}")


def run_smoke_test() -> bool:
    """
    Run a quick smoke test of the training loop.

    Returns:
        True if test passes
    """
    print("\n=== AlphaZero Smoke Test ===\n")

    # Create minimal config for testing
    config = AlphaZeroConfig()
    config.training.num_iterations = 2
    config.training.games_per_iteration = 2
    config.training.max_moves_per_game = 10
    config.mcts.n_playout = 5
    config.training.min_buffer_size = 10
    config.verbose = True

    # Use temp directories
    import tempfile
    temp_dir = Path(tempfile.mkdtemp())
    config.model_dir = temp_dir / "models"
    config.log_dir = temp_dir / "logs"

    try:
        trainer = AlphaZeroTrainer(config)
        results = trainer.train()

        # Verify results
        if results["iterations_completed"] < 2:
            print(f"ERROR: Only {results['iterations_completed']} iterations completed")
            return False

        # Check files were created
        if not (config.model_dir / "best_model.pth").exists():
            print("ERROR: Best model not saved")
            return False

        if not (config.log_dir / "training_log.csv").exists():
            print("ERROR: Training log not created")
            return False

        print("\n=== Smoke Test PASSED ===")
        print(f"Best reward: {results['best_reward']:.4f}")
        print(f"Iterations: {results['iterations_completed']}")

        return True

    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(description="Train AlphaZero for Pt/Pd calibration")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--episodes", type=int, default=50, help="Training iterations")
    parser.add_argument("--games", type=int, default=25, help="Games per iteration")
    parser.add_argument("--smoke-test", action="store_true", help="Run quick smoke test")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    if args.smoke_test:
        success = run_smoke_test()
        exit(0 if success else 1)

    # Create config
    config = AlphaZeroConfig()
    config.mcts.n_playout = args.simulations
    config.training.num_iterations = args.episodes
    config.training.games_per_iteration = args.games
    config.device = args.device

    # Run training
    trainer = AlphaZeroTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
