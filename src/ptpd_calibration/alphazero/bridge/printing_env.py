"""
Printing environment wrapper for AlphaZero integration.

This module provides a simulator interface that wraps the Pt/Pd printing
tool's prediction logic, returning deterministic density predictions
for any given set of printing parameters.
"""

from dataclasses import dataclass

import numpy as np

from ptpd_calibration.alphazero.config import AlphaZeroConfig


@dataclass
class PrintState:
    """State representation for the printing process.

    Attributes:
        metal_ratio: Platinum to palladium ratio (0=Pd, 1=Pt)
        contrast_active: Whether contrast agent is active (0 or 1)
        contrast_amount: Amount of contrast agent (drops)
        exposure_time: Exposure time in seconds
        humidity: Environmental humidity percentage
        temperature: Environmental temperature in Celsius
        densities: Predicted density values for each step (21 steps)
    """

    metal_ratio: float
    contrast_active: float
    contrast_amount: float
    exposure_time: float
    humidity: float
    temperature: float
    densities: np.ndarray

    def to_vector(self) -> np.ndarray:
        """Convert state to feature vector."""
        params = np.array([
            self.metal_ratio,
            self.contrast_active,
            self.contrast_amount,
            self.exposure_time,
            self.humidity,
            self.temperature,
        ], dtype=np.float32)
        return np.concatenate([params, self.densities.astype(np.float32)])

    @classmethod
    def from_vector(cls, vector: np.ndarray, num_density_steps: int = 21) -> "PrintState":
        """Create state from feature vector."""
        return cls(
            metal_ratio=float(vector[0]),
            contrast_active=float(vector[1]),
            contrast_amount=float(vector[2]),
            exposure_time=float(vector[3]),
            humidity=float(vector[4]),
            temperature=float(vector[5]),
            densities=vector[6:6 + num_density_steps],
        )

    def copy(self) -> "PrintState":
        """Create a copy of this state."""
        return PrintState(
            metal_ratio=self.metal_ratio,
            contrast_active=self.contrast_active,
            contrast_amount=self.contrast_amount,
            exposure_time=self.exposure_time,
            humidity=self.humidity,
            temperature=self.temperature,
            densities=self.densities.copy(),
        )

    def is_valid(self) -> bool:
        """Check if state parameters are within valid ranges."""
        return (
            0.0 <= self.metal_ratio <= 1.0
            and self.contrast_active in (0.0, 1.0)
            and self.contrast_amount >= 0.0
            and self.exposure_time > 0.0
            and 0.0 <= self.humidity <= 100.0
            and -20.0 <= self.temperature <= 50.0
        )


class PrintingSimulator:
    """
    Simulator wrapper for Pt/Pd printing process.

    Provides a deterministic simulation of the printing process,
    predicting density values for any given set of parameters.
    This serves as the "environment" for the AlphaZero agent.
    """

    def __init__(
        self,
        config: AlphaZeroConfig | None = None,
        use_neural_predictor: bool = False,
    ):
        """
        Initialize the printing simulator.

        Args:
            config: AlphaZero configuration
            use_neural_predictor: If True, use the neural predictor from
                the printing tool. If False, use a physics-based mock.
        """
        self.config = config or AlphaZeroConfig()
        self.state_config = self.config.state
        self.reward_config = self.config.reward
        self.physics_config = self.config.physics
        self.use_neural_predictor = use_neural_predictor

        # Lazy-load the predictor if using neural mode
        self._predictor = None

        # Generate target linear response
        self._target_densities = self._generate_target_curve()

    def _generate_target_curve(self) -> np.ndarray:
        """Generate the target linear density curve.

        Returns:
            Array of target density values (21 steps from Dmin to Dmax)
        """
        num_steps = self.state_config.num_density_steps
        physics = self.physics_config
        return np.linspace(physics.target_dmin, physics.target_dmax, num_steps)

    def _get_predictor(self):
        """Lazy-load the neural predictor."""
        if self._predictor is None and self.use_neural_predictor:
            try:
                from ptpd_calibration.ml.deep.predictor import DeepCurvePredictor
                self._predictor = DeepCurvePredictor()
            except ImportError:
                # Fall back to physics-based model
                self._predictor = None
        return self._predictor

    def predict_densities(self, state: PrintState) -> np.ndarray:
        """
        Predict density values for a given print state.

        This is the core simulation function that models the physics
        of the printing process.

        Args:
            state: Current print state with parameters

        Returns:
            Array of predicted density values (21 steps)
        """
        if self.use_neural_predictor and self._get_predictor() is not None:
            return self._predict_with_neural(state)
        else:
            return self._predict_with_physics(state)

    def _predict_with_physics(self, state: PrintState) -> np.ndarray:
        """
        Physics-based density prediction using parametric model.

        Models the characteristic curve of Pt/Pd printing based on:
        - Metal ratio affects tone color and contrast
        - Contrast agent affects gamma
        - Exposure time affects overall density
        - Humidity affects coating absorption

        Args:
            state: Current print state

        Returns:
            Predicted density values
        """
        physics = self.physics_config
        num_steps = self.state_config.num_density_steps
        input_values = np.linspace(0, 1, num_steps)

        # Base gamma (tone curve exponent)
        # Platinum tends to be more contrasty than palladium
        base_gamma = physics.base_gamma + physics.gamma_metal_ratio_factor * state.metal_ratio

        # Contrast agent effect on gamma
        if state.contrast_active > self.state_config.contrast_active_threshold:
            gamma = base_gamma + physics.contrast_gamma_factor * state.contrast_amount
        else:
            gamma = base_gamma

        # Clamp gamma to reasonable range
        gamma = np.clip(gamma, physics.gamma_min, physics.gamma_max)

        # Paper base density (Dmin)
        dmin = physics.dmin_base + physics.dmin_humidity_factor * (state.humidity / 100.0)

        # Maximum density (Dmax) - affected by exposure and chemistry
        base_dmax = physics.dmax_base + physics.dmax_metal_ratio_factor * state.metal_ratio
        exposure_factor = 1.0 - np.exp(-state.exposure_time / physics.exposure_saturation_time)
        dmax = dmin + (base_dmax - dmin) * exposure_factor

        # Apply humidity effect on Dmax
        humidity_offset = (state.humidity - physics.humidity_center) / physics.humidity_center
        humidity_factor = 1.0 + physics.humidity_factor * humidity_offset
        dmax = dmax * np.clip(humidity_factor, physics.humidity_dmax_min, physics.humidity_dmax_max)

        # Generate characteristic curve
        # Using a modified power function with shoulder and toe
        shoulder_position = physics.shoulder_position
        toe_position = physics.toe_position

        # Apply gamma curve
        response = np.power(input_values, gamma)

        # Apply shoulder compression
        shoulder_mask = input_values > shoulder_position
        shoulder_compression = physics.shoulder_compression_factor * (input_values - shoulder_position) ** 2
        response[shoulder_mask] -= shoulder_compression[shoulder_mask]

        # Apply toe expansion
        toe_mask = input_values < toe_position
        toe_expansion = physics.toe_expansion_factor * (toe_position - input_values) ** 2
        response[toe_mask] += toe_expansion[toe_mask]

        # Scale to density range
        densities = dmin + (dmax - dmin) * np.clip(response, 0, 1)

        # Add small amount of realistic noise (deterministic based on params)
        noise_seed = int(state.metal_ratio * 1000 + state.exposure_time)
        rng = np.random.RandomState(noise_seed)
        noise = rng.normal(0, physics.physics_noise_std, num_steps)
        densities += noise

        return densities.astype(np.float32)

    def _predict_with_neural(self, state: PrintState) -> np.ndarray:
        """
        Neural network-based density prediction.

        Uses the trained neural predictor from the printing tool.

        Args:
            state: Current print state

        Returns:
            Predicted density values
        """
        predictor = self._get_predictor()
        if predictor is None:
            return self._predict_with_physics(state)

        try:
            from ptpd_calibration.core.models import CalibrationRecord
            from ptpd_calibration.core.types import ChemistryType, ContrastAgent

            # Convert state to CalibrationRecord
            record = CalibrationRecord(
                paper_type="default",
                chemistry_type=ChemistryType.PLATINUM_PALLADIUM,
                metal_ratio=state.metal_ratio,
                contrast_agent=(
                    ContrastAgent.NA2 if state.contrast_active > 0.5
                    else ContrastAgent.NONE
                ),
                contrast_amount=state.contrast_amount,
                exposure_time=state.exposure_time,
                humidity=state.humidity,
                temperature=state.temperature,
            )

            # Get prediction
            result = predictor.predict(record)
            return result.curve.astype(np.float32)

        except Exception:
            # Fall back to physics model on any error
            return self._predict_with_physics(state)

    def calculate_linearity_score(
        self,
        densities: np.ndarray,
        target: np.ndarray | None = None,
    ) -> float:
        """
        Calculate how linear the density response is.

        Linearity score is 1.0 for perfect linear response,
        decreasing toward 0.0 as deviation increases.

        Args:
            densities: Measured/predicted density values
            target: Target density values (default: linear ramp)

        Returns:
            Linearity score (0.0 to 1.0)
        """
        if target is None:
            target = self._target_densities

        # Ensure same length
        if len(densities) != len(target):
            # Interpolate to match
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(densities))
            x_new = np.linspace(0, 1, len(target))
            f = interp1d(x_old, densities, kind="linear", fill_value="extrapolate")
            densities = f(x_new)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((densities - target) ** 2))

        # Convert to score (1.0 at rmse=0, approaching 0 as rmse increases)
        # Using exponential decay with configurable characteristic scale
        linearity_score = np.exp(-rmse / self.reward_config.linearity_decay_scale)

        return float(linearity_score)

    def calculate_reward(self, state: PrintState) -> float:
        """
        Calculate reward for a given print state.

        Reward is based on:
        - Linearity of the density response (primary)
        - Monotonicity of the curve
        - Smoothness of the curve
        - Achieved density range

        Args:
            state: Current print state

        Returns:
            Reward value (0.0 to 1.0, or negative for invalid states)
        """
        # Check validity
        if not state.is_valid():
            return self.reward_config.invalid_penalty

        densities = state.densities

        # Linearity score (primary component)
        linearity = self.calculate_linearity_score(densities)

        # Monotonicity score
        diffs = np.diff(densities)
        num_increasing = np.sum(diffs > 0)
        monotonicity = num_increasing / max(len(diffs), 1)

        # Smoothness score (penalize large second derivatives)
        second_diff = np.diff(diffs)
        smoothness_penalty = self.reward_config.smoothness_penalty_factor
        smoothness = 1.0 / (1.0 + np.std(second_diff) * smoothness_penalty)

        # Density range score
        density_range = densities.max() - densities.min()
        target_range = self.reward_config.density_range_target
        range_score = min(1.0, density_range / target_range)

        # Combine components
        reward = (
            self.reward_config.rmse_weight * linearity
            + self.reward_config.monotonicity_weight * monotonicity
            + self.reward_config.smoothness_weight * smoothness
            + self.reward_config.density_range_weight * range_score
        )

        # Normalize by total weight
        total_weight = (
            self.reward_config.rmse_weight
            + self.reward_config.monotonicity_weight
            + self.reward_config.smoothness_weight
            + self.reward_config.density_range_weight
        )
        reward = reward / total_weight

        # Apply scaling
        reward = reward * self.reward_config.reward_scale

        return float(reward)

    def get_initial_state(
        self,
        metal_ratio: float | None = None,
        exposure_time: float | None = None,
        humidity: float | None = None,
        temperature: float | None = None,
    ) -> PrintState:
        """
        Create an initial print state.

        Args:
            metal_ratio: Initial metal ratio (0=Pd, 1=Pt), defaults from config
            exposure_time: Initial exposure time in seconds, defaults from config
            humidity: Environmental humidity, defaults from config
            temperature: Environmental temperature, defaults from config

        Returns:
            Initial PrintState with predicted densities
        """
        state_cfg = self.state_config
        state = PrintState(
            metal_ratio=metal_ratio if metal_ratio is not None else state_cfg.initial_metal_ratio,
            contrast_active=0.0,
            contrast_amount=0.0,
            exposure_time=exposure_time if exposure_time is not None else state_cfg.initial_exposure_time,
            humidity=humidity if humidity is not None else state_cfg.initial_humidity,
            temperature=temperature if temperature is not None else state_cfg.initial_temperature,
            densities=np.zeros(state_cfg.num_density_steps, dtype=np.float32),
        )

        # Predict initial densities
        state.densities = self.predict_densities(state)

        return state

    def step(
        self,
        state: PrintState,
        action_idx: int,
        action_delta: tuple[int, float],
    ) -> tuple[PrintState, float, bool]:
        """
        Apply an action to the state and return the new state.

        Args:
            state: Current print state
            action_idx: Index of the action taken
            action_delta: Tuple of (state_index, delta_value) for the action

        Returns:
            Tuple of (new_state, reward, done)
        """
        from ptpd_calibration.alphazero.config import ActionType

        # Get action type from index
        action_types = list(ActionType)
        if action_idx < 0 or action_idx >= len(action_types):
            return state, self.reward_config.invalid_penalty, True

        action = action_types[action_idx]

        # Check for terminal actions
        if action == ActionType.FINISH:
            reward = self.calculate_reward(state)
            return state, reward, True

        if action == ActionType.NO_OP:
            return state, 0.0, False

        # Create new state
        new_state = state.copy()
        state_idx, delta = action_delta

        # Apply action
        if action == ActionType.CONTRAST_TOGGLE:
            # Toggle contrast agent
            new_state.contrast_active = 1.0 - new_state.contrast_active
        elif state_idx >= 0:
            # Get current value
            state_vec = new_state.to_vector()
            current_value = state_vec[state_idx]
            new_value = current_value + delta

            # Apply to appropriate field
            if state_idx == 0:  # metal_ratio
                new_state.metal_ratio = float(np.clip(new_value, 0.0, 1.0))
            elif state_idx == 2:  # contrast_amount
                new_state.contrast_amount = float(max(0.0, new_value))
            elif state_idx == 3:  # exposure_time
                new_state.exposure_time = float(max(1.0, new_value))
            elif state_idx == 4:  # humidity
                new_state.humidity = float(np.clip(new_value, 0.0, 100.0))

        # Predict new densities
        new_state.densities = self.predict_densities(new_state)

        # Calculate intermediate reward (small positive for valid moves)
        reward = 0.01 if new_state.is_valid() else self.reward_config.invalid_penalty
        done = not new_state.is_valid()

        return new_state, reward, done


def check_simulator() -> bool:
    """
    Verify simulator returns valid densities.

    Returns:
        True if simulator is working correctly
    """
    sim = PrintingSimulator()
    state = sim.get_initial_state()

    # Check state validity
    if not state.is_valid():
        print("ERROR: Initial state is invalid")
        return False

    # Check density values
    densities = state.densities
    if len(densities) != 21:
        print(f"ERROR: Expected 21 densities, got {len(densities)}")
        return False

    if densities.min() < 0:
        print(f"ERROR: Negative density values: {densities.min()}")
        return False

    if densities.max() > 5:
        print(f"ERROR: Unrealistic density values: {densities.max()}")
        return False

    # Check reward calculation
    reward = sim.calculate_reward(state)
    if not 0.0 <= reward <= 1.0:
        print(f"ERROR: Reward out of range: {reward}")
        return False

    # Check linearity score
    linearity = sim.calculate_linearity_score(densities)
    if not 0.0 <= linearity <= 1.0:
        print(f"ERROR: Linearity score out of range: {linearity}")
        return False

    print("Simulator check passed!")
    print(f"  Initial state: metal_ratio={state.metal_ratio:.2f}, exposure={state.exposure_time:.1f}s")
    print(f"  Density range: {densities.min():.3f} - {densities.max():.3f}")
    print(f"  Linearity score: {linearity:.3f}")
    print(f"  Reward: {reward:.3f}")

    return True


if __name__ == "__main__":
    check_simulator()
