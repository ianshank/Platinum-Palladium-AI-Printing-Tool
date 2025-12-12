"""
Curve adjustment based on paper characteristics and print feedback.

This module provides paper-specific QTR curve adjustments and
iterative refinement based on print scan analysis.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ptpd_calibration.papers.profiles import PaperProfile, BUILTIN_PAPERS


@dataclass
class CalibrationProfile:
    """
    Paper-specific calibration parameters derived from empirical testing.

    These values represent adjustments needed to compensate for paper
    absorption characteristics and achieve accurate tonal reproduction.
    """
    name: str
    absorption_factor: float = 1.0  # 1.0 = neutral, >1 = more absorbent
    highlight_boost: float = 0.0    # % boost for highlights (0-15% zone)
    midtone_boost: float = 0.0      # % boost for midtones (15-70% zone)
    shadow_boost: float = 0.0       # % boost for shadows (70-100% zone)
    dot_gain_curve: List[float] = field(default_factory=lambda: [1.0] * 7)
    notes: str = ""

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'absorption_factor': self.absorption_factor,
            'highlight_boost': self.highlight_boost,
            'midtone_boost': self.midtone_boost,
            'shadow_boost': self.shadow_boost,
            'dot_gain_curve': self.dot_gain_curve,
            'notes': self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationProfile':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_paper_profile(cls, paper: PaperProfile) -> 'CalibrationProfile':
        """
        Create calibration profile from existing paper profile.

        Uses paper characteristics to estimate calibration values.
        """
        # Map absorbency to absorption factor
        absorbency_map = {
            'low': 1.03,
            'medium': 1.06,
            'high': 1.08
        }
        absorption = absorbency_map.get(paper.absorbency, 1.05)

        # Map contrast tendency to boosts
        contrast_map = {
            'low': {'highlight': 0.02, 'midtone': 0.06, 'shadow': 0.05},
            'neutral': {'highlight': 0.02, 'midtone': 0.08, 'shadow': 0.06},
            'high': {'highlight': 0.01, 'midtone': 0.04, 'shadow': 0.05}
        }
        boosts = contrast_map.get(
            paper.characteristics.contrast_tendency,
            contrast_map['neutral']
        )

        # Adjust for high absorbency papers (need more midtone boost)
        if paper.absorbency == 'high':
            boosts['midtone'] += 0.02

        # Generate dot gain curve based on characteristics
        base_gain = absorption - 1.0
        dot_gain = [
            1.0 + base_gain * 0.3,   # Zone 0-15% (highlights)
            1.0 + base_gain * 0.6,   # Zone 15-30%
            1.0 + base_gain * 0.8,   # Zone 30-45%
            1.0 + base_gain * 1.0,   # Zone 45-55% (midtones)
            1.0 + base_gain * 1.0,   # Zone 55-70%
            1.0 + base_gain * 0.8,   # Zone 70-85%
            1.0 + base_gain * 0.7,   # Zone 85-100% (shadows)
        ]

        return cls(
            name=paper.name,
            absorption_factor=absorption,
            highlight_boost=boosts['highlight'],
            midtone_boost=boosts['midtone'],
            shadow_boost=boosts['shadow'],
            dot_gain_curve=dot_gain,
            notes=f"Auto-generated from {paper.name} profile"
        )


# Pre-defined calibration profiles based on empirical testing
# These values derived from actual print calibration sessions
CALIBRATION_PROFILES: Dict[str, CalibrationProfile] = {
    "arches_platine": CalibrationProfile(
        name="Arches Platine",
        absorption_factor=1.08,
        highlight_boost=0.02,
        midtone_boost=0.10,  # Key finding: needs aggressive midtone boost
        shadow_boost=0.07,
        dot_gain_curve=[1.02, 1.06, 1.08, 1.10, 1.10, 1.08, 1.07],
        notes="100% cotton, high absorbency. Midtones tend to appear muted without compensation."
    ),
    "revere_platinum": CalibrationProfile(
        name="Revere Platinum",
        absorption_factor=1.05,
        highlight_boost=0.02,
        midtone_boost=0.05,
        shadow_boost=0.06,
        dot_gain_curve=[1.02, 1.04, 1.05, 1.05, 1.05, 1.05, 1.06],
        notes="Good all-around paper with moderate absorption. Reliable baseline."
    ),
    "bergger_cot320": CalibrationProfile(
        name="Bergger COT320",
        absorption_factor=1.03,
        highlight_boost=0.01,
        midtone_boost=0.04,
        shadow_boost=0.05,
        dot_gain_curve=[1.01, 1.03, 1.04, 1.04, 1.04, 1.04, 1.05],
        notes="Less absorbent 320gsm cotton. Good for fine detail, holds highlights well."
    ),
    "hahnemuhle_platinum": CalibrationProfile(
        name="Hahnemuhle Platinum Rag",
        absorption_factor=1.06,
        highlight_boost=0.02,
        midtone_boost=0.06,
        shadow_boost=0.06,
        dot_gain_curve=[1.02, 1.04, 1.06, 1.06, 1.06, 1.05, 1.06],
        notes="German cotton rag, 300gsm. Consistent results, slightly cool base."
    ),
    "stonehenge": CalibrationProfile(
        name="Stonehenge",
        absorption_factor=1.04,
        highlight_boost=0.02,
        midtone_boost=0.05,
        shadow_boost=0.05,
        dot_gain_curve=[1.02, 1.04, 1.05, 1.05, 1.05, 1.05, 1.05],
        notes="Affordable option. Warm base, moderate absorption."
    ),
    "weston_diploma": CalibrationProfile(
        name="Weston Diploma Parchment",
        absorption_factor=1.07,
        highlight_boost=0.02,
        midtone_boost=0.08,
        shadow_boost=0.07,
        dot_gain_curve=[1.02, 1.05, 1.07, 1.08, 1.08, 1.07, 1.07],
        notes="Heavy 100% cotton. Good Dmax potential but needs midtone compensation."
    ),
    "custom": CalibrationProfile(
        name="Custom",
        absorption_factor=1.0,
        highlight_boost=0.0,
        midtone_boost=0.0,
        shadow_boost=0.0,
        dot_gain_curve=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        notes="User-defined profile. Start here for new papers."
    )
}


class QuadCurveParser:
    """Parse and write QuadTone RIP .quad curve files."""

    CHANNEL_ORDER = ['K', 'C', 'M', 'Y', 'LC', 'LM', 'LK', 'LLK', 'V', 'MK']

    @staticmethod
    def parse(filepath: str) -> Tuple[List[str], Dict[str, List[int]]]:
        """
        Parse a .quad file into header comments and curve data.

        Args:
            filepath: Path to .quad file

        Returns:
            Tuple of (header_lines, curves_dict)
        """
        with open(filepath, 'r') as f:
            lines = f.readlines()

        header = []
        curves: Dict[str, List[int]] = {}
        current_curve: Optional[str] = None
        curve_values: List[int] = []

        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                if 'Curve' in line:
                    if current_curve and curve_values:
                        curves[current_curve] = curve_values
                    # Extract channel name
                    current_curve = line.replace('# ', '').replace(' Curve', '')
                    curve_values = []
                else:
                    header.append(line)
            elif line and current_curve:
                try:
                    curve_values.append(int(line))
                except ValueError:
                    pass

        # Don't forget the last curve
        if current_curve and curve_values:
            curves[current_curve] = curve_values

        return header, curves

    @staticmethod
    def write(
        filepath: str,
        header: List[str],
        curves: Dict[str, List[int]],
        extra_comments: Optional[List[str]] = None
    ) -> None:
        """
        Write curves to a .quad file.

        Args:
            filepath: Output path
            header: Header comment lines
            curves: Dictionary of channel -> values
            extra_comments: Additional comments to include
        """
        with open(filepath, 'w') as f:
            # Write header
            for line in header:
                f.write(line + '\n')

            # Write extra comments if provided
            if extra_comments:
                for comment in extra_comments:
                    f.write(f'# {comment}\n')

            # Write curves in standard order
            for channel in QuadCurveParser.CHANNEL_ORDER:
                if channel in curves:
                    f.write(f'# {channel} Curve\n')
                    for val in curves[channel]:
                        f.write(f'{val}\n')


class CurveCalibrator:
    """
    Adjust QTR curves based on paper profiles and print feedback.

    This class implements the core calibration logic for adapting
    curves to different papers and refining based on print results.
    """

    def __init__(self, calibration_profile: Optional[CalibrationProfile] = None):
        """
        Initialize with optional calibration profile.

        Args:
            calibration_profile: CalibrationProfile instance or None for neutral
        """
        self.profile = calibration_profile or CALIBRATION_PROFILES['custom']

    def set_profile(self, profile_name: str) -> None:
        """
        Set calibration profile by name.

        Args:
            profile_name: Name of profile in CALIBRATION_PROFILES

        Raises:
            ValueError: If profile name not found
        """
        key = profile_name.lower().replace(" ", "_")
        if key in CALIBRATION_PROFILES:
            self.profile = CALIBRATION_PROFILES[key]
        else:
            raise ValueError(
                f"Unknown profile: {profile_name}. "
                f"Available: {list(CALIBRATION_PROFILES.keys())}"
            )

    def set_profile_from_paper(self, paper_name: str) -> None:
        """
        Set profile from built-in paper profile.

        Args:
            paper_name: Name of paper in BUILTIN_PAPERS
        """
        key = paper_name.lower().replace(" ", "_")

        # Check if we have a pre-defined calibration profile
        if key in CALIBRATION_PROFILES:
            self.profile = CALIBRATION_PROFILES[key]
            return

        # Otherwise generate from paper profile
        if key in BUILTIN_PAPERS:
            self.profile = CalibrationProfile.from_paper_profile(BUILTIN_PAPERS[key])
            return

        raise ValueError(f"Unknown paper: {paper_name}")

    def adjust_curve(
        self,
        curve_values: List[int],
        channel: str = 'K'
    ) -> List[int]:
        """
        Apply paper-specific adjustments to a single curve channel.

        The adjustment uses a position-dependent boost factor derived from
        the paper's dot gain curve and channel-specific modifiers.

        Args:
            curve_values: List of 256 values in 0-65535 range
            channel: Channel name ('K', 'C', 'M', 'Y', 'LK', 'LLK', etc.)

        Returns:
            Adjusted curve values
        """
        adjusted = []
        dot_gain = self.profile.dot_gain_curve
        n_zones = len(dot_gain) if dot_gain else 7

        # Use default curve if none specified
        if not dot_gain:
            dot_gain = [1.0] * n_zones

        for i, val in enumerate(curve_values):
            if val == 0:
                adjusted.append(0)
                continue

            # Position in tonal range (0 = paper white, 1 = max black)
            pos = i / 255.0

            # Determine zone index for dot gain lookup
            zone_idx = min(int(pos * n_zones), n_zones - 1)

            # Get base factor from dot gain curve
            factor = dot_gain[zone_idx]

            # Apply channel-specific modifiers
            # Different channels respond differently to paper absorption
            if channel == 'K':
                # Main density channel - use full adjustment
                pass
            elif channel in ['C', 'M']:
                # Color channels - slightly less aggressive (90%)
                factor = 1.0 + (factor - 1.0) * 0.9
            elif channel == 'Y':
                # Yellow UV blocking - conservative (40%)
                # Already typically boosted in base curve
                factor = 1.0 + (factor - 1.0) * 0.4
            elif channel in ['LK', 'LLK']:
                # Light inks for smooth transitions - moderate (60%)
                factor = 1.0 + (factor - 1.0) * 0.6
            else:
                # Unknown channel - no adjustment
                factor = 1.0

            # Apply adjustment with 16-bit clamping
            new_val = int(val * factor)
            adjusted.append(min(65535, new_val))

        return adjusted

    def adjust_all_curves(
        self,
        curves: Dict[str, List[int]]
    ) -> Dict[str, List[int]]:
        """
        Apply paper-specific adjustments to all channels.

        Args:
            curves: Dictionary mapping channel names to value lists

        Returns:
            Dictionary of adjusted curves
        """
        adjusted = {}
        for channel, values in curves.items():
            if all(v == 0 for v in values):
                # Unused channel - keep as zeros
                adjusted[channel] = values
            else:
                adjusted[channel] = self.adjust_curve(values, channel)
        return adjusted

    def adjust_from_feedback(
        self,
        curve_values: List[int],
        highlight_delta: float = 0.0,
        midtone_delta: float = 0.0,
        shadow_delta: float = 0.0,
        channel: str = 'K'
    ) -> List[int]:
        """
        Adjust curve based on print analysis feedback.

        This method applies targeted adjustments to specific tonal zones
        based on analysis of a test print. Use positive values to increase
        ink/density, negative to decrease.

        Args:
            curve_values: Original curve values
            highlight_delta: Adjustment for highlights (-0.15 to +0.15)
            midtone_delta: Adjustment for midtones (-0.15 to +0.15)
            shadow_delta: Adjustment for shadows (-0.15 to +0.15)
            channel: Channel name for channel-specific scaling

        Returns:
            Refined curve values
        """
        # Scale adjustments by channel
        channel_scale = {
            'K': 1.0,
            'C': 0.9,
            'M': 0.9,
            'Y': 0.4,
            'LK': 0.6,
            'LLK': 0.6
        }.get(channel, 1.0)

        h_adj = highlight_delta * channel_scale
        m_adj = midtone_delta * channel_scale
        s_adj = shadow_delta * channel_scale

        adjusted = []

        for i, val in enumerate(curve_values):
            if val == 0:
                adjusted.append(0)
                continue

            pos = i / 255.0

            # Smooth interpolation between zones using cosine blending
            if pos < 0.15:
                # Pure highlight zone
                factor = 1.0 + h_adj
            elif pos < 0.40:
                # Transition: highlight -> midtone
                blend = (pos - 0.15) / 0.25
                # Smooth cosine interpolation
                blend = 0.5 * (1 - math.cos(blend * math.pi))
                factor = 1.0 + h_adj * (1 - blend) + m_adj * blend
            elif pos < 0.70:
                # Pure midtone zone
                factor = 1.0 + m_adj
            elif pos < 0.85:
                # Transition: midtone -> shadow
                blend = (pos - 0.70) / 0.15
                blend = 0.5 * (1 - math.cos(blend * math.pi))
                factor = 1.0 + m_adj * (1 - blend) + s_adj * blend
            else:
                # Pure shadow zone
                factor = 1.0 + s_adj

            new_val = int(val * factor)
            adjusted.append(min(65535, max(0, new_val)))

        return adjusted

    def adjust_all_from_feedback(
        self,
        curves: Dict[str, List[int]],
        highlight_delta: float = 0.0,
        midtone_delta: float = 0.0,
        shadow_delta: float = 0.0
    ) -> Dict[str, List[int]]:
        """
        Apply feedback-based adjustments to all channels.

        Args:
            curves: Dictionary of channel -> values
            highlight_delta: Highlight adjustment
            midtone_delta: Midtone adjustment
            shadow_delta: Shadow adjustment

        Returns:
            Dictionary of adjusted curves
        """
        adjusted = {}
        for channel, values in curves.items():
            if all(v == 0 for v in values):
                adjusted[channel] = values
            else:
                adjusted[channel] = self.adjust_from_feedback(
                    values, highlight_delta, midtone_delta, shadow_delta, channel
                )
        return adjusted


def adjust_curve_for_paper(
    input_quad_path: str,
    output_quad_path: str,
    paper_name: str
) -> Dict[str, List[int]]:
    """
    One-step curve adjustment for a specific paper.

    Args:
        input_quad_path: Path to source .quad file
        output_quad_path: Path for adjusted .quad file
        paper_name: Paper profile name (e.g., 'arches_platine')

    Returns:
        Dictionary of adjusted curves
    """
    # Parse input
    header, curves = QuadCurveParser.parse(input_quad_path)

    # Get or create profile
    key = paper_name.lower().replace(" ", "_")
    if key in CALIBRATION_PROFILES:
        profile = CALIBRATION_PROFILES[key]
    elif key in BUILTIN_PAPERS:
        profile = CalibrationProfile.from_paper_profile(BUILTIN_PAPERS[key])
    else:
        raise ValueError(f"Unknown paper: {paper_name}")

    # Adjust
    calibrator = CurveCalibrator(profile)
    adjusted = calibrator.adjust_all_curves(curves)

    # Write output
    extra_comments = [
        f'Adjusted for {profile.name}',
        f'Highlights +{profile.highlight_boost*100:.0f}%, '
        f'Midtones +{profile.midtone_boost*100:.0f}%, '
        f'Shadows +{profile.shadow_boost*100:.0f}%'
    ]
    QuadCurveParser.write(output_quad_path, header, adjusted, extra_comments)

    return adjusted


def refine_curve_from_print(
    input_quad_path: str,
    output_quad_path: str,
    highlight_adj: float,
    midtone_adj: float,
    shadow_adj: float
) -> Dict[str, List[int]]:
    """
    Refine curve based on print feedback.

    Args:
        input_quad_path: Path to source .quad file
        output_quad_path: Path for refined .quad file
        highlight_adj: Highlight adjustment (-0.15 to +0.15)
        midtone_adj: Midtone adjustment (-0.15 to +0.15)
        shadow_adj: Shadow adjustment (-0.15 to +0.15)

    Returns:
        Dictionary of refined curves
    """
    header, curves = QuadCurveParser.parse(input_quad_path)

    calibrator = CurveCalibrator()
    refined = calibrator.adjust_all_from_feedback(
        curves, highlight_adj, midtone_adj, shadow_adj
    )

    extra_comments = [
        'Refined from print feedback',
        f'Adjustments: H={highlight_adj:+.0%}, M={midtone_adj:+.0%}, S={shadow_adj:+.0%}'
    ]
    QuadCurveParser.write(output_quad_path, header, refined, extra_comments)

    return refined


def get_available_calibration_profiles() -> List[str]:
    """Get list of available calibration profile names."""
    return list(CALIBRATION_PROFILES.keys())


def get_calibration_profile(name: str) -> Optional[CalibrationProfile]:
    """
    Get calibration profile by name.

    Args:
        name: Profile name or key

    Returns:
        CalibrationProfile or None
    """
    key = name.lower().replace(" ", "_")
    return CALIBRATION_PROFILES.get(key)
