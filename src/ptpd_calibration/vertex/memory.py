"""
Memory Bank integration for persistent user profiles.

Provides cross-session memory for:
- User's printer/paper preferences
- Calibration history and drift tracking
- Successful recipe database per user
- Aesthetic preferences (warm/neutral/cool)

When deployed to Vertex AI Agent Engine, Memory Bank is automatic.
This module provides a local fallback for development and testing,
plus helpers for structuring user profile data.

Usage:
    from ptpd_calibration.vertex.memory import MemoryBankClient, UserProfile

    memory = MemoryBankClient(storage_path="~/.ptpd/memory")
    profile = memory.get_profile("user123")
    profile.update_preference("paper_type", "Hahnemühle Platinum Rag")
    memory.save_profile(profile)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ptpd_calibration.config import get_settings

logger = logging.getLogger(__name__)


class CalibrationSnapshot(BaseModel):
    """A snapshot of calibration data for memory tracking.

    Args:
        timestamp: When the calibration was performed.
        paper_type: Paper used.
        pt_pd_ratio: Metal ratio as string.
        exposure_seconds: Exposure time.
        dmin: Minimum density achieved.
        dmax: Maximum density achieved.
        density_range: Dmax - Dmin.
        notes: User notes about the session.
    """

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    paper_type: str = ""
    pt_pd_ratio: str = "50:50"
    exposure_seconds: float = 0.0
    dmin: float = 0.0
    dmax: float = 0.0
    density_range: float = 0.0
    notes: str = ""


class UserProfile(BaseModel):
    """Persistent user profile for the Pt/Pd printing assistant.

    Stores preferences, history, and successful configurations
    that persist across conversation sessions.

    Args:
        user_id: Unique user identifier.
        display_name: User's display name.
        preferences: Dict of user preferences (paper, chemistry, etc.).
        calibration_history: List of calibration snapshots.
        successful_recipes: List of recipes that produced good results.
        notes: Free-form user notes.
        created_at: Profile creation timestamp.
        updated_at: Last update timestamp.
    """

    user_id: str
    display_name: str = ""
    preferences: dict[str, Any] = Field(default_factory=dict)
    calibration_history: list[CalibrationSnapshot] = Field(default_factory=list)
    successful_recipes: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def update_preference(self, key: str, value: Any) -> None:
        """Update a user preference.

        Args:
            key: Preference key (e.g., "paper_type", "default_pt_pd_ratio").
            value: Preference value.
        """
        self.preferences[key] = value
        self.updated_at = datetime.now().isoformat()

    def add_calibration(self, snapshot: CalibrationSnapshot) -> None:
        """Add a calibration snapshot to history.

        Args:
            snapshot: CalibrationSnapshot to record.
        """
        self.calibration_history.append(snapshot)
        self.updated_at = datetime.now().isoformat()

    def add_successful_recipe(self, recipe: dict[str, Any]) -> None:
        """Record a recipe that produced good results.

        Args:
            recipe: Recipe dict with chemistry details and quality notes.
        """
        recipe["saved_at"] = datetime.now().isoformat()
        self.successful_recipes.append(recipe)
        self.updated_at = datetime.now().isoformat()

    def add_note(self, note: str) -> None:
        """Add a free-form note.

        Args:
            note: Note text to record.
        """
        timestamped = f"[{datetime.now().isoformat()}] {note}"
        self.notes.append(timestamped)
        self.updated_at = datetime.now().isoformat()

    def get_summary(self) -> str:
        """Get a natural language summary of the user's profile.

        Used to provide context to the AI assistant at session start.

        Returns:
            Summary string describing the user's preferences and history.
        """
        parts = []

        if self.display_name:
            parts.append(f"User: {self.display_name}")

        if self.preferences:
            pref_items = []
            for key, value in self.preferences.items():
                pref_items.append(f"  - {key}: {value}")
            parts.append("Preferences:\n" + "\n".join(pref_items))

        if self.calibration_history:
            recent = self.calibration_history[-3:]
            cal_items = []
            for cal in recent:
                cal_items.append(
                    f"  - {cal.paper_type}, {cal.pt_pd_ratio}, "
                    f"{cal.exposure_seconds}s, Dmax={cal.dmax:.2f}"
                )
            parts.append(
                f"Recent calibrations ({len(self.calibration_history)} total):\n"
                + "\n".join(cal_items)
            )

        if self.successful_recipes:
            parts.append(f"Saved recipes: {len(self.successful_recipes)}")

        return "\n\n".join(parts) if parts else "New user — no history yet."

    def detect_drift(self, threshold: float = 0.1) -> list[str]:
        """Detect calibration drift from recent history.

        Compares recent calibrations for the same paper/chemistry to
        identify shifts that may require re-calibration.

        Args:
            threshold: Density change threshold to flag as drift.

        Returns:
            List of drift warnings, empty if no drift detected.
        """
        if len(self.calibration_history) < 2:
            return []

        warnings = []
        # Group by paper type
        by_paper: dict[str, list[CalibrationSnapshot]] = {}
        for cal in self.calibration_history:
            by_paper.setdefault(cal.paper_type, []).append(cal)

        for paper, cals in by_paper.items():
            if len(cals) < 2:
                continue

            recent = cals[-2:]
            dmax_delta = abs(recent[1].dmax - recent[0].dmax)
            dmin_delta = abs(recent[1].dmin - recent[0].dmin)

            if dmax_delta > threshold:
                warnings.append(
                    f"Dmax drift on {paper}: {recent[0].dmax:.2f} -> {recent[1].dmax:.2f} "
                    f"(Δ{dmax_delta:.3f}). Consider re-calibrating."
                )
            if dmin_delta > threshold:
                warnings.append(
                    f"Dmin drift on {paper}: {recent[0].dmin:.2f} -> {recent[1].dmin:.2f} "
                    f"(Δ{dmin_delta:.3f}). Check for staining or clearing issues."
                )

        return warnings


class MemoryBankClient:
    """Local memory bank for development and testing.

    In production (Agent Engine), Memory Bank is provided automatically.
    This client provides a compatible local file-based fallback.

    Args:
        storage_path: Directory to store user profiles.
    """

    def __init__(self, storage_path: str | Path | None = None):
        settings = get_settings()
        self.storage_path = Path(storage_path or settings.data_dir / "memory")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, UserProfile] = {}

    def _profile_path(self, user_id: str) -> Path:
        """Get the file path for a user profile.

        Args:
            user_id: User identifier.

        Returns:
            Path to the profile JSON file.
        """
        safe_id = user_id.replace("/", "_").replace("\\", "_")
        return self.storage_path / f"{safe_id}.json"

    def get_profile(self, user_id: str) -> UserProfile:
        """Get or create a user profile.

        Args:
            user_id: User identifier.

        Returns:
            UserProfile for the given user.
        """
        if user_id in self._cache:
            return self._cache[user_id]

        path = self._profile_path(user_id)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            profile = UserProfile(**data)
        else:
            profile = UserProfile(user_id=user_id)

        self._cache[user_id] = profile
        return profile

    def save_profile(self, profile: UserProfile) -> None:
        """Save a user profile to disk.

        Args:
            profile: UserProfile to persist.
        """
        profile.updated_at = datetime.now().isoformat()
        path = self._profile_path(profile.user_id)
        path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
        self._cache[profile.user_id] = profile
        logger.debug("Saved profile for user %s", profile.user_id)

    def delete_profile(self, user_id: str) -> bool:
        """Delete a user profile.

        Args:
            user_id: User identifier.

        Returns:
            True if profile was deleted, False if not found.
        """
        path = self._profile_path(user_id)
        self._cache.pop(user_id, None)

        if path.exists():
            path.unlink()
            return True
        return False

    def list_profiles(self) -> list[str]:
        """List all stored user IDs.

        Returns:
            List of user IDs with stored profiles.
        """
        return [p.stem for p in self.storage_path.iterdir() if p.is_file() and p.suffix == ".json"]

    def get_context_for_session(self, user_id: str) -> str:
        """Get formatted context for injecting into an AI session.

        This generates a system prompt addendum with the user's
        preferences, recent history, and any drift warnings.

        Args:
            user_id: User identifier.

        Returns:
            Formatted context string for the AI assistant.
        """
        profile = self.get_profile(user_id)
        parts = [profile.get_summary()]

        drift_warnings = profile.detect_drift()
        if drift_warnings:
            parts.append("Calibration drift warnings:")
            for warning in drift_warnings:
                parts.append(f"  ⚠ {warning}")

        return "\n\n".join(parts)
