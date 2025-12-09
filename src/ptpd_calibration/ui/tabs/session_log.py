"""
Session log tab for tracking print history and statistics.

Provides timeline view, success rate tracking, and best practice recommendations.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gradio as gr

from ptpd_calibration.session import (
    ChemistryUsed,
    PrintRecord,
    PrintResult,
    PrintSession,
    SessionLogger,
)


@dataclass
class SessionStatistics:
    """Computed statistics from print sessions."""

    total_prints: int = 0
    success_count: int = 0
    acceptable_count: int = 0
    poor_count: int = 0
    failed_count: int = 0

    # Paper statistics
    paper_success_rates: dict[str, tuple[int, int]] = None  # paper -> (success, total)
    best_paper: Optional[str] = None

    # Exposure statistics
    avg_exposure_minutes: float = 0.0
    best_exposure_range: tuple[float, float] = (0.0, 0.0)

    # Curve statistics
    most_used_curve: Optional[str] = None
    curve_success_rates: dict[str, tuple[int, int]] = None

    # Recommendations
    recommendations: list[str] = None

    def __post_init__(self):
        if self.paper_success_rates is None:
            self.paper_success_rates = {}
        if self.curve_success_rates is None:
            self.curve_success_rates = {}
        if self.recommendations is None:
            self.recommendations = []

    @property
    def success_rate(self) -> float:
        """Calculate overall success rate (excellent + good)."""
        if self.total_prints == 0:
            return 0.0
        return (self.success_count / self.total_prints) * 100

    @property
    def acceptable_rate(self) -> float:
        """Calculate acceptable rate."""
        if self.total_prints == 0:
            return 0.0
        return (self.acceptable_count / self.total_prints) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate (poor + failed)."""
        if self.total_prints == 0:
            return 0.0
        return ((self.poor_count + self.failed_count) / self.total_prints) * 100


def compute_session_statistics(
    session_logger: SessionLogger,
    max_sessions: int = 50,
) -> SessionStatistics:
    """
    Compute statistics from recent print sessions.

    Args:
        session_logger: Session logger instance
        max_sessions: Maximum number of sessions to analyze

    Returns:
        SessionStatistics with computed metrics
    """
    stats = SessionStatistics()
    all_records: list[PrintRecord] = []

    # Collect records from recent sessions
    try:
        sessions = session_logger.list_sessions(limit=max_sessions)
        for sess_summary in sessions:
            try:
                session = session_logger.load_session(Path(sess_summary["filepath"]))
                if session and session.records:
                    all_records.extend(session.records)
            except Exception:
                continue
    except Exception:
        return stats

    if not all_records:
        return stats

    stats.total_prints = len(all_records)

    # Count results by category
    paper_results: dict[str, list[PrintResult]] = defaultdict(list)
    curve_results: dict[str, list[PrintResult]] = defaultdict(list)
    exposures_by_result: dict[PrintResult, list[float]] = defaultdict(list)

    for record in all_records:
        result = record.result

        # Count by result type
        if result in (PrintResult.EXCELLENT, PrintResult.GOOD):
            stats.success_count += 1
        elif result == PrintResult.ACCEPTABLE:
            stats.acceptable_count += 1
        elif result == PrintResult.POOR:
            stats.poor_count += 1
        elif result == PrintResult.FAILED:
            stats.failed_count += 1

        # Track by paper
        if record.paper_type:
            paper_results[record.paper_type].append(result)

        # Track by curve
        if record.curve_name:
            curve_results[record.curve_name].append(result)

        # Track exposures
        if record.exposure_time_minutes > 0:
            exposures_by_result[result].append(record.exposure_time_minutes)

    # Compute paper success rates
    for paper, results in paper_results.items():
        successes = sum(
            1 for r in results if r in (PrintResult.EXCELLENT, PrintResult.GOOD)
        )
        stats.paper_success_rates[paper] = (successes, len(results))

    # Find best paper
    if stats.paper_success_rates:
        # Calculate success rate, require at least 3 prints for statistical significance
        best_paper = max(
            (
                (paper, successes / total if total >= 3 else 0)
                for paper, (successes, total) in stats.paper_success_rates.items()
            ),
            key=lambda x: x[1],
            default=(None, 0),
        )
        if best_paper[1] > 0:
            stats.best_paper = best_paper[0]

    # Compute curve success rates
    for curve, results in curve_results.items():
        successes = sum(
            1 for r in results if r in (PrintResult.EXCELLENT, PrintResult.GOOD)
        )
        stats.curve_success_rates[curve] = (successes, len(results))

    # Find most used curve
    if curve_results:
        stats.most_used_curve = max(curve_results.keys(), key=lambda c: len(curve_results[c]))

    # Compute exposure statistics
    all_exposures = [
        record.exposure_time_minutes
        for record in all_records
        if record.exposure_time_minutes > 0
    ]
    if all_exposures:
        stats.avg_exposure_minutes = sum(all_exposures) / len(all_exposures)

    # Find best exposure range (from successful prints)
    success_exposures = exposures_by_result.get(
        PrintResult.EXCELLENT, []
    ) + exposures_by_result.get(PrintResult.GOOD, [])
    if success_exposures:
        stats.best_exposure_range = (min(success_exposures), max(success_exposures))

    # Generate recommendations
    stats.recommendations = _generate_recommendations(stats, all_records)

    return stats


def _generate_recommendations(
    stats: SessionStatistics,
    records: list[PrintRecord],
) -> list[str]:
    """Generate best practice recommendations based on statistics."""
    recommendations = []

    # Success rate recommendations
    if stats.total_prints >= 10:
        if stats.success_rate >= 80:
            recommendations.append(
                "🏆 Excellent consistency! Your process is well-dialed in."
            )
        elif stats.success_rate >= 60:
            recommendations.append(
                "📈 Good progress. Review your failed prints for common patterns."
            )
        elif stats.success_rate < 40:
            recommendations.append(
                "⚠️ High failure rate. Consider reviewing calibration and chemistry."
            )

    # Paper recommendations
    if stats.best_paper and stats.paper_success_rates:
        best_rate = stats.paper_success_rates.get(stats.best_paper, (0, 0))
        if best_rate[1] >= 3 and best_rate[0] / best_rate[1] >= 0.7:
            recommendations.append(
                f"📝 {stats.best_paper} shows best results "
                f"({best_rate[0]}/{best_rate[1]} successful)."
            )

    # Exposure recommendations
    if stats.best_exposure_range[0] > 0:
        low, high = stats.best_exposure_range
        if high - low <= 2:
            recommendations.append(
                f"⏱️ Consistent exposure sweet spot: {low:.1f}-{high:.1f} minutes."
            )
        elif high - low > 5:
            recommendations.append(
                "⚠️ Wide exposure variation in successful prints. "
                "Consider standardizing."
            )

    # Curve recommendations
    if stats.most_used_curve and stats.curve_success_rates:
        curve_rate = stats.curve_success_rates.get(stats.most_used_curve, (0, 0))
        if curve_rate[1] >= 5:
            rate_pct = (curve_rate[0] / curve_rate[1]) * 100
            recommendations.append(
                f"📊 Most used curve: {stats.most_used_curve} "
                f"({rate_pct:.0f}% success rate)."
            )

    # Environmental recommendations
    humidity_issues = sum(
        1
        for r in records
        if r.humidity_percent
        and (r.humidity_percent < 40 or r.humidity_percent > 70)
        and r.result in (PrintResult.POOR, PrintResult.FAILED)
    )
    if humidity_issues >= 3:
        recommendations.append(
            "💧 Several failures correlated with humidity extremes. "
            "Aim for 45-65% RH."
        )

    if not recommendations:
        recommendations.append(
            "📊 Keep logging prints to build process insights!"
        )

    return recommendations


def format_statistics_markdown(stats: SessionStatistics) -> str:
    """Format statistics as markdown for display."""
    if stats.total_prints == 0:
        return "Not enough data yet. Log some prints to see statistics!"

    lines = [
        f"**Total Prints:** {stats.total_prints}",
        f"**Success Rate:** {stats.success_rate:.1f}%",
        "",
        "**Results Breakdown:**",
        f"- ✅ Excellent/Good: {stats.success_count}",
        f"- 🟡 Acceptable: {stats.acceptable_count}",
        f"- ❌ Poor/Failed: {stats.poor_count + stats.failed_count}",
    ]

    if stats.best_paper:
        lines.extend(["", f"**Best Paper:** {stats.best_paper}"])

    if stats.avg_exposure_minutes > 0:
        lines.extend(["", f"**Avg Exposure:** {stats.avg_exposure_minutes:.1f} min"])

    if stats.most_used_curve:
        lines.extend(["", f"**Most Used Curve:** {stats.most_used_curve}"])

    return "\n".join(lines)


def format_recommendations_markdown(stats: SessionStatistics) -> str:
    """Format recommendations as markdown."""
    if not stats.recommendations:
        return "Log more prints to receive personalized recommendations!"

    return "\n\n".join(stats.recommendations)


def build_session_log_tab(session_logger: SessionLogger):
    """Build the Session Log tab."""
    with gr.TabItem("📓 Print Session Log"):
        with gr.Row():
            # Timeline view
            with gr.Column(scale=2):
                gr.Markdown("## Print History")

                # Filter controls
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh History")
                    filter_paper = gr.Dropdown(
                        label="Filter Paper",
                        choices=["All"],
                        value="All",
                    )

                # Timeline visualization
                timeline_view = gr.HTML(label="Timeline")

                # Hidden data store
                session_data = gr.State([])

            # Stats sidebar
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Statistics")
                stats_display = gr.Markdown("Loading statistics...")

                gr.Markdown("### 🏆 Best Practices")
                best_practices = gr.Markdown("Loading recommendations...")

        def render_timeline_html(sessions: list[dict]) -> str:
            """Render session timeline as HTML."""
            if not sessions:
                return (
                    "<div style='padding: 20px; text-align: center; color: gray;'>"
                    "No print sessions found."
                    "</div>"
                )

            html_parts = [
                "<div style='display: flex; flex-direction: column; gap: 10px; "
                "max-height: 600px; overflow-y: auto; padding-right: 10px;'>"
            ]

            # Result color mapping
            result_colors = {
                PrintResult.EXCELLENT: "#4ade80",  # Green
                PrintResult.GOOD: "#4ade80",  # Green
                PrintResult.ACCEPTABLE: "#facc15",  # Yellow
                PrintResult.POOR: "#f87171",  # Red
                PrintResult.FAILED: "#ef4444",  # Deeper red
            }

            for sess_summary in sessions:
                try:
                    sess = session_logger.load_session(Path(sess_summary["filepath"]))
                    if not sess or not sess.records:
                        continue

                    date_str = sess.started_at.strftime("%Y-%m-%d")
                    record_count = len(sess.records)

                    html_parts.append(
                        f"""
                        <div style="background: var(--ptpd-card);
                            border: 1px solid rgba(255,255,255,0.1);
                            border-radius: 8px; padding: 12px;">
                            <div style="font-weight: bold; margin-bottom: 8px;
                                border-bottom: 1px solid rgba(255,255,255,0.1);
                                padding-bottom: 4px;">
                                {date_str}
                                <span style="font-weight: normal; opacity: 0.7;">
                                    ({record_count} prints)
                                </span>
                            </div>
                        """
                    )

                    for rec in sess.records:
                        color = result_colors.get(rec.result, "#888888")
                        image_name = rec.image_name or "Untitled"
                        paper_type = rec.paper_type or "Unknown paper"
                        exposure = f"{rec.exposure_time_minutes}m" if rec.exposure_time_minutes else "N/A"

                        html_parts.append(
                            f"""
                            <div style="display: flex; align-items: center; gap: 10px;
                                margin-bottom: 6px; font-size: 0.9em;">
                                <div style="width: 10px; height: 10px; border-radius: 50%;
                                    background-color: {color};"
                                    title="{rec.result.name}">
                                </div>
                                <div style="font-weight: 500; width: 120px;
                                    overflow: hidden; text-overflow: ellipsis;">
                                    {image_name}
                                </div>
                                <div style="opacity: 0.8; flex: 1;">{paper_type}</div>
                                <div style="opacity: 0.8;">{exposure}</div>
                            </div>
                            """
                        )

                    html_parts.append("</div>")
                except Exception:
                    continue

            html_parts.append("</div>")
            return "".join(html_parts)

        def refresh_data():
            """Refresh timeline and statistics."""
            try:
                sessions = session_logger.list_sessions(limit=20)
                html = render_timeline_html(sessions)

                # Compute and format statistics
                stats = compute_session_statistics(session_logger)
                stats_md = format_statistics_markdown(stats)
                recommendations_md = format_recommendations_markdown(stats)

                return html, stats_md, recommendations_md
            except Exception as e:
                error_html = f"<div>Error loading history: {str(e)}</div>"
                return error_html, "Error loading stats", "Error loading recommendations"

        refresh_btn.click(
            refresh_data,
            outputs=[timeline_view, stats_display, best_practices],
        )
