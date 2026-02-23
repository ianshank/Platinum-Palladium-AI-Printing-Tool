"""
Session log tab for the PTPD Calibration UI.

Displays print history timeline, calculates statistics, and generates
personalized best practices recommendations based on printing history.
"""

from collections import defaultdict
from pathlib import Path

import gradio as gr

from ptpd_calibration.session import PrintResult, SessionLogger


def calculate_session_statistics(session_logger: SessionLogger) -> dict:
    """Calculate comprehensive statistics from all sessions.

    Aggregates data across all sessions to provide insights on:
    - Overall success rates
    - Paper performance
    - Optimal exposure ranges
    - Chemistry patterns

    Args:
        session_logger: SessionLogger instance to query.

    Returns:
        Dictionary containing calculated statistics.
    """
    stats = {
        "total_prints": 0,
        "total_sessions": 0,
        "results_by_type": defaultdict(int),
        "papers_used": defaultdict(lambda: {"count": 0, "success": 0}),
        "exposure_times": [],
        "success_rate": 0.0,
        "avg_exposure": 0.0,
        "best_paper": None,
        "optimal_exposure_range": None,
        "chemistry_patterns": defaultdict(list),
    }

    try:
        sessions = session_logger.list_sessions(limit=100)
        stats["total_sessions"] = len(sessions)

        for sess_summary in sessions:
            try:
                session = session_logger.load_session(Path(sess_summary["filepath"]))
                for record in session.records:
                    stats["total_prints"] += 1
                    stats["results_by_type"][record.result.value] += 1

                    # Track paper performance
                    if record.paper_type:
                        stats["papers_used"][record.paper_type]["count"] += 1
                        if record.result in (
                            PrintResult.EXCELLENT,
                            PrintResult.GOOD,
                        ):
                            stats["papers_used"][record.paper_type]["success"] += 1

                    # Track exposure times for successful prints
                    if record.exposure_time_minutes > 0 and record.result in (
                        PrintResult.EXCELLENT,
                        PrintResult.GOOD,
                    ):
                        stats["exposure_times"].append(record.exposure_time_minutes)

                    # Track chemistry patterns for successful prints
                    if (
                        record.result in (PrintResult.EXCELLENT, PrintResult.GOOD)
                        and record.chemistry
                    ):
                        ratio = record.chemistry.platinum_ratio
                        stats["chemistry_patterns"]["platinum_ratio"].append(ratio)

            except Exception:
                continue

        # Calculate derived statistics
        if stats["total_prints"] > 0:
            successful = (
                stats["results_by_type"].get("excellent", 0)
                + stats["results_by_type"].get("good", 0)
                + stats["results_by_type"].get("acceptable", 0)
            )
            stats["success_rate"] = (successful / stats["total_prints"]) * 100

        if stats["exposure_times"]:
            stats["avg_exposure"] = sum(stats["exposure_times"]) / len(stats["exposure_times"])
            # Calculate optimal exposure range (middle 80% of successful exposures)
            sorted_times = sorted(stats["exposure_times"])
            n = len(sorted_times)
            if n >= 5:
                low_idx = int(n * 0.1)
                high_idx = int(n * 0.9)
                stats["optimal_exposure_range"] = (
                    sorted_times[low_idx],
                    sorted_times[high_idx],
                )

        # Find best performing paper
        best_success_rate = 0
        for paper, paper_stats in stats["papers_used"].items():
            if paper_stats["count"] >= 3:  # Need at least 3 prints for meaningful data
                paper_success_rate = paper_stats["success"] / paper_stats["count"]
                if paper_success_rate > best_success_rate:
                    best_success_rate = paper_success_rate
                    stats["best_paper"] = {
                        "name": paper,
                        "success_rate": paper_success_rate * 100,
                        "count": paper_stats["count"],
                    }

    except Exception:
        pass

    return stats


def generate_best_practices(stats: dict) -> str:
    """Generate best practices recommendations based on session statistics.

    Analyzes the user's printing history and provides personalized
    recommendations for improving their workflow.

    Args:
        stats: Statistics dictionary from calculate_session_statistics.

    Returns:
        Markdown-formatted best practices text.
    """
    if stats["total_prints"] < 5:
        return (
            "**Not enough data yet.**\n\n"
            "Log at least 5 prints to see personalized recommendations "
            "based on your printing history."
        )

    recommendations = []

    # Overall success rate feedback
    success_rate = stats["success_rate"]
    if success_rate >= 80:
        recommendations.append(
            f"**Excellent work!** Your success rate is {success_rate:.1f}%. "
            "Keep following your current workflow."
        )
    elif success_rate >= 60:
        recommendations.append(
            f"**Good progress!** Your success rate is {success_rate:.1f}%. "
            "Consider reviewing failed prints for patterns."
        )
    else:
        recommendations.append(
            f"**Room for improvement.** Your success rate is {success_rate:.1f}%. "
            "Focus on consistent chemistry preparation and exposure testing."
        )

    # Paper recommendation
    if stats["best_paper"]:
        bp = stats["best_paper"]
        recommendations.append(
            f"**Best paper:** {bp['name']} ({bp['success_rate']:.0f}% success rate "
            f"over {bp['count']} prints)"
        )

    # Exposure recommendation
    if stats["optimal_exposure_range"]:
        low, high = stats["optimal_exposure_range"]
        recommendations.append(
            f"**Optimal exposure range:** {low:.1f} - {high:.1f} minutes "
            f"(based on {len(stats['exposure_times'])} successful prints)"
        )
    elif stats["avg_exposure"] > 0:
        recommendations.append(f"**Average exposure time:** {stats['avg_exposure']:.1f} minutes")

    # Paper diversity insight
    num_papers = len(stats["papers_used"])
    if num_papers == 1:
        recommendations.append(
            "**Tip:** Consider testing additional papers to find your ideal match."
        )
    elif num_papers > 5:
        recommendations.append(
            f"**Great exploration!** You've tested {num_papers} different papers. "
            "Consider focusing on your top performers."
        )

    # Chemistry insights
    pt_ratios = stats["chemistry_patterns"].get("platinum_ratio", [])
    if len(pt_ratios) >= 5:
        avg_pt = sum(pt_ratios) / len(pt_ratios) * 100
        recommendations.append(f"**Typical Pt ratio:** {avg_pt:.0f}% platinum in successful prints")

    # Results breakdown
    results = stats["results_by_type"]
    if results:
        excellent = results.get("excellent", 0)
        failed = results.get("failed", 0)
        if excellent > 0 or failed > 0:
            recommendations.append(
                f"**Quality distribution:** {excellent} excellent, "
                f"{results.get('good', 0)} good, "
                f"{results.get('acceptable', 0)} acceptable, "
                f"{results.get('poor', 0)} poor, {failed} failed"
            )

    # Session frequency insight
    if stats["total_sessions"] > 0:
        prints_per_session = stats["total_prints"] / stats["total_sessions"]
        recommendations.append(
            f"**Productivity:** {prints_per_session:.1f} prints per session average"
        )

    return "\n\n".join(recommendations)


def build_session_log_tab(session_logger: SessionLogger):
    """Build the Session Log tab.

    Creates a UI tab with:
    - Timeline view of recent print sessions
    - Statistics sidebar
    - Best practices recommendations

    Args:
        session_logger: SessionLogger instance for data access.
    """
    with gr.TabItem("📓 Print Session Log"):
        with gr.Row():
            # Timeline view
            with gr.Column(scale=2):
                gr.Markdown("## Print History")

                # Filter controls
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh History")
                    _filter_paper = gr.Dropdown(label="Filter Paper", choices=["All"], value="All")

                # Timeline visualization
                timeline_view = gr.HTML(label="Timeline")

                # Hidden data store
                _session_data = gr.State([])  # Reserved for event handler

            # Stats sidebar
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Statistics")
                _stats_plot = gr.Plot(label="Success Rate Over Time")  # Reserved for event handler

                gr.Markdown("### 🏆 Best Practices")
                best_practices = gr.Markdown("Not enough data yet.")

        # Logic
        def render_timeline_html(sessions):
            """Render sessions as an HTML timeline."""
            if not sessions:
                return (
                    "<div style='padding: 20px; text-align: center; color: gray;'>"
                    "No print sessions found.</div>"
                )

            html = (
                "<div style='display: flex; flex-direction: column; gap: 10px; "
                "max-height: 600px; overflow-y: auto; padding-right: 10px;'>"
            )

            for sess_summary in sessions:
                try:
                    sess = session_logger.load_session(Path(sess_summary["filepath"]))
                    if not sess.records:
                        continue

                    date_str = sess.started_at.strftime("%Y-%m-%d")

                    html += f"""
                    <div style="background: var(--ptpd-card); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 12px;">
                        <div style="font-weight: bold; margin-bottom: 8px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 4px;">
                            {date_str} <span style="font-weight: normal; opacity: 0.7;">({len(sess.records)} prints)</span>
                        </div>
                    """

                    for rec in sess.records:
                        # Color based on result
                        color = "#4ade80"  # Green for excellent/good
                        if rec.result == PrintResult.ACCEPTABLE:
                            color = "#facc15"  # Yellow
                        elif rec.result == PrintResult.POOR:
                            color = "#f87171"  # Red
                        elif rec.result == PrintResult.FAILED:
                            color = "#ef4444"  # Darker red

                        html += f"""
                        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 6px; font-size: 0.9em;">
                            <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {color};" title="{rec.result.name}"></div>
                            <div style="font-weight: 500; width: 120px; overflow: hidden; text-overflow: ellipsis;">{rec.image_name}</div>
                            <div style="opacity: 0.8; flex: 1;">{rec.paper_type}</div>
                            <div style="opacity: 0.8;">{rec.exposure_time_minutes}m</div>
                        </div>
                        """

                    html += "</div>"
                except Exception:
                    continue

            html += "</div>"
            return html

        def refresh_data():
            """Refresh timeline, calculate stats, and generate recommendations."""
            try:
                sessions = session_logger.list_sessions(limit=20)
                html = render_timeline_html(sessions)

                # Calculate statistics and generate best practices
                stats = calculate_session_statistics(session_logger)
                practices_md = generate_best_practices(stats)

                return html, practices_md
            except Exception as e:
                return (
                    f"<div>Error loading history: {str(e)}</div>",
                    "Unable to calculate statistics.",
                )

        refresh_btn.click(refresh_data, outputs=[timeline_view, best_practices])
