from datetime import datetime, timedelta
from pathlib import Path

import gradio as gr

from ptpd_calibration.session import PrintResult, SessionLogger


def build_dashboard_tab(onboarding_state: gr.State, session_logger: SessionLogger):
    """
    Build the Dashboard tab.

    Args:
        onboarding_state: Gradio state for onboarding.
        session_logger: SessionLogger instance for retrieving stats.
    """
    with gr.TabItem("🏠 Dashboard"):
        # Hero Section
        with gr.Row():
            gr.Markdown(
                """
                # Platinum/Palladium Calibration Studio
                ### Professional Darkroom Workflow
                """
            )

        def make_stat_card(label, value):
            return f"""
            <div class="stat-card">
                <div class="stat-value">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """

        def compute_dashboard_metrics():
            now = datetime.now()
            week_ago = now - timedelta(days=7)
            summary_rows = []
            total_records = 0
            recent_count = 0
            success_count = 0
            total_hours = 0.0
            active_curve = "None"

            try:
                sessions = session_logger.list_sessions(limit=5)
            except Exception:
                sessions = []

            for summary in sessions:
                try:
                    session = session_logger.load_session(Path(summary["filepath"]))
                except Exception:
                    continue

                duration = session.duration_hours or 0.0
                total_hours += duration

                if session.records:
                    session_records = len(session.records)
                    total_records += session_records
                    last_record = session.records[-1]
                    if last_record.curve_name:
                        active_curve = last_record.curve_name

                    successful = sum(
                        1
                        for record in session.records
                        if record.result
                        in (PrintResult.EXCELLENT, PrintResult.GOOD, PrintResult.ACCEPTABLE)
                    )
                    success_pct = (successful / session_records) * 100 if session_records else 0
                    summary_rows.append(
                        [
                            session.records[0].timestamp.strftime("%b %d"),
                            session.records[0].paper_type or "Unknown",
                            session_records,
                            f"{success_pct:.1f}%",
                        ]
                    )

                    for record in session.records:
                        if record.timestamp >= week_ago:
                            recent_count += 1
                        if record.result in (
                            PrintResult.EXCELLENT,
                            PrintResult.GOOD,
                            PrintResult.ACCEPTABLE,
                        ):
                            success_count += 1
                else:
                    summary_rows.append(
                        [
                            summary.get("started_at", "")[:10],
                            "—",
                            0,
                            "0%",
                        ]
                    )

            overall_success_rate = (success_count / total_records * 100) if total_records else 0.0

            return (
                make_stat_card("Print Volume (7d)", recent_count),
                make_stat_card("Success Rate", f"{overall_success_rate:.1f}%"),
                make_stat_card("Active Curve", active_curve),
                make_stat_card("Total Lab Time", f"{total_hours:.1f}h"),
                summary_rows,
            )

        with gr.Group(elem_classes="ptpd-card"):
            gr.Markdown("### 📈 Lab Statistics")
            with gr.Row():
                stat_prints = gr.HTML(make_stat_card("Print Volume (7d)", "-"))
                stat_success = gr.HTML(make_stat_card("Success Rate", "-"))
                stat_curve = gr.HTML(make_stat_card("Active Curve", "-"))
                stat_time = gr.HTML(make_stat_card("Total Lab Time", "-"))

        with gr.Group(elem_classes="ptpd-card"):
            gr.Markdown("### ⚡ Quick Actions")
            with gr.Row():
                btn_scan = gr.Button("📷 Read Step Tablet", variant="primary")
                btn_chem = gr.Button("📐 Calculate Chemistry")
                btn_expo = gr.Button("⏱️ Exposure Calculator")
            with gr.Row():
                btn_neg = gr.Button("🖼️ Create Negative")
                btn_ai = gr.Button("💬 Ask AI Assistant")

        with gr.Group(elem_classes="ptpd-card"):
            gr.Markdown("### 🕐 Recent Sessions")
            recent_sessions = gr.Dataframe(
                headers=["Date", "Paper", "Prints", "Success Rate"],
                interactive=False,
                value=[],
            )

        # Auto-refresh dashboard on load
        dashboard_timer = gr.Timer(value=10)  # Refresh every 10 seconds or on load

        dashboard_timer.tick(
            compute_dashboard_metrics,
            outputs=[
                stat_prints,
                stat_success,
                stat_curve,
                stat_time,
                recent_sessions,
            ],
        )

        # Helper to generate JS click
        def jump_js(tab_index):
            # Using querySelectorAll is brittle if DOM changes, but works for now.
            return (
                f"() => {{ document.querySelectorAll('.main-tabs button')[{tab_index}].click(); }}"
            )

        btn_scan.click(None, None, None, js=jump_js(1))  # Calibration
        # Darkroom tab might be at index 3 or 4 depending on tab order
        btn_chem.click(
            None, None, None, js=jump_js(4)
        )  # Chemistry (Assuming separate tab now or subtab)
        btn_expo.click(None, None, None, js=jump_js(4))
        btn_neg.click(None, None, None, js=jump_js(3))  # Image Prep
        btn_ai.click(None, None, None, js=jump_js(6))  # AI Assistant
