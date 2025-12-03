import gradio as gr
from datetime import datetime, timedelta
from pathlib import Path
from ptpd_calibration.session import SessionLogger, PrintResult

def build_dashboard_tab(onboarding_state: gr.State, session_logger: SessionLogger):
    """
    Build the Dashboard tab.
    
    Args:
        onboarding_state: Gradio state for onboarding.
        session_logger: SessionLogger instance for retrieving stats.
    """
    with gr.TabItem("🏠 Dashboard"):
        gr.Markdown("## Platinum/Palladium Calibration Studio")
        gr.Markdown("Orchestrate calibration, chemistry, and AI workflows from a single hub.")

        def compute_dashboard_metrics():
            now = datetime.now()
            week_ago = now - timedelta(days=7)
            summary_rows = []
            total_records = 0
            recent_records = 0
            success_count = 0
            total_hours = 0.0
            active_curve = "Not set"

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
                    success_pct = (
                        (successful / session_records) * 100 if session_records else 0
                    )
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
                            recent_records += 1
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

            overall_success_rate = (
                (success_count / total_records * 100) if total_records else 0.0
            )

            return (
                recent_records,
                f"{overall_success_rate:.1f}%",
                active_curve,
                f"{total_hours:.1f}h",
                summary_rows,
            )

        with gr.Row():
            # Quick Stats
            with gr.Column(scale=1):
                gr.Markdown("### 📈 Session Stats")
                with gr.Row():
                    prints_today = gr.Number(label="Prints (7d)", value=0, interactive=False)
                    success_rate = gr.Number(label="Success Rate", value=0, interactive=False)
                with gr.Row():
                    active_curve_disp = gr.Textbox(label="Active Curve", interactive=False)
                    session_time = gr.Textbox(label="Total Time", interactive=False)

            # Quick Actions
            with gr.Column(scale=2):
                gr.Markdown("### ⚡ Quick Actions")
                with gr.Row():
                    btn_scan = gr.Button("📷 Read Step Tablet", variant="primary")
                    btn_chem = gr.Button("📐 Calculate Chemistry")
                    btn_expo = gr.Button("⏱️ Exposure Calculator")
                with gr.Row():
                    btn_neg = gr.Button("🖼️ Create Negative")
                    btn_ai = gr.Button("💬 Ask AI Assistant")

        # Recent Activity
        gr.Markdown("### 🕐 Recent Sessions")
        recent_sessions = gr.Dataframe(
            headers=["Date", "Paper", "Prints", "Success Rate"],
            interactive=False,
            value=[],
        )

        # Auto-refresh dashboard on load
        dashboard_timer = gr.Timer(value=10) # Refresh every 10 seconds or on load
        
        dashboard_timer.tick(
            compute_dashboard_metrics,
            outputs=[
                prints_today,
                success_rate,
                active_curve_disp,
                session_time,
                recent_sessions,
            ],
        )

        # Wire up Quick Action buttons (need to know tab IDs to switch tabs)
        # For now we just return the code. In main app we can link them using Javascript or Tab selection if Gradio supports it (it does via selected=Index)
        # But easier is to use js to click the tab button.
        
        # Helper to generate JS click
        def jump_js(tab_index):
            return f"() => {{ document.querySelectorAll('.main-tabs button')[{tab_index}].click(); }}"

        # Assuming tab indices: 0=Dashboard, 1=Calibration, 2=Image Prep, 3=Darkroom, 4=AI
        # And subtabs... this is tricky. Gradio doesn't easily support deep linking to subtabs.
        # We'll just link to main tabs for now.
        
        btn_scan.click(None, None, None, js=jump_js(1)) # Calibration
        btn_chem.click(None, None, None, js=jump_js(3)) # Darkroom
        btn_expo.click(None, None, None, js=jump_js(3)) # Darkroom
        btn_neg.click(None, None, None, js=jump_js(2)) # Image Prep
        btn_ai.click(None, None, None, js=jump_js(4))   # AI

