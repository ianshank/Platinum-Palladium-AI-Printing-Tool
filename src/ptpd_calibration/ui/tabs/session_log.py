from pathlib import Path

import gradio as gr

from ptpd_calibration.session import PrintResult, SessionLogger


def build_session_log_tab(session_logger: SessionLogger) -> None:
    """Build the Session Log tab."""
    with gr.TabItem("📓 Print Session Log"):
        with gr.Row():
            # Timeline view
            with gr.Column(scale=2):
                gr.Markdown("## Print History")

                # Filter controls
                with gr.Row():
                    # Note: DateRange not standard in all Gradio versions, using Textbox for simplicity or 2 DatePickers if available.
                    # We'll just use a "Refresh" button for now.
                    refresh_btn = gr.Button("🔄 Refresh History")
                    gr.Dropdown(label="Filter Paper", choices=["All"], value="All")

                # Timeline visualization
                timeline_view = gr.HTML(label="Timeline")

                # Hidden data store
                gr.State([])

            # Stats sidebar
            with gr.Column(scale=1):
                gr.Markdown("### 📊 Statistics")
                gr.Plot(label="Success Rate Over Time")

                gr.Markdown("### 🏆 Best Practices")
                best_practices = gr.Markdown("Not enough data yet.")

        # Logic
        def render_timeline_html(sessions: list[dict]) -> str:
            if not sessions:
                return "<div style='padding: 20px; text-align: center; color: gray;'>No print sessions found.</div>"

            html = "<div style='display: flex; flex-direction: column; gap: 10px; max-height: 600px; overflow-y: auto; padding-right: 10px;'>"

            for sess_summary in sessions:
                # We might need to load full session to get records
                try:
                    sess = session_logger.load_session(Path(sess_summary['filepath']))
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
                        color = "#4ade80" # Green
                        if rec.result == PrintResult.ACCEPTABLE: color = "#facc15" # Yellow
                        elif rec.result == PrintResult.POOR: color = "#f87171" # Red
                        elif rec.result == PrintResult.FAILED: color = "#ef4444" # Redder

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

        def refresh_data() -> tuple[str, dict]:
            try:
                sessions = session_logger.list_sessions(limit=20)
                html = render_timeline_html(sessions)

                # TODO: Calculate stats and best practices
                return html, gr.update()
            except Exception as e:
                return f"<div>Error loading history: {str(e)}</div>", gr.update()

        refresh_btn.click(refresh_data, outputs=[timeline_view, best_practices])

        # Initial load
        # gr.Timer(1).tick(refresh_data, outputs=[timeline_view, best_practices]) # Timer not always reliable for init, check main app

