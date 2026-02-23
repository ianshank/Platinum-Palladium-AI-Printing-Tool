from pathlib import Path

import gradio as gr

from ptpd_calibration.llm import create_assistant


def build_ai_assistant_tab() -> None:
    """Build the AI Assistant tab."""
    with gr.TabItem("🤖 AI Assistant"):
        gr.Markdown(
            """
            ### Pt/Pd Printing Assistant

            Ask about coating recipes, exposure issues, humidity control, or curve tuning.
            Provide context on the right to receive more targeted advice.
            """
        )

        with gr.Row():
            # Chat area
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="Pt/Pd Printing Assistant", height=500, avatar_images=(None, "🤖")
                )

                with gr.Row():
                    prompt_one = gr.Button("Why is my print too dark?", size="sm")
                    prompt_two = gr.Button("Coating recipe for 8x10", size="sm")
                    prompt_three = gr.Button("Compare Arches vs Bergger", size="sm")

                msg_input = gr.Textbox(
                    placeholder="Ask about Pt/Pd printing...",
                    show_label=False,
                )

                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear")

            # Context sidebar
            with gr.Column(scale=1):
                gr.Markdown("### 📎 Context")
                gr.Markdown("Share context to get better answers:")

                context_curve = gr.Dropdown(
                    label="Active Curve",
                    choices=["None", "Arches Platine v2", "Bergger COT320"],
                    value="None",
                    allow_custom_value=True,
                )
                context_paper = gr.Dropdown(
                    label="Current Paper",
                    choices=["Arches Platine", "Bergger COT320", "Hahnemühle Platinum Rag"],
                    value="Arches Platine",
                    allow_custom_value=True,
                )
                context_image = gr.Image(label="Reference Image (optional)", type="filepath")

                gr.Markdown("### 📚 Resources")
                gr.Markdown("""
                - [Bostick-Sullivan Guide](https://www.bostick-sullivan.com/)
                - [Zone System Primer](https://en.wikipedia.org/wiki/Zone_System)
                - [Troubleshooting Guide](https://github.com/ianshank/Platinum-Palladium-AI-Printing-Tool)
                """)

        # Logic
        prompt_one.click(lambda: "Why is my print too dark?", outputs=msg_input)
        prompt_two.click(lambda: "Coating recipe for 8x10", outputs=msg_input)
        prompt_three.click(lambda: "Compare Arches vs Bergger", outputs=msg_input)

        async def chat(
            message: str, history: list, curve: str, paper: str, image_path: str | None
        ) -> tuple[str, list]:
            try:
                assistant = create_assistant()
                context_lines = [
                    f"Curve: {curve}",
                    f"Paper: {paper}",
                ]
                if image_path:
                    context_lines.append(f"Reference image: {Path(image_path).name}")
                context_block = "\n".join(context_lines)
                prompt = f"[Context]\n{context_block}\n\nQuestion: {message}"

                # Gradio Chatbot expects list of [user, bot] tuples
                # We need to adapt if assistant.chat returns just string
                response = await assistant.chat(prompt, include_history=False)

                history = history or []
                history.append((message, response))
                return "", history
            except Exception as e:
                history = history or []
                history.append((message, f"Error: {str(e)}"))
                return "", history

        send_btn.click(
            chat,
            inputs=[msg_input, chatbot, context_curve, context_paper, context_image],
            outputs=[msg_input, chatbot],
        )

        msg_input.submit(
            chat,
            inputs=[msg_input, chatbot, context_curve, context_paper, context_image],
            outputs=[msg_input, chatbot],
        )

        clear_btn.click(lambda: None, None, chatbot, queue=False)
