"""
UI Component Builders

Provides builder patterns for creating Gradio UIs with:
- Modular tab construction
- Consistent styling
- Error handling integration
- State management
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from ptpd_calibration.template.errors import ErrorBoundary, create_gradio_error_wrapper
from ptpd_calibration.template.logging_config import get_logger

logger = get_logger(__name__)


class UITheme(BaseModel):
    """UI theme configuration."""

    primary_color: str = "#2563eb"
    secondary_color: str = "#64748b"
    background_color: str = "#ffffff"
    text_color: str = "#1f2937"
    error_color: str = "#ef4444"
    success_color: str = "#22c55e"
    warning_color: str = "#f59e0b"
    font_family: str = "system-ui, sans-serif"
    border_radius: str = "0.5rem"
    spacing: str = "1rem"


class TabConfig(BaseModel):
    """Configuration for a UI tab."""

    id: str
    label: str
    icon: str | None = None
    description: str = ""
    enabled: bool = True
    visible: bool = True
    order: int = 0
    requires_features: list[str] = Field(default_factory=list)


class UIComponentBuilder:
    """
    Builder for creating individual UI components.

    Provides consistent styling and error handling for Gradio components.

    Usage:
        builder = UIComponentBuilder(theme=theme)

        # Build components
        input_box = builder.text_input("Query", placeholder="Enter query...")
        submit_btn = builder.button("Submit", variant="primary")
    """

    def __init__(self, theme: UITheme | None = None):
        """Initialize component builder."""
        self.theme = theme or UITheme()
        self._components: list[Any] = []

    def _check_gradio(self) -> Any:
        """Check if Gradio is available and return module."""
        try:
            import gradio as gr
            return gr
        except ImportError:
            raise ImportError("Gradio is required for UI components")

    def text_input(
        self,
        label: str,
        placeholder: str = "",
        value: str = "",
        lines: int = 1,
        max_lines: int = 5,
        **kwargs: Any,
    ) -> Any:
        """Create a text input component."""
        gr = self._check_gradio()

        if lines > 1:
            component = gr.Textbox(
                label=label,
                placeholder=placeholder,
                value=value,
                lines=lines,
                max_lines=max_lines,
                **kwargs,
            )
        else:
            component = gr.Textbox(
                label=label,
                placeholder=placeholder,
                value=value,
                **kwargs,
            )

        self._components.append(component)
        return component

    def button(
        self,
        label: str,
        variant: str = "secondary",
        size: str = "md",
        icon: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a button component."""
        gr = self._check_gradio()

        component = gr.Button(
            label,
            variant=variant,
            size=size,
            icon=icon,
            **kwargs,
        )

        self._components.append(component)
        return component

    def dropdown(
        self,
        label: str,
        choices: list[str],
        value: str | None = None,
        multiselect: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create a dropdown component."""
        gr = self._check_gradio()

        component = gr.Dropdown(
            label=label,
            choices=choices,
            value=value or (choices[0] if choices else None),
            multiselect=multiselect,
            **kwargs,
        )

        self._components.append(component)
        return component

    def slider(
        self,
        label: str,
        minimum: float = 0,
        maximum: float = 100,
        value: float = 50,
        step: float = 1,
        **kwargs: Any,
    ) -> Any:
        """Create a slider component."""
        gr = self._check_gradio()

        component = gr.Slider(
            label=label,
            minimum=minimum,
            maximum=maximum,
            value=value,
            step=step,
            **kwargs,
        )

        self._components.append(component)
        return component

    def checkbox(
        self,
        label: str,
        value: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create a checkbox component."""
        gr = self._check_gradio()

        component = gr.Checkbox(
            label=label,
            value=value,
            **kwargs,
        )

        self._components.append(component)
        return component

    def image_input(
        self,
        label: str = "Image",
        type: str = "pil",
        **kwargs: Any,
    ) -> Any:
        """Create an image input component."""
        gr = self._check_gradio()

        component = gr.Image(
            label=label,
            type=type,
            **kwargs,
        )

        self._components.append(component)
        return component

    def file_input(
        self,
        label: str = "File",
        file_types: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Create a file input component."""
        gr = self._check_gradio()

        component = gr.File(
            label=label,
            file_types=file_types,
            **kwargs,
        )

        self._components.append(component)
        return component

    def output_text(
        self,
        label: str = "Output",
        **kwargs: Any,
    ) -> Any:
        """Create a text output component."""
        gr = self._check_gradio()

        component = gr.Textbox(
            label=label,
            interactive=False,
            **kwargs,
        )

        self._components.append(component)
        return component

    def output_json(
        self,
        label: str = "Output",
        **kwargs: Any,
    ) -> Any:
        """Create a JSON output component."""
        gr = self._check_gradio()

        component = gr.JSON(
            label=label,
            **kwargs,
        )

        self._components.append(component)
        return component

    def output_image(
        self,
        label: str = "Output",
        **kwargs: Any,
    ) -> Any:
        """Create an image output component."""
        gr = self._check_gradio()

        component = gr.Image(
            label=label,
            interactive=False,
            **kwargs,
        )

        self._components.append(component)
        return component

    def markdown(
        self,
        content: str,
        **kwargs: Any,
    ) -> Any:
        """Create a markdown component."""
        gr = self._check_gradio()

        component = gr.Markdown(content, **kwargs)
        self._components.append(component)
        return component

    def row(self) -> Any:
        """Create a row layout."""
        gr = self._check_gradio()
        return gr.Row()

    def column(self, scale: int = 1) -> Any:
        """Create a column layout."""
        gr = self._check_gradio()
        return gr.Column(scale=scale)

    def accordion(self, label: str, open: bool = False) -> Any:
        """Create an accordion component."""
        gr = self._check_gradio()
        return gr.Accordion(label=label, open=open)


class TabBuilder(ABC):
    """
    Abstract base class for building UI tabs.

    Subclass to create specific tabs with consistent structure.

    Usage:
        class SettingsTab(TabBuilder):
            def build_components(self) -> None:
                self.api_key = self.builder.text_input("API Key")

            def setup_events(self) -> None:
                self.api_key.change(self.on_api_key_change, ...)

        tab = SettingsTab(config)
        tab.build()
    """

    def __init__(
        self,
        config: TabConfig,
        theme: UITheme | None = None,
    ):
        """Initialize tab builder."""
        self.config = config
        self.theme = theme or UITheme()
        self.builder = UIComponentBuilder(theme)
        self._tab = None
        self._built = False
        self._error_boundary = ErrorBoundary(
            component=f"tab.{config.id}",
            reraise=False,
        )

    @abstractmethod
    def build_components(self) -> None:
        """Build the tab's components. Must be implemented by subclasses."""
        pass

    def setup_events(self) -> None:
        """Set up event handlers. Override in subclasses."""
        pass

    def build(self) -> Any:
        """
        Build the complete tab.

        Returns:
            Gradio Tab component
        """
        if self._built:
            return self._tab

        try:
            import gradio as gr
        except ImportError:
            raise ImportError("Gradio is required for tab building")

        with gr.Tab(
            label=self.config.label,
            id=self.config.id,
            visible=self.config.visible,
        ) as tab:
            self._tab = tab

            # Add description if provided
            if self.config.description:
                gr.Markdown(f"*{self.config.description}*")

            # Build components
            with self._error_boundary.protect(operation="build_components"):
                self.build_components()

            # Setup events
            with self._error_boundary.protect(operation="setup_events"):
                self.setup_events()

        self._built = True
        logger.debug(f"Built tab: {self.config.id}")

        return tab

    def wrap_handler(
        self,
        handler: Callable,
        name: str | None = None,
    ) -> Callable:
        """Wrap an event handler with error handling."""
        return create_gradio_error_wrapper(
            component=f"tab.{self.config.id}.{name or handler.__name__}"
        )(handler)


@dataclass
class AppConfig:
    """Configuration for a Gradio application."""

    title: str = "Application"
    description: str = ""
    theme: UITheme | None = None
    analytics_enabled: bool = False
    show_api: bool = False
    share: bool = False
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    favicon_path: str | None = None
    css: str | None = None


class GradioAppBuilder:
    """
    Builder for creating complete Gradio applications.

    Provides:
    - Tab management
    - Theme application
    - State management
    - Error handling

    Usage:
        builder = GradioAppBuilder(AppConfig(title="My App"))

        # Add tabs
        builder.add_tab(SettingsTab(config))
        builder.add_tab(ProcessingTab(config))

        # Build and launch
        app = builder.build()
        app.launch()
    """

    def __init__(self, config: AppConfig):
        """Initialize app builder."""
        self.config = config
        self.theme = config.theme or UITheme()
        self._tabs: list[TabBuilder] = []
        self._header_content: str | None = None
        self._footer_content: str | None = None
        self._custom_css: list[str] = []
        self._app = None

    def add_tab(self, tab: TabBuilder) -> GradioAppBuilder:
        """Add a tab to the application."""
        self._tabs.append(tab)
        return self

    def set_header(self, content: str) -> GradioAppBuilder:
        """Set header markdown content."""
        self._header_content = content
        return self

    def set_footer(self, content: str) -> GradioAppBuilder:
        """Set footer markdown content."""
        self._footer_content = content
        return self

    def add_css(self, css: str) -> GradioAppBuilder:
        """Add custom CSS."""
        self._custom_css.append(css)
        return self

    def _generate_css(self) -> str:
        """Generate combined CSS."""
        theme_css = f"""
        :root {{
            --primary-color: {self.theme.primary_color};
            --secondary-color: {self.theme.secondary_color};
            --background-color: {self.theme.background_color};
            --text-color: {self.theme.text_color};
            --error-color: {self.theme.error_color};
            --success-color: {self.theme.success_color};
            --warning-color: {self.theme.warning_color};
            --font-family: {self.theme.font_family};
            --border-radius: {self.theme.border_radius};
            --spacing: {self.theme.spacing};
        }}
        """

        custom_css = "\n".join(self._custom_css)

        return f"{theme_css}\n{self.config.css or ''}\n{custom_css}"

    def build(self) -> Any:
        """
        Build the complete Gradio application.

        Returns:
            Gradio Blocks application
        """
        try:
            import gradio as gr
        except ImportError:
            raise ImportError("Gradio is required for app building")

        # Sort tabs by order
        sorted_tabs = sorted(self._tabs, key=lambda t: t.config.order)

        with gr.Blocks(
            title=self.config.title,
            analytics_enabled=self.config.analytics_enabled,
            css=self._generate_css(),
        ) as app:
            self._app = app

            # Header
            if self._header_content:
                gr.Markdown(self._header_content)
            else:
                gr.Markdown(f"# {self.config.title}")
                if self.config.description:
                    gr.Markdown(self.config.description)

            # Tabs
            with gr.Tabs():
                for tab in sorted_tabs:
                    if tab.config.enabled and tab.config.visible:
                        tab.build()

            # Footer
            if self._footer_content:
                gr.Markdown(self._footer_content)

        logger.info(
            f"Built Gradio app: {self.config.title}",
            tabs=len(sorted_tabs),
        )

        return app

    def launch(self, **kwargs: Any) -> None:
        """Launch the application."""
        if self._app is None:
            self.build()

        launch_kwargs = {
            "server_name": self.config.server_name,
            "server_port": self.config.server_port,
            "share": self.config.share,
            "show_api": self.config.show_api,
            "favicon_path": self.config.favicon_path,
            **kwargs,
        }

        logger.info(
            f"Launching app on {self.config.server_name}:{self.config.server_port}"
        )

        self._app.launch(**launch_kwargs)


class StateManager:
    """
    Manages UI state across components.

    Provides:
    - Centralized state storage
    - State change callbacks
    - Persistence support
    """

    def __init__(self):
        """Initialize state manager."""
        self._state: dict[str, Any] = {}
        self._callbacks: dict[str, list[Callable]] = {}

    def get(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self._state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a state value and trigger callbacks."""
        old_value = self._state.get(key)
        self._state[key] = value

        # Trigger callbacks
        if key in self._callbacks:
            for callback in self._callbacks[key]:
                try:
                    callback(key, old_value, value)
                except Exception as e:
                    logger.error(f"State callback error for {key}: {e}")

    def subscribe(self, key: str, callback: Callable[[str, Any, Any], None]) -> None:
        """Subscribe to state changes."""
        if key not in self._callbacks:
            self._callbacks[key] = []
        self._callbacks[key].append(callback)

    def unsubscribe(self, key: str, callback: Callable) -> None:
        """Unsubscribe from state changes."""
        if key in self._callbacks and callback in self._callbacks[key]:
            self._callbacks[key].remove(callback)

    def to_dict(self) -> dict[str, Any]:
        """Export state as dictionary."""
        return dict(self._state)

    def from_dict(self, data: dict[str, Any]) -> None:
        """Import state from dictionary."""
        for key, value in data.items():
            self.set(key, value)
