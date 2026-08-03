"""
Base Page Object Model for PTPD Calibration UI.

Provides common methods and utilities for interacting with Gradio-based UI components.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver
    from selenium.webdriver.remote.webelement import WebElement

try:
    from selenium.common.exceptions import NoSuchElementException, TimeoutException
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys  # noqa: F401
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    By = None
    WebDriverWait = None
    EC = None


class BasePage:
    """Base class for all Page Objects."""

    DEFAULT_TIMEOUT = 30

    def __init__(self, driver: "WebDriver"):
        """Initialize with WebDriver instance."""
        self.driver = driver
        self._wait = WebDriverWait(driver, self.DEFAULT_TIMEOUT) if SELENIUM_AVAILABLE else None

    # --- Core Element Methods ---

    def find_element(self, by: str, value: str) -> "WebElement":
        """Find a single element."""
        return self.driver.find_element(by, value)

    def find_elements(self, by: str, value: str) -> list["WebElement"]:
        """Find multiple elements."""
        return self.driver.find_elements(by, value)

    def wait_for_element(self, by: str, value: str, timeout: int | None = None) -> "WebElement":
        """Wait for element to be present."""
        timeout = timeout or self.DEFAULT_TIMEOUT
        wait = WebDriverWait(self.driver, timeout)
        return wait.until(EC.presence_of_element_located((by, value)))

    def wait_for_clickable(self, by: str, value: str, timeout: int | None = None) -> "WebElement":
        """Wait for element to be clickable."""
        timeout = timeout or self.DEFAULT_TIMEOUT
        wait = WebDriverWait(self.driver, timeout)
        return wait.until(EC.element_to_be_clickable((by, value)))

    def wait_for_visible(self, by: str, value: str, timeout: int | None = None) -> "WebElement":
        """Wait for element to be visible."""
        timeout = timeout or self.DEFAULT_TIMEOUT
        wait = WebDriverWait(self.driver, timeout)
        return wait.until(EC.visibility_of_element_located((by, value)))

    def wait_for_invisible(self, by: str, value: str, timeout: int | None = None) -> bool:
        """Wait for element to become invisible."""
        timeout = timeout or self.DEFAULT_TIMEOUT
        wait = WebDriverWait(self.driver, timeout)
        return wait.until(EC.invisibility_of_element_located((by, value)))

    def wait_for_text(
        self, by: str, value: str, text: str, timeout: int | None = None
    ) -> "WebElement":
        """Wait for element to contain specific text."""
        timeout = timeout or self.DEFAULT_TIMEOUT
        wait = WebDriverWait(self.driver, timeout)
        return wait.until(EC.text_to_be_present_in_element((by, value), text))

    def element_exists(self, by: str, value: str) -> bool:
        """Check if element exists on page."""
        try:
            self.driver.find_element(by, value)
            return True
        except NoSuchElementException:
            return False

    # --- Gradio-Specific Methods ---

    def wait_for_gradio_ready(self, timeout: int | None = None) -> None:
        """Wait for Gradio app to be fully loaded."""
        timeout = timeout or self.DEFAULT_TIMEOUT
        wait = WebDriverWait(self.driver, timeout)

        # Wait for main container
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".gradio-container")))

        # Wait for loading overlays to disappear
        import contextlib

        with contextlib.suppress(TimeoutException):
            wait.until(
                EC.invisibility_of_element_located((By.CSS_SELECTOR, ".loading, .svelte-loading"))
            )

    def click_tab(self, tab_name: str) -> None:
        """Click on a Gradio tab by its name."""
        # Find all tabs and match by text content
        tabs = self.find_elements(By.CSS_SELECTOR, "button[role='tab']")
        for tab in tabs:
            if tab_name.lower() in tab.text.lower():
                tab.click()
                return
        raise NoSuchElementException(f"Tab '{tab_name}' not found")

    def get_active_tab(self) -> str:
        """Get the name of the currently active tab."""
        active_tab = self.find_element(By.CSS_SELECTOR, "button[role='tab'][aria-selected='true']")
        return active_tab.text

    def fill_textbox(self, label: str, value: str) -> None:
        """Fill a Gradio textbox by its label."""
        # Find textbox by label
        textbox = self._find_gradio_component(label, "textbox")
        textbox.clear()
        textbox.send_keys(value)

    def fill_number(self, label: str, value: float | int) -> None:
        """Fill a Gradio number input by its label."""
        number_input = self._find_gradio_component(label, "number")
        number_input.clear()
        number_input.send_keys(str(value))

    def select_dropdown(self, label: str, option: str) -> None:
        """Select an option from a Gradio dropdown."""
        dropdown = self._find_gradio_component(label, "dropdown")
        dropdown.click()

        # Wait for dropdown to open and find option
        self.wait_for_element(By.CSS_SELECTOR, ".dropdown-content, ul[role='listbox']")

        options = self.find_elements(By.CSS_SELECTOR, ".dropdown-item, li[role='option']")
        for opt in options:
            if option.lower() in opt.text.lower():
                opt.click()
                return
        raise NoSuchElementException(f"Option '{option}' not found in dropdown")

    def adjust_slider(self, label: str, value: float) -> None:
        """Adjust a Gradio slider to a specific value."""
        slider_container = self._find_gradio_component(label, "slider")
        slider_input = slider_container.find_element(By.CSS_SELECTOR, "input[type='range']")

        # Get slider min/max
        min_val = float(slider_input.get_attribute("min") or 0)
        max_val = float(slider_input.get_attribute("max") or 100)
        width = slider_input.size["width"]

        # Calculate offset
        percentage = (value - min_val) / (max_val - min_val)
        offset = int(width * percentage) - int(width / 2)

        actions = ActionChains(self.driver)
        actions.click_and_hold(slider_input)
        actions.move_by_offset(offset, 0)
        actions.release()
        actions.perform()

    def check_checkbox(self, label: str, checked: bool = True) -> None:
        """Check or uncheck a Gradio checkbox."""
        checkbox = self._find_gradio_component(label, "checkbox")
        is_checked = checkbox.get_attribute("checked") == "true"
        if is_checked != checked:
            checkbox.click()

    def click_button(self, text: str) -> None:
        """Click a button by its text content."""
        buttons = self.find_elements(By.CSS_SELECTOR, "button")
        for button in buttons:
            if text.lower() in button.text.lower():
                self.wait_for_clickable(By.CSS_SELECTOR, "button")
                button.click()
                return
        raise NoSuchElementException(f"Button '{text}' not found")

    def upload_file(self, label: str, file_path: str) -> None:
        """Upload a file to a Gradio file upload component."""
        upload_container = self._find_gradio_component(label, "upload")
        file_input = upload_container.find_element(By.CSS_SELECTOR, "input[type='file']")
        file_input.send_keys(file_path)

    def get_output_text(self, label: str) -> str:
        """Get text from a Gradio output component."""
        output = self._find_gradio_component(label, "output")
        return output.text

    def get_markdown_content(self, container_selector: str | None = None) -> str:
        """Get content from a Gradio markdown component."""
        if container_selector:
            container = self.find_element(By.CSS_SELECTOR, container_selector)
            markdown = container.find_element(By.CSS_SELECTOR, ".markdown-content, .prose")
        else:
            markdown = self.find_element(By.CSS_SELECTOR, ".markdown-content, .prose")
        return markdown.text

    def _find_gradio_component(self, label: str, component_type: str) -> "WebElement":
        """Find a Gradio component by its label."""
        # Try multiple strategies to find the component

        # Strategy 1: Find by label text
        labels = self.find_elements(By.CSS_SELECTOR, "label, span.label-text")
        for lbl in labels:
            if label.lower() in lbl.text.lower():
                # Get parent container and find input
                parent = lbl.find_element(By.XPATH, "./..")
                inputs = parent.find_elements(By.CSS_SELECTOR, "input, textarea, select")
                if inputs:
                    return inputs[0]

        # Strategy 2: Find by aria-label
        try:
            return self.find_element(By.CSS_SELECTOR, f"[aria-label*='{label}']")
        except NoSuchElementException:
            pass

        # Strategy 3: Find by placeholder
        try:
            return self.find_element(By.CSS_SELECTOR, f"[placeholder*='{label}']")
        except NoSuchElementException:
            pass

        raise NoSuchElementException(f"Could not find Gradio {component_type} with label '{label}'")

    # --- Navigation Methods ---

    def navigate_to(self, url: str) -> None:
        """Navigate to a specific URL."""
        self.driver.get(url)

    def refresh(self) -> None:
        """Refresh the current page."""
        self.driver.refresh()

    def go_back(self) -> None:
        """Go back in browser history."""
        self.driver.back()

    def get_current_url(self) -> str:
        """Get the current page URL."""
        return self.driver.current_url

    def get_page_title(self) -> str:
        """Get the current page title."""
        return self.driver.title

    # --- Screenshot Methods ---

    def take_screenshot(self, name: str) -> str:
        """Take a screenshot and return the file path."""
        from pathlib import Path

        screenshots_dir = Path(__file__).parent.parent / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        filepath = screenshots_dir / f"{name}.png"
        self.driver.save_screenshot(str(filepath))
        return str(filepath)

    def get_element_screenshot(self, by: str, value: str, name: str) -> str:
        """Take a screenshot of a specific element."""
        from pathlib import Path

        element = self.find_element(by, value)
        screenshots_dir = Path(__file__).parent.parent / "screenshots"
        screenshots_dir.mkdir(parents=True, exist_ok=True)
        filepath = screenshots_dir / f"{name}.png"
        element.screenshot(str(filepath))
        return str(filepath)

    # --- Alert Handling ---

    def accept_alert(self, timeout: int | None = None) -> str:
        """Accept an alert and return its text."""
        timeout = timeout or self.DEFAULT_TIMEOUT
        wait = WebDriverWait(self.driver, timeout)
        alert = wait.until(EC.alert_is_present())
        text = alert.text
        alert.accept()
        return text

    def dismiss_alert(self, timeout: int | None = None) -> str:
        """Dismiss an alert and return its text."""
        timeout = timeout or self.DEFAULT_TIMEOUT
        wait = WebDriverWait(self.driver, timeout)
        alert = wait.until(EC.alert_is_present())
        text = alert.text
        alert.dismiss()
        return text

    # --- Scroll Methods ---

    def scroll_to_element(self, element: "WebElement") -> None:
        """Scroll to bring element into view."""
        self.driver.execute_script(
            "arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});",
            element,
        )

    def scroll_to_top(self) -> None:
        """Scroll to the top of the page."""
        self.driver.execute_script("window.scrollTo(0, 0);")

    def scroll_to_bottom(self) -> None:
        """Scroll to the bottom of the page."""
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # --- JavaScript Execution ---

    def execute_script(self, script: str, *args) -> Any:
        """Execute JavaScript in the browser."""
        return self.driver.execute_script(script, *args)

    def get_local_storage(self, key: str) -> str | None:
        """Get a value from local storage."""
        return self.driver.execute_script(f"return localStorage.getItem('{key}');")

    def set_local_storage(self, key: str, value: str) -> None:
        """Set a value in local storage."""
        self.driver.execute_script(f"localStorage.setItem('{key}', '{value}');")
