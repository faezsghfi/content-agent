import time
from abc import ABC, abstractmethod
from tempfile import mkdtemp

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from content_agent.data.documents import NoSQLBaseDocument
from selenium.webdriver.chrome.service import Service


class BaseCrawler(ABC):
    # The document model that will be used to store the extracted data.
    model: type[NoSQLBaseDocument]


    # Extract data from the given link.
    # This method must be implemented by every concrete crawler.
    @abstractmethod
    def extract(self, link: str, **kwargs) -> None: ...


class BaseSeleniumCrawler(BaseCrawler, ABC):
    def __init__(self, scroll_limit: int = 5) -> None:

        # Create Chrome browser options.
        options = webdriver.ChromeOptions()

        options.add_argument("--no-sandbox") # Allow Chrome to run without the sandbox.
        options.add_argument("--headless=new") # Run Chrome in headless mode, so no browser window is displayed.
        options.add_argument("--disable-dev-shm-usage") # Prevent issues caused by limited shared memory in Docker or Linux environments.
        options.add_argument("--log-level=3") # Reduce the amount of Chrome logging output.
        options.add_argument("--disable-popup-blocking") # Disable browser pop-up blocking.
        options.add_argument("--disable-notifications") # Disable browser notifications.
        options.add_argument("--disable-extensions") # Disable Chrome extensions.
        options.add_argument("--disable-background-networking") # Disable unnecessary background network activity.
        options.add_argument("--ignore-certificate-errors") # Ignore SSL certificate errors.
        options.add_argument(f"--user-data-dir={mkdtemp()}") # Create a temporary directory for Chrome's user profile.
        options.add_argument(f"--data-path={mkdtemp()}") # Create a temporary directory for Chrome's data files.
        options.add_argument(f"--disk-cache-dir={mkdtemp()}") # Create a temporary directory for Chrome's disk cache.
        options.add_argument("--remote-debugging-port=9226") # Enable remote debugging on port 9226.

        self.set_extra_driver_options(options) # Allow subclasses to add their own Chrome options if needed.

        self.scroll_limit = scroll_limit # Store the maximum number of times the page can be scrolled.

        # Create the Selenium Chrome WebDriver using the configured options.
        service = Service(r"C:\Users\QHS\AppData\Local\Temp\chromedriver-151\chromedriver-win64\chromedriver.exe")

        self.driver = webdriver.Chrome(
            service=service,
            options=options,
        )

    def set_extra_driver_options(self, options: Options) -> None:
        # Hook for subclasses to add additional Chrome options.
        # This method can be overridden when a crawler needs custom settings.
        pass

    def login(self) -> None:
        # Hook for subclasses that require authentication.
        # This method can be overridden to implement the login process.
        pass

    def scroll_page(self) -> None:
        """Scroll through the LinkedIn page based on the scroll limit."""
        current_scroll = 0 # Keep track of how many times the page has been scrolled.


        last_height = self.driver.execute_script("return document.body.scrollHeight") # Get the initial height of the page.
        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);") # Scroll to the bottom of the page.
            time.sleep(5) # Wait for new content to load after scrolling.
            new_height = self.driver.execute_script("return document.body.scrollHeight") # Get the new height of the page.

            # Stop scrolling if:
            # 1. The page height has not changed, meaning no new content was loaded.
            # 2. The configured scroll limit has been reached.
            if new_height == last_height or (self.scroll_limit and current_scroll >= self.scroll_limit):
                break

             # Update the previous page height for the next iteration.
            last_height = new_height
            current_scroll += 1 # Increase the scroll counter.
