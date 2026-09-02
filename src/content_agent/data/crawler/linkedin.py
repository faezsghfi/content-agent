import time
from typing import Dict, List

from bs4 import BeautifulSoup
from bs4.element import Tag
from loguru import logger
from selenium.webdriver.common.by import By

from content_agent.data.documents.documents import PostDocument
from content_agent.exceptions import ImproperlyConfigured
from content_agent.settings import settings

from .base import BaseSeleniumCrawler


class LinkedInCrawler(BaseSeleniumCrawler):
    model = PostDocument # Use PostDocument as the model for storing LinkedIn posts.

    def __init__(self, scroll_limit: int = 5, is_deprecated: bool = True) -> None:
        super().__init__(scroll_limit)

        self._is_deprecated = is_deprecated

    def set_extra_driver_options(self, options) -> None:
        options.add_experimental_option("detach", True) # Keep the browser open after the Selenium process finishes.

    def login(self) -> None:
        # Stop execution because LinkedIn login is no longer supported.
        if self._is_deprecated:
            raise DeprecationWarning(
                "As LinkedIn has updated its security measures, the login() method is no longer supported."
            )

        # Open the LinkedIn login page.
        self.driver.get("https://www.linkedin.com/login")

        # Make sure the LinkedIn credentials are configured.
        if not settings.LINKEDIN_USERNAME or not settings.LINKEDIN_PASSWORD:
            raise ImproperlyConfigured(
                "LinkedIn scraper requires the {LINKEDIN_USERNAME} and {LINKEDIN_PASSWORD} settings."
            )

        self.driver.find_element(By.ID, "username").send_keys(settings.LINKEDIN_USERNAME) # Enter the LinkedIn username.
        self.driver.find_element(By.ID, "password").send_keys(settings.LINKEDIN_PASSWORD) # Enter the LinkedIn password.
        self.driver.find_element(By.CSS_SELECTOR, ".login__form_action_container button").click() # Click the login button.

    def extract(self, link: str, **kwargs) -> None:
        # Stop execution because the LinkedIn crawler is deprecated.
        if self._is_deprecated:
            raise DeprecationWarning(
                "As LinkedIn has updated its feed structure, the extract() method is no longer supported."
            )
         # Skip scraping if the post already exists.
        if self.model.link is not None:
            old_model = self.model.find(link=link)
            if old_model is not None:
                logger.info(f"Post already exists in the database: {link}")

                return

        logger.info(f"Starting scrapping data for profile: {link}")

        # Log in to LinkedIn before accessing the profile.
        self.login()

        # Load and parse the profile page.
        soup = self._get_page_content(link)

        # Extract the main profile information.
        data = {  # noqa
            "Name": self._scrape_section(soup, "h1", class_="text-heading-xlarge"),
            "About": self._scrape_section(soup, "div", class_="display-flex ph5 pv3"),
            "Main Page": self._scrape_section(soup, "div", {"id": "main-content"}),
            "Experience": self._scrape_experience(link),
            "Education": self._scrape_education(link),
        }

        self.driver.get(link) # Open the profile page again.
        time.sleep(5) # Wait for the page to load.

        # Find the button used to access the user's posts/content.
        button = self.driver.find_element(
            By.CSS_SELECTOR, ".app-aware-link.profile-creator-shared-content-view__footer-action"
        )
        button.click() # Click the button to open the user's posts.

        # Scrolling and scraping posts
        self.scroll_page()
        soup = BeautifulSoup(self.driver.page_source, "html.parser") # Parse the fully loaded HTML with BeautifulSoup.
        # Find all elements containing post text.
        post_elements = soup.find_all(
            "div",
            class_="update-components-text relative update-components-update-v2__commentary",
        )
        buttons = soup.find_all("button", class_="update-components-image__image-link") # Find buttons containing post images.
        post_images = self._extract_image_urls(buttons) # Extract image URLs from the image buttons.

        posts = self._extract_posts(post_elements, post_images)  # Combine post text with their corresponding images.
        logger.info(f"Found {len(posts)} posts for profile: {link}")

        self.driver.close() # Close the Selenium browser and release its resources.

        user = kwargs["user"] # Get the user associated with the scraped data.

        # Convert the scraped posts into PostDocument objects and save them in bulk.
        self.model.bulk_insert(
            [
                PostDocument(platform="linkedin", content=post, author_id=user.id, author_full_name=user.full_name)
                for post in posts
            ]
        )

        logger.info(f"Finished scrapping data for profile: {link}")

    def _scrape_section(self, soup: BeautifulSoup, *args, **kwargs) -> str:
        """Scrape a specific section of the LinkedIn profile."""
        # Example: Scrape the 'About' section
        # Find the requested HTML element.
        parent_div = soup.find(*args, **kwargs)


        # Return the text inside the element, or an empty string if not found.
        return parent_div.get_text(strip=True) if parent_div else ""

    def _extract_image_urls(self, buttons: List[Tag]) -> Dict[str, str]:
        """
        Extracts image URLs from button elements.

        Args:
            buttons (List[Tag]): A list of BeautifulSoup Tag objects representing buttons.

        Returns:
            Dict[str, str]: A dictionary mapping post indexes to image URLs.
        """

        post_images = {} # Store image URLs using the post index as the key.

        # Process each image button.
        for i, button in enumerate(buttons):
            # Find the image element inside the button.
            img_tag = button.find("img")

            # Store the image URL if it exists.
            if img_tag and "src" in img_tag.attrs:
                post_images[f"Post_{i}"] = img_tag["src"]
            else:
                logger.warning("No image found in this button")
        return post_images

    def _get_page_content(self, url: str) -> BeautifulSoup:
        """Retrieve the page content of a given URL."""

        # Open the requested URL in the Selenium browser.
        self.driver.get(url)
        time.sleep(5) # Wait for the page to load.


        # Parse the loaded HTML with BeautifulSoup.
        return BeautifulSoup(self.driver.page_source, "html.parser")

    def _extract_posts(self, post_elements: List[Tag], post_images: Dict[str, str]) -> Dict[str, Dict[str, str]]:
        """
        Extracts post texts and combines them with their respective images.

        Args:
            post_elements (List[Tag]): A list of BeautifulSoup Tag objects representing post elements.
            post_images (Dict[str, str]): A dictionary containing image URLs mapped by post index.

        Returns:
            Dict[str, Dict[str, str]]: A dictionary containing post data with text and optional image URL.
        """

        # Store the extracted post data.
        posts_data = {}
        # Process each post element.
        for i, post_element in enumerate(post_elements):
            # Extract the text content of the post.
            post_text = post_element.get_text(strip=True, separator="\n")
            post_data = {"text": post_text} # Create the initial post data with the text.

            # Add the image URL if an image exists for this post.
            if f"Post_{i}" in post_images:
                post_data["image"] = post_images[f"Post_{i}"]

            # Store the post using its index as the key.
            posts_data[f"Post_{i}"] = post_data

        return posts_data

    def _scrape_experience(self, profile_url: str) -> str:
            """Scrape the Experience section of the LinkedIn profile."""

            # Open the profile's Experience page.
            self.driver.get(profile_url + "/details/experience/")

            # Wait for the page to load.
            time.sleep(5)

            # Parse the loaded HTML.
            soup = BeautifulSoup(self.driver.page_source, "html.parser")

            # Find the Experience section.
            experience_content = soup.find(
                "section",
                {"id": "experience-section"}
            )

            # Return the extracted text or an empty string if not found.
            return (
                experience_content.get_text(strip=True)
                if experience_content
                else ""
            )

    def _scrape_education(self, profile_url: str) -> str:
        """Scrape the Education section of the LinkedIn profile."""

        # Open the profile's Education page.
        self.driver.get(profile_url + "/details/education/")

        # Wait for the page to load.
        time.sleep(5)

        # Parse the loaded HTML.
        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        # Find the Education section.
        education_content = soup.find(
            "section",
            {"id": "education-section"}
        )

        # Return the extracted text or an empty string if not found.
        return (
            education_content.get_text(strip=True)
            if education_content
            else ""
        )
