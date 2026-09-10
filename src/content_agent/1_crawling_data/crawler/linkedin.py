import time
from typing import Dict, List

from bs4 import BeautifulSoup
from bs4.element import Tag
from loguru import logger

from content_agent.data.documents.documents import PostDocument

from .base import BaseSeleniumCrawler


class LinkedInCrawler(BaseSeleniumCrawler):
    model = PostDocument

    def __init__(
        self,
        scroll_limit: int = 5,
        is_deprecated: bool = False,
    ) -> None:
        super().__init__(scroll_limit)
        self._is_deprecated = is_deprecated

    def set_extra_driver_options(self, options) -> None:
        pass

    def login(self) -> None:
        if self._is_deprecated:
            raise DeprecationWarning(
                "As LinkedIn has updated its security measures, "
                "the login() method is no longer supported."
            )

        self.driver.get("https://www.linkedin.com")
        time.sleep(3)

    def extract(self, link: str, **kwargs) -> None:
        if self._is_deprecated:
            raise DeprecationWarning(
                "As LinkedIn has updated its feed structure, "
                "the extract() method is no longer supported."
            )

        old_model = self.model.find(link=link)

        if old_model is not None:
            logger.info(
                f"Post already exists in the database: {link}"
            )
            return

        logger.info(
            f"Starting scrapping data for profile: {link}"
        )

        self.login()

        soup = self._get_page_content(link)

        data = {
            "Name": self._scrape_section(
                soup,
                "h1",
                class_="text-heading-xlarge",
            ),
            "About": self._scrape_section(
                soup,
                "div",
                class_="display-flex ph5 pv3",
            ),
            "Main Page": self._scrape_section(
                soup,
                "div",
                {"id": "main-content"},
            ),
            "Experience": self._scrape_experience(link),
            "Education": self._scrape_education(link),
        }

        logger.info(
            f"Profile information scraped for: {link}"
        )

        posts_url = (
            link.rstrip("/")
            + "/recent-activity/all/"
        )

        logger.info(
            f"Opening LinkedIn activity page: {posts_url}"
        )

        self.driver.get(posts_url)
        time.sleep(5)

        logger.info(
            f"Activity page title: {self.driver.title}"
        )

        self.scroll_page()

        soup = BeautifulSoup(
            self.driver.page_source,
            "html.parser",
        )

        logger.info(
            f"Activity page source length: "
            f"{len(self.driver.page_source)}"
        )

        post_elements = soup.find_all(
            "div",
            class_=(
                "update-components-text "
                "relative "
                "update-components-update-v2__commentary"
            ),
        )

        logger.info(
            f"Found {len(post_elements)} post text elements"
        )

        image_buttons = soup.find_all(
            "button",
            class_="update-components-image__image-link",
        )

        post_images = self._extract_image_urls(
            image_buttons
        )

        logger.info(
            f"Found {len(post_images)} post images"
        )

        posts = self._extract_posts(
            post_elements,
            post_images,
        )

        logger.info(
            f"Found {len(posts)} posts for profile: {link}"
        )

        if not posts:
            logger.warning(
                f"No posts found for profile: {link}"
            )
            self.driver.close()
            return

        self.driver.close()

        user = kwargs["user"]

        self.model.bulk_insert(
            [
                PostDocument(
                    platform="linkedin",
                    content=post,
                    author_id=user.id,
                    author_full_name=user.full_name,
                )
                for post in posts.values()
            ]
        )

        logger.info(
            f"Finished scrapping data for profile: {link}"
        )

    def _scrape_section(
        self,
        soup: BeautifulSoup,
        *args,
        **kwargs,
    ) -> str:
        parent_div = soup.find(*args, **kwargs)

        return (
            parent_div.get_text(strip=True)
            if parent_div
            else ""
        )

    def _extract_image_urls(
        self,
        buttons: List[Tag],
    ) -> Dict[str, str]:
        post_images: Dict[str, str] = {}

        for i, button in enumerate(buttons):
            img_tag = button.find("img")

            if img_tag and "src" in img_tag.attrs:
                post_images[f"Post_{i}"] = img_tag["src"]
            else:
                logger.warning(
                    "No image found in this button"
                )

        return post_images

    def _get_page_content(
        self,
        url: str,
    ) -> BeautifulSoup:
        self.driver.get(url)
        time.sleep(5)

        return BeautifulSoup(
            self.driver.page_source,
            "html.parser",
        )

    def _extract_posts(
        self,
        post_elements: List[Tag],
        post_images: Dict[str, str],
    ) -> Dict[str, Dict[str, str]]:
        posts_data: Dict[str, Dict[str, str]] = {}

        for i, post_element in enumerate(post_elements):
            post_text = post_element.get_text(
                strip=True,
                separator="\n",
            )

            post_data: Dict[str, str] = {
                "text": post_text,
            }

            if f"Post_{i}" in post_images:
                post_data["image"] = post_images[f"Post_{i}"]

            posts_data[f"Post_{i}"] = post_data

        return posts_data

    def _scrape_experience(
        self,
        profile_url: str,
    ) -> str:
        self.driver.get(
            profile_url.rstrip("/")
            + "/details/experience/"
        )

        time.sleep(5)

        soup = BeautifulSoup(
            self.driver.page_source,
            "html.parser",
        )

        experience_content = soup.find(
            "section",
            {"id": "experience-section"},
        )

        return (
            experience_content.get_text(strip=True)
            if experience_content
            else ""
        )

    def _scrape_education(
        self,
        profile_url: str,
    ) -> str:
        self.driver.get(
            profile_url.rstrip("/")
            + "/details/education/"
        )

        time.sleep(5)

        soup = BeautifulSoup(
            self.driver.page_source,
            "html.parser",
        )

        education_content = soup.find(
            "section",
            {"id": "education-section"},
        )

        return (
            education_content.get_text(strip=True)
            if education_content
            else ""
        )