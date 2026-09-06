from bs4 import BeautifulSoup
from loguru import logger

from content_agent.data.documents.documents import ArticleDocument

from .base import BaseSeleniumCrawler


class MediumCrawler(BaseSeleniumCrawler):
    model = ArticleDocument # Use ArticleDocument as the model for storing scraped articles.

    def set_extra_driver_options(self, options) -> None:
        # Add additional options to the Selenium WebDriver.
        pass

    def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link)  # Check if the article already exists in the database.

        # Stop the scraping process if the article already exists
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")

            return

        logger.info(f"Starting scrapping Medium article: {link}")

        # Open the article URL in the browser.
        self.driver.get(link)
        self.scroll_page() # Scroll through the page to make sure all content is loaded.

        soup = BeautifulSoup(self.driver.page_source, "html.parser") # Parse the loaded HTML using BeautifulSoup.

        # Find the article title and subtitle.
        title = soup.find_all("h1", class_="pw-post-title")
        subtitle = soup.find_all("h2", class_="pw-subtitle-paragraph")

        # Extract the title, subtitle, and full text of the article.
        data = {
            "Title": title[0].string if title else None,
            "Subtitle": subtitle[0].string if subtitle else None,
            "Content": soup.get_text(),
        }

        self.driver.quit() # Close the WebDriver to release browser resources.

        user = kwargs["user"] # Get the user who owns or submitted the article.

        # Create a new ArticleDocument with the scraped data.
        instance = self.model(
            platform="medium",
            content=data,
            link=link,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save()

        logger.info(f"Successfully scraped and saved article: {link}")
