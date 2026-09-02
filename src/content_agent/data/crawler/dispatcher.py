import re
from urllib.parse import urlparse

from loguru import logger

from .base import BaseCrawler
from .custom_article import CustomArticleCrawler
from .github import GithubCrawler
from .linkedin import LinkedInCrawler
from .medium import MediumCrawler


class CrawlerDispatcher:
    def __init__(self) -> None:
        self._crawlers = {} # Store the mapping between URL patterns and their crawlers.

    @classmethod
    def build(cls) -> "CrawlerDispatcher":
        # Create and return a new CrawlerDispatcher instance.
        dispatcher = cls()
        return dispatcher

    def register_medium(self) -> "CrawlerDispatcher":
        # Register MediumCrawler for Medium URLs.
        self.register("https://medium.com", MediumCrawler)
        # Return self so multiple register methods can be chained.
        return self

    def register_linkedin(self) -> "CrawlerDispatcher":
        # Register LinkedInCrawler for LinkedIn URLs.
        self.register("https://linkedin.com", LinkedInCrawler)
        # Return self to allow method chaining.
        return self

    def register_github(self) -> "CrawlerDispatcher":
        # Register GithubCrawler for GitHub URLs.
        self.register("https://github.com", GithubCrawler)
        # Return self to allow method chaining.
        return self

    def register(self, domain: str, crawler: type[BaseCrawler]) -> None:
        # Parse the URL and extract only its domain.
        parsed_domain = urlparse(domain)
        domain = parsed_domain.netloc

        # Create a URL pattern and map it to the corresponding crawler.
        # This pattern will be used later to match incoming URLs.
        self._crawlers[r"https://(www\.)?{}/*".format(re.escape(domain))] = crawler

    def get_crawler(self, url: str) -> BaseCrawler:
        # Check the registered URL patterns one by one.
        for pattern, crawler in self._crawlers.items():
            # If the URL matches a registered pattern,
            # create and return an instance of the corresponding crawler.
            if re.match(pattern, url):
                return crawler()
        else:
            # If no specific crawler matches the URL,
            # use CustomArticleCrawler as the default crawler.
            logger.warning(f"No crawler found for {url}. Defaulting to CustomArticleCrawler.")

            return CustomArticleCrawler()
