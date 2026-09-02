from urllib.parse import urlparse

from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers.html2text import Html2TextTransformer
from loguru import logger

from content_agent.data.documents.documents import ArticleDocument

from .base import BaseCrawler


class CustomArticleCrawler(BaseCrawler):
    model = ArticleDocument # Use ArticleDocument as the database model for scraped articles.

    def __init__(self) -> None:
        super().__init__()

    def extract(self, link: str, **kwargs) -> None:
        old_model = self.model.find(link=link) # Check if the article has already been stored in the database.

        # Skip the scraping process if the article already exists.
        if old_model is not None:
            logger.info(f"Article already exists in the database: {link}")

            return

        logger.info(f"Starting scrapping article: {link}")

        # Load the HTML content of the given URL.
        loader = AsyncHtmlLoader([link])
        docs = loader.load()


        # Convert the HTML content into plain text.
        html2text = Html2TextTransformer()
        docs_transformed = html2text.transform_documents(docs)

        # Get the first transformed document.
        doc_transformed = docs_transformed[0]


        # Extract the article content and relevant metadata.
        content = {
            "Title": doc_transformed.metadata.get("title"),
            "Subtitle": doc_transformed.metadata.get("description"),
            "Content": doc_transformed.page_content,
            "language": doc_transformed.metadata.get("language"),
        }

        # Parse the URL and extract the domain name.
        parsed_url = urlparse(link)
        platform = parsed_url.netloc

        # Get the user who is responsible for this article.
        user = kwargs["user"]

        # Create a new ArticleDocument instance with the scraped data.
        instance = self.model(
            content=content,
            link=link,
            platform=platform,
            author_id=user.id,
            author_full_name=user.full_name,
        )
        instance.save()

        logger.info(f"Finished scrapping custom article: {link}")
