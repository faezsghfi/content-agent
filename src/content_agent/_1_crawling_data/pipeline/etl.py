from typing_extensions import Annotated

from zenml import get_step_context, pipeline, step

from content_agent.crawling_data.crawler.dispatcher import CrawlerDispatcher
from content_agent.crawling_data.documents.documents import UserDocument
from content_agent.utils import split_user_full_name
from loguru import logger


@step
def get_or_create_user(
    user_full_name: str,
) -> UserDocument:
    """Get an existing user or create a new one."""

    first_name, last_name = split_user_full_name(user_full_name)

    return UserDocument.get_or_create(
        first_name=first_name,
        last_name=last_name,
    )


@step
def crawl_links(
    user: UserDocument,
    links: list[str],
) -> Annotated[list[str], "crawled_links"]:
    """Crawl all provided links and store the extracted data."""

    dispatcher = (
        CrawlerDispatcher.build()
        .register_linkedin()
        .register_medium()
        .register_github()
    )

    successful_crawls = 0
    metadata = {}

    for link in links:
        domain = link.split("/")[2]

        if domain not in metadata:
            metadata[domain] = {
                "successful": 0,
                "total": 0,
            }

        metadata[domain]["total"] += 1

        try:
            crawler = dispatcher.get_crawler(link)

            crawler.extract(
                link=link,
                user=user,
            )

            successful_crawls += 1
            metadata[domain]["successful"] += 1

        except Exception as e:
            logger.error(f"Failed to crawl {link}: {e}")

    step_context = get_step_context()

    step_context.add_output_metadata(
        output_name="crawled_links",
        metadata=metadata,
    )

    return links


@pipeline(enable_cache=False)
def digital_data_etl(
    user_full_name: str,
    links: list[str],
) -> str:
    """Run the digital data ETL pipeline."""

    user = get_or_create_user(
        user_full_name=user_full_name,
    )

    last_step = crawl_links(
        user=user,
        links=links,
    )

    return last_step.invocation_id