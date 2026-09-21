from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from qdrant_client.http import exceptions
from typing_extensions import Annotated
from zenml import step

from content_agent._1_crawling_data.documents.NoSQLBaseDocument import NoSQLBaseDocument
from content_agent._3_instruction_preference_dataset.generation.cleaned_documents import (
    CleanedArticleDocument,
    CleanedDocument,
    CleanedPostDocument,
    CleanedRepositoryDocument,
)

# Queries the feature store and retrieves all cleaned documents needed
# for generating the instruction dataset.
@step
def query_feature_store() -> Annotated[list, "queried_cleaned_documents"]:
    logger.info("Querying feature store.")

    # Fetch articles, posts, and repositories in parallel.
    results = fetch_all_data()

    # Merge the results from all document types into a single list.
    cleaned_documents = [doc for query_result in results.values() for doc in query_result]

    return cleaned_documents


def fetch_all_data() -> dict[str, list[NoSQLBaseDocument]]:
    # Use multiple threads to query different document types concurrently.
    with ThreadPoolExecutor() as executor:
        future_to_query = {
            executor.submit(
                __fetch_articles,
            ): "articles",
            executor.submit(
                __fetch_posts,
            ): "posts",
            executor.submit(
                __fetch_repositories,
            ): "repositories",
        }

        results = {}
        # Process each query as soon as it finishes.
        for future in as_completed(future_to_query):
            query_name = future_to_query[future]
            try:
                results[query_name] = future.result() # Store the documents returned by the completed query.
            except Exception:
                logger.exception(f"'{query_name}' request failed.") # Log the error and continue with the other document types.

                results[query_name] = []

    return results


def __fetch_articles() -> list[CleanedDocument]:
    return __fetch(CleanedArticleDocument)  # Fetch all cleaned article documents.


def __fetch_posts() -> list[CleanedDocument]:
    return __fetch(CleanedPostDocument) # Fetch all cleaned post documents.


def __fetch_repositories() -> list[CleanedDocument]:
    return __fetch(CleanedRepositoryDocument) # Fetch all cleaned repository documents.


def __fetch(cleaned_document_type: type[CleanedDocument], limit: int = 1) -> list[CleanedDocument]:
    try:
        # Fetch the first batch of documents and get the offset
        # required for retrieving the next batch.
        cleaned_documents, next_offset = cleaned_document_type.bulk_find(limit=limit)
    except exceptions.UnexpectedResponse:
        return [] # Return an empty list if the data store returns an unexpected response.

    # Keep fetching documents until there are no more pages.
    while next_offset:
        documents, next_offset = cleaned_document_type.bulk_find(limit=limit, offset=next_offset)
        cleaned_documents.extend(documents) # Add the newly fetched documents to the existing list.

    return cleaned_documents
