from typing import Any

from loguru import logger
from pymongo import MongoClient
from typing_extensions import Annotated
from zenml import step

from content_agent.utilities.settings import settings


@step
def save_to_mongodb(
    dataset: Annotated[Any, "dataset_split"],
    dataset_id: Annotated[str, "dataset_id"],
) -> None:
    logger.info(f"Saving dataset {dataset_id} to MongoDB.")

    # Check dataset structure before converting it to records.
    logger.info(f"Dataset type: {type(dataset).__name__}")
    logger.info(f"Train categories: {list(dataset.train.keys())}")
    logger.info(f"Test categories: {list(dataset.test.keys())}")

    for category, train_dataset in dataset.train.items():
        logger.info(
            f"Train category '{category}': "
            f"{train_dataset.num_samples} samples"
        )

    for category, test_dataset in dataset.test.items():
        logger.info(
            f"Test category '{category}': "
            f"{test_dataset.num_samples} samples"
        )

    # Convert the dataset into MongoDB-compatible records.
    records = dataset.to_records(flatten=True)

    train_records = records["train"]
    test_records = records["test"]

    logger.info(
        f"Prepared records: "
        f"{len(train_records)} train, "
        f"{len(test_records)} test"
    )

    # Do not silently succeed with an empty dataset.
    if not train_records and not test_records:
        raise ValueError(
            f"Dataset '{dataset_id}' is empty. "
            "No train or test records were generated."
        )

    client = MongoClient(settings.DATABASE_HOST)

    try:
        db = client[settings.DATABASE_NAME]

        train_collection = db["instruction_dataset_train"]
        test_collection = db["instruction_dataset_test"]

        # Replace previous dataset contents.
        train_collection.delete_many({})
        test_collection.delete_many({})

        if train_records:
            train_collection.insert_many(train_records)

        if test_records:
            test_collection.insert_many(test_records)

        logger.info(
            f"Dataset {dataset_id} saved to MongoDB: "
            f"{len(train_records)} train records, "
            f"{len(test_records)} test records."
        )

    finally:
        client.close()