from enum import Enum

from content_agent._3_instruction_dataset.generation.base.vector import VectorBaseDocument
from content_agent._3_instruction_dataset.generation.types import DataCategory


# Defines the two types of datasets generated in this stage.
#
# INSTRUCTION:
#   instruction + answer
#
# PREFERENCE:
#   instruction + chosen answer + rejected answer
class DatasetType(Enum):
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"


class InstructDatasetSample(VectorBaseDocument):
    instruction: str
    answer: str

    class Config:
        category = DataCategory.INSTRUCT_DATASET_SAMPLES


class PreferenceDatasetSample(VectorBaseDocument):
    instruction: str
    rejected: str
    chosen: str

    class Config:
        category = DataCategory.PREFERENCE_DATASET_SAMPLES


class InstructDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[InstructDatasetSample]

    class Config:
        category = DataCategory.INSTRUCT_DATASET

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def to_records(self) -> list[dict]:
        return [
            {
                "instruction": sample.instruction,
                "output": sample.answer,
            }
            for sample in self.samples
        ]


class TrainTestSplit(VectorBaseDocument):
    train: dict
    test: dict
    test_split_size: float

    def to_records(self, flatten: bool = False) -> dict:
        train_records = {
            category.value: dataset.to_records()
            for category, dataset in self.train.items()
        }

        test_records = {
            category.value: dataset.to_records()
            for category, dataset in self.test.items()
        }

        if flatten:
            train_records = [
                record
                for records in train_records.values()
                for record in records
            ]

            test_records = [
                record
                for records in test_records.values()
                for record in records
            ]

        return {
            "train": train_records,
            "test": test_records,
        }


class InstructTrainTestSplit(TrainTestSplit):
    train: dict[DataCategory, InstructDataset]
    test: dict[DataCategory, InstructDataset]
    test_split_size: float

    class Config:
        category = DataCategory.INSTRUCT_DATASET


class PreferenceDataset(VectorBaseDocument):
    category: DataCategory
    samples: list[PreferenceDatasetSample]

    class Config:
        category = DataCategory.PREFERENCE_DATASET

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def to_records(self) -> list[dict]:
        return [
            {
                "prompt": sample.instruction,
                "rejected": sample.rejected,
                "chosen": sample.chosen,
            }
            for sample in self.samples
        ]


class PreferenceTrainTestSplit(TrainTestSplit):
    train: dict[DataCategory, PreferenceDataset]
    test: dict[DataCategory, PreferenceDataset]
    test_split_size: float

    class Config:
        category = DataCategory.PREFERENCE_DATASET


def build_dataset(
    dataset_type,
    *args,
    **kwargs,
) -> InstructDataset | PreferenceDataset:
    if dataset_type == DatasetType.INSTRUCTION:
        return InstructDataset(*args, **kwargs)

    elif dataset_type == DatasetType.PREFERENCE:
        return PreferenceDataset(*args, **kwargs)

    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")