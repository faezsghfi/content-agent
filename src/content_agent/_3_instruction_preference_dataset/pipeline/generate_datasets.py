from pathlib import Path

from zenml.utils import source_utils

source_utils.set_custom_source_root(
    str(Path(__file__).resolve().parents[3])
)

from zenml import pipeline

from content_agent._3_instruction_preference_dataset.generation.dataset import DatasetType

from content_agent._3_instruction_preference_dataset.generation.generate_instruction_dataset import generate_instruction_dataset
from content_agent._3_instruction_preference_dataset.generation.generate_preference_dataset import generate_preference_dataset
from content_agent._3_instruction_preference_dataset.generation.create_prompts import create_prompts

from content_agent._3_instruction_preference_dataset.extraction.query_feature_store import query_feature_store

from content_agent._3_instruction_preference_dataset.storage.save_to_mongodb import save_to_mongodb 
from content_agent._3_instruction_preference_dataset.storage.push_to_huggingface import push_to_huggingface 

@pipeline(enable_cache=False)
def generate_datasets(
    dataset_type: DatasetType = DatasetType.INSTRUCTION,
    test_split_size: float = 0.1,
    save_in_storage: bool = False,
    publish_in_huggingface : bool = False,
    dataset_id: str | None = None,
    mock: bool = False,
    wait_for: str | list[str] | None = None,
) -> None:
    cleaned_documents = query_feature_store(after=wait_for)
    prompts = create_prompts(documents=cleaned_documents, dataset_type=dataset_type)
    if dataset_type == DatasetType.INSTRUCTION:
        dataset = generate_instruction_dataset(prompts=prompts, test_split_size=test_split_size, mock=mock)
    elif dataset_type == DatasetType.PREFERENCE:
        dataset = generate_preference_dataset(prompts=prompts, test_split_size=test_split_size, mock=mock)
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")

    if save_in_storage:
        save_to_mongodb(dataset=dataset, dataset_id=dataset_id)

    if publish_in_huggingface:
        push_to_huggingface(dataset=dataset, dataset_id=dataset_id)
