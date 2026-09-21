from typing import Any

from typing_extensions import Annotated
from zenml import ArtifactConfig, get_step_context, step

from content_agent._3_instruction_preference_dataset.generation import generation
from content_agent._3_instruction_preference_dataset.generation.dataset import DatasetType, InstructTrainTestSplit
from content_agent._3_instruction_preference_dataset.generation.prompt import GenerateDatasetSamplesPrompt
from content_agent._3_instruction_preference_dataset.generation.types import DataCategory

# ZenML step that generates preference datasets from the given prompts.
#
# A preference dataset contains multiple candidate answers for an instruction,
# where the model learns which answer is preferred over another one.
@step
def generate_instruction_dataset(    
    prompts: Annotated[dict[DataCategory, list[GenerateDatasetSamplesPrompt]], "prompts"], # Prompts grouped by data category.
    test_split_size: Annotated[float, "test_split_size"], # Percentage of generated samples that will be used for the test set.
    mock: Annotated[bool, "mock_generation"] = False, # If True, generate mock data instead of calling the language model.
) -> Annotated[
    # The output contains the generated instruction dataset
    # split into training and test sets.
    InstructTrainTestSplit,
    # Configuration for the ZenML artifact produced by this step.
    ArtifactConfig(
        name="instruct_datasets",
        tags=["dataset", "instruct", "cleaned"],
    ),
]:

    # Generate instruction-response samples from the prompts
    # and split them into training and test datasets.
    dataset_generator = generation.get_dataset_generator(DatasetType.INSTRUCTION)
    datasets = dataset_generator.generate(prompts, test_size=test_split_size, mock=mock)


    # Get the current ZenML step context so we can attach
    # additional metadata to the generated dataset artifact.
    step_context = get_step_context()
    step_context.add_output_metadata(output_name="instruct_datasets", metadata=_get_metadata_instruct_dataset(datasets))


    # Return the generated train/test datasets.
    return datasets


def _get_metadata_instruct_dataset(datasets: InstructTrainTestSplit) -> dict[str, Any]:
    instruct_dataset_categories = list(datasets.train.keys()) # Get the categories available in the training dataset.
    train_num_samples = {category: instruct_dataset.num_samples for category, instruct_dataset in datasets.train.items()} # Count the number of training samples for each category.
    test_num_samples = {category: instruct_dataset.num_samples for category, instruct_dataset in datasets.test.items()}


    # Return metadata that ZenML will attach to the dataset artifact.
    return {
        "data_categories": instruct_dataset_categories,
        "test_split_size": datasets.test_split_size,
        "train_num_samples_per_category": train_num_samples,
        "test_num_samples_per_category": test_num_samples,
    }
