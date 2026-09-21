from content_agent._3_instruction_preference_dataset.generation.base.vector import VectorBaseDocument
from content_agent._3_instruction_preference_dataset.generation.cleaned_documents import CleanedDocument
from content_agent._3_instruction_preference_dataset.generation.types import DataCategory

# Defines the prompt templates used for generating structured training examples.

class Prompt(VectorBaseDocument):
    template: str
    input_variables: dict
    content: str
    num_tokens: int | None = None

    class Config:
        category = DataCategory.PROMPT

class GenerateDatasetSamplesPrompt(Prompt):
    data_category: DataCategory
    document: CleanedDocument
