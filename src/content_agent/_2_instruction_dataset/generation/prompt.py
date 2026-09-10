from src.content_agent._4_rag.base import VectorBaseDocument
from content_agent._2_instruction_dataset.generation.cleaned_documents import CleanedDocument
from content_agent._2_instruction_dataset.generation.types import DataCategory


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
