from abc import ABC, abstractmethod

import tiktoken
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from loguru import logger

from content_agent.utilities import misc
from content_agent._3_instruction_preference_dataset.generation.cleaned_documents import CleanedDocument
from content_agent._3_instruction_preference_dataset.generation.dataset import DatasetType, TrainTestSplit , build_dataset , PreferenceDatasetSample , InstructDatasetSample , InstructDataset , PreferenceDataset
from content_agent._3_instruction_preference_dataset.generation.prompt import GenerateDatasetSamplesPrompt, Prompt
from content_agent._3_instruction_preference_dataset.generation.types import DataCategory
from content_agent.utilities.settings import settings

from . import constants
from . import utils as generation_utils
from .output_parsers import ListPydanticOutputParser

# Base class that contains the common logic for generating
# both Instruction and Preference datasets.
#
# The subclasses only define:
# - which type of dataset they generate
# - the prompt used to generate samples
# - how the generated datasets are post-processed

class DatasetGenerator(ABC):
    # Tokenizer used to count and limit the number of tokens
    # sent to the OpenAI model.
    tokenizer = tiktoken.encoding_for_model(settings.OPENAI_MODEL_ID)
    dataset_type: DatasetType | None = None # The subclass sets this to INSTRUCTION or PREFERENCE.

    system_prompt_template = """You are a helpful assistant who generates {dataset_format} based on the given context. \
Provide your response in JSON format.
"""
    prompt_template_str: str | None = None

    @classmethod
    def get_system_prompt(cls) -> Prompt:
        # A dataset type must be defined before creating the system prompt.
        assert cls.dataset_type is not None, "Dataset type must be set before calling get_system_prompt()"


        # Instruction datasets contain pairs:
        # instruction + answer
        #
        # Preference datasets contain triples:
        # instruction + rejected + chosen
        dataset_format = (
            "instruction-answer pairs" if cls.dataset_type == DatasetType.INSTRUCTION else "instruction-answer triples"
        )
        input_variables = {
            "dataset_format": dataset_format,
        }
        system_prompt = cls.system_prompt_template.format(**input_variables)

        return Prompt(
            template=cls.system_prompt_template,
            input_variables=input_variables,
            content=system_prompt,
        )

    @classmethod
    def get_prompts(cls, documents: list[CleanedDocument]) -> dict[DataCategory, list[GenerateDatasetSamplesPrompt]]:
        # Extract the relevant parts of the cleaned documents
        # before creating prompts.
        documents = generation_utils.extract_substrings(documents)


        # Group documents according to their data category.
        grouped_prompts = {}
         # Create one generation prompt for each document.
        grouped_cleaned_documents = CleanedDocument.group_by_category(documents)
        for category, category_documents in grouped_cleaned_documents.items():
            category_prompts = [cls.get_prompt(document) for document in category_documents]
            grouped_prompts[category] = category_prompts

        return grouped_prompts

    @classmethod
    def get_prompt(cls, document: CleanedDocument) -> GenerateDatasetSamplesPrompt:
        # Every concrete dataset generator must define
        # its own prompt template.
        assert cls.prompt_template_str is not None, "Prompt template must be set before calling get_prompt()"

        # Get the category of the current document.
        data_category = document.get_category()

        # Convert the Jinja2 prompt template into a LangChain PromptTemplate.
        prompt_template = PromptTemplate.from_template(
            template=cls.prompt_template_str,
            template_format="jinja2",
        )

        # The cleaned document is inserted into the {{extract}} variable
        # of the prompt template.
        input_variables = {
            "extract": document.content,
        }
        prompt = prompt_template.format(**input_variables)
        prompt_tokens = cls.tokenizer.encode(prompt) # Count the number of tokens in the generated prompt.
        if len(prompt_tokens) > settings.OPENAI_MAX_TOKEN_WINDOW:
            prompt_tokens = prompt_tokens[: settings.OPENAI_MAX_TOKEN_WINDOW]
            prompt = cls.tokenizer.decode(prompt_tokens)


        logger.info(
            f"document type: {type(document)}"
        )

        logger.info(
            f"document module: {type(document).__module__}"
        )

        logger.info(
            f"expected CleanedDocument: {CleanedDocument}"
        )

        logger.info(
            f"expected module: {CleanedDocument.__module__}"
        )

        logger.info(
            f"isinstance result: {isinstance(document, CleanedDocument)}"
        )

        # Store the generated prompt together with useful metadata.
        prompt = GenerateDatasetSamplesPrompt(
            template=prompt_template.template,
            input_variables=input_variables,
            content=prompt,
            num_tokens=len(prompt_tokens),
            data_category=data_category,
            document=document,
        )

        return prompt

    @classmethod
    def generate(
        cls,
        prompts: dict[DataCategory, list[GenerateDatasetSamplesPrompt]],
        test_size: float = 0.2,
        mock: bool = False,
    ) -> TrainTestSplit:

        # The dataset type must be defined by the subclass.
        assert cls.dataset_type is not None, (
            "Dataset type must be set before calling generate()"
        )

        # Convert our prompts into the message format expected by LangChain.
        def _to_langchain(
            prompt: GenerateDatasetSamplesPrompt,
        ) -> list[BaseMessage]:
            messages = [
                # System message tells the LLM what kind of dataset to generate.
                SystemMessage(content=cls.get_system_prompt().content),
                # Human message contains the actual prompt and document extract.
                HumanMessage(content=prompt.content),
            ]

            return messages

        # During development/testing, FakeListLLM can be used
        # instead of making real API calls.
        if mock:
            llm = FakeListLLM(
                responses=[constants.get_mocked_response(cls.dataset_type)]
            )
        else:
            # Real dataset generation requires an OpenAI API key.
            assert settings.OPENAI_API_KEY is not None, (
                "OpenAI API key must be set to generate datasets"
            )

            llm = ChatOpenAI(
                model=settings.OPENAI_MODEL_ID,
                api_key=settings.OPENAI_API_KEY,
                max_tokens=(
                    2000
                    if cls.dataset_type == DatasetType.PREFERENCE
                    else 1200
                ),
                temperature=0.7,
            )

        # Parser converts the LLM's JSON output into our Pydantic
        # dataset sample objects.
        parser = ListPydanticOutputParser(
            pydantic_object=cls._get_dataset_sample_type()
        )

        # LangChain pipeline:
        # prompts -> LLM -> output parser -> structured dataset samples
        chain = llm | parser

        datasets = {}

        for category, category_prompts in prompts.items():
            logger.info(
                f"Category '{category}': {len(category_prompts)} prompts"
            )

            langchain_category_prompts = [
                _to_langchain(prompt)
                for prompt in category_prompts
            ]

            # Split prompts into batches of 24.
            batches = misc.batch(
                langchain_category_prompts,
                size=24,
            )

            flattened_dataset_samples = []

            for batch in batches:
                try:
                    # Run the LLM + parser chain on the batch.
                    batched_dataset_samples = chain.batch(
                        batch,
                        stop=None,
                    )

                    # Flatten the results of different batches
                    # into one list of samples.
                    for dataset_sample_batch in batched_dataset_samples:
                        flattened_dataset_samples.extend(
                            dataset_sample_batch
                        )

                except OutputParserException as e:
                    logger.error(
                        "Failed to parse the output JSON for a batch "
                        f"for category {category}"
                    )
                    logger.error(f"Raw LLM output:\n{e.llm_output}")
                    raise

            logger.info(
                f"Category '{category}': "
                f"{len(flattened_dataset_samples)} generated samples"
            )

            # Convert the generated samples into our domain dataset object.
            dataset = build_dataset(
                dataset_type=cls.dataset_type,
                category=category,
                samples=flattened_dataset_samples,
            )

            datasets[category] = dataset

            logger.info(
                f"Generated {len(dataset.samples)} samples "
                f"for category '{category}'."
            )

        # Apply dataset-specific post-processing and create
        # the final train/test split.
        processed_datasets = cls.post_process_datasets(
            datasets,
            test_size=test_size,
        )

        return processed_datasets

    @classmethod
    def _get_dataset_sample_type(cls,) -> type[InstructDatasetSample] | type[PreferenceDatasetSample]:
        return (
            InstructDatasetSample
            if cls.dataset_type == DatasetType.INSTRUCTION
            else PreferenceDatasetSample
        )

    @classmethod
    @abstractmethod
    def post_process_datasets(cls, datasets: dict[DataCategory, InstructDataset], test_size: float) -> TrainTestSplit:
        pass

# Generator specifically for Instruction datasets.
class InstructionDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.INSTRUCTION

    prompt_template_str = """Based on the following extract, generate five instruction-answer pairs. Each instruction \
must ask to write about a specific topic contained in the context. Each answer \
must provide a relevant paragraph based on the information found in the \
context. Only use concepts from the context to generate the instructions. \
Instructions must never explicitly mention a context, a system, a course, or an extract. \
Instructions must be self-contained and general. \
Answers must imitate the writing style of the context. \
Do not use LaTeX notation or backslashes in the generated answers. \

Example instruction: Explain the concept of an LLM Twin. \
Example answer: An LLM Twin is essentially an AI character that mimics your writing style, personality, and voice. \
It's designed to write just like you by incorporating these elements into a language model. \
The idea is to create a digital replica of your writing habits using advanced AI techniques. \

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {"instruction": "...", "answer": "..."},
    ...
]

Extract:
{{extract}}
"""

    @classmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, InstructDataset], test_size: float
    ) -> TrainTestSplit:
        train_test_split = generation_utils.create_instruct_train_test_split(
            datasets, test_size=test_size, random_state=42
        )

        return train_test_split


class PreferenceDatasetGenerator(DatasetGenerator):
    dataset_type = DatasetType.PREFERENCE

    prompt_template_str = """Based on the following extract, generate five instruction-answer triples. Each triple should consist of:
1. An instruction asking about a specific topic in the context.
2. A generated answer that attempts to answer the instruction based on the context, named as 'rejected'.
3. An extracted answer that is a relevant excerpt directly from the given context, named as 'chosen'.

Instructions must be self-contained and general, without explicitly mentioning a context, system, course, or extract.

Important:
- Ensure that the extracted answer, the chosen one, is a verbatim copy from the context, including all punctuation and apostrophes.
- Do not add any ellipsis (...) or [...]  to indicate skipped text in the extracted answer.
- If the relevant text is not continuous, use two separate sentences from the context instead of skipping text.

Structure the answer in JSON format, ready to be loaded in Python by json.loads(), as a list of objects.
Do not add any extra characters and provide your response in JSON format with the following structure:
[
    {
        "instruction": "...",
        "rejected": "...",
        "chosen": "..."
    },
    ...
]

Extract:
{{extract}}
"""

    @classmethod
    def post_process_datasets(
        cls, datasets: dict[DataCategory, PreferenceDataset], test_size: float
    ) -> TrainTestSplit:
        datasets = generation_utils.filter_short_answers(datasets)
        datasets = generation_utils.filter_answer_format(datasets)

        remaining_samples = sum([dataset.num_samples for dataset in datasets.values()])
        logger.info(
            f"Filtered out short answers and answers with incorrect format. Remaining samples: {remaining_samples}"
        )

        train_test_split = generation_utils.create_preference_train_test_split(
            datasets, test_size=test_size, random_state=42
        )

        return train_test_split


def get_dataset_generator(dataset_type: DatasetType) -> type[DatasetGenerator]:
    if dataset_type == DatasetType.INSTRUCTION:
        return InstructionDatasetGenerator
    elif dataset_type == DatasetType.PREFERENCE:
        return PreferenceDatasetGenerator
    else:
        raise ValueError(f"Invalid dataset type: {dataset_type}")
