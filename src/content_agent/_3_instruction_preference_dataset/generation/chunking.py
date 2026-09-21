import re

from langchain_text_splitters import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter

from content_agent._5_rag.embeddings import EmbeddingModelSingleton

embedding_model = EmbeddingModelSingleton()

# Splits documents into smaller chunks suitable for instruction dataset generation.

# Splits text first by paragraphs and then by the embedding model's token limit.
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:

    # Split the text into sections using paragraph breaks.
    character_splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=chunk_size, chunk_overlap=0)
    text_split_by_characters = character_splitter.split_text(text)

    # Further split each section according to the maximum number of
    # tokens supported by the embedding model.
    token_splitter = SentenceTransformersTokenTextSplitter(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=embedding_model.max_input_length,
        model_name=embedding_model.model_id,
    )
    chunks_by_tokens = []
    for section in text_split_by_characters:
        chunks_by_tokens.extend(token_splitter.split_text(section))

    return chunks_by_tokens

# Alias for chunk_article().
def chunk_document(text: str, min_length: int, max_length: int) -> list[str]:
    """Alias for chunk_article()."""

    return chunk_article(text, min_length, max_length)

# Splits an article into sentence-based chunks constrained by character length.
def chunk_article(text: str, min_length: int, max_length: int) -> list[str]:
    # Split the article into individual sentences.
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)

    extracts = []
    current_chunk = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) <= max_length:
            current_chunk += sentence + " "
        else:
            if len(current_chunk) >= min_length:
                extracts.append(current_chunk.strip())
            current_chunk = sentence + " "

    if len(current_chunk) >= min_length:
        extracts.append(current_chunk.strip())

    return extracts
