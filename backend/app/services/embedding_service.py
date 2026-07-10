"""
Embedding Service - Converts text into vector representations.

This is the bridge between human-readable text and machine-searchable vectors.
Every piece of text that enters our RAG system passes through here.

The embedding model is NOT the same as the LLM:
- LLM (Qwen): takes text in, generates text out
- Embedding model (MiniLM): takes text in, generates numbers (vectors) out

They serve completely different purposes.
"""

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Generates embeddings using a local model (no API calls, no cost).

    Why local instead of OpenAI's embedding API?
    - Free: no per-token charges
    - Fast: no network latency
    - Private: document text never leaves your machine
    - Offline: works without internet

    Trade-off: slightly lower quality than OpenAI's latest models,
    but for RAG retrieval, the difference is negligible.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.

        'all-MiniLM-L6-v2' produces 384-dimensional vectors.
        What does 384-dimensional mean?
        - Each text gets converted to a list of 384 numbers
        - More dimensions = more nuance in meaning, but more storage
        - 384 is a sweet spot: good quality, reasonable size
        - OpenAI uses 1536 dimensions (higher quality, 4x storage)

        First run downloads the model (~80MB). Subsequent runs use cache.
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_text(self, text: str) -> list[float]:
        """
        Convert a single text string into a vector.

        Args:
            text: Any text (a chunk, a query, a sentence)

        Returns:
            A list of floats (the embedding vector)
            Example: [0.12, -0.45, 0.78, ...] (384 numbers)
        """
        # .encode() does all the work:
        # 1. Tokenizes the text (splits into subwords)
        # 2. Runs it through the neural network
        # 3. Returns the vector representation
        # .tolist() converts numpy array to plain Python list
        embedding = self.model.encode(text)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Convert multiple texts into vectors at once.

        Why batch instead of calling embed_text in a loop?
        - GPU/CPU can process multiple texts in parallel
        - 100 texts in a batch is MUCH faster than 100 individual calls
        - For our 1120 chunks, batch processing is essential

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors (same order as input)
        """
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings.tolist()