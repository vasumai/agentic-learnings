"""
Lesson 05 — Semantic Memory & Embeddings
==========================================
Semantic memory lets you store and retrieve information by *meaning*, not
exact keywords.  Under the hood, text is converted to dense vectors (embeddings)
and retrieved by cosine similarity.

Why this matters for agents:
  - An agent can remember facts from previous conversations
  - Instead of "find me the sentence that contains the word 'Tokyo'" (keyword),
    you ask "find me facts about Japanese cities" (semantic)

Comparison to what you know:
  LangGraph  → MemorySaver / SQLite — checkpoints graph state per thread
  CrewAI     → memory=True on Crew — automatic short/long-term/entity memory
  SK         → SemanticTextMemory + MemoryStore — you control save/search explicitly

SK Memory Stack (bottom → top):
  ┌─────────────────────────────────┐
  │  SemanticTextMemory             │  ← high-level: save_information / search
  ├─────────────────────────────────┤
  │  IMemoryStore (VolatileMemory)  │  ← stores (key, vector, metadata) records
  ├─────────────────────────────────┤
  │  EmbeddingGeneratorBase         │  ← turns text → numpy vectors
  └─────────────────────────────────┘

NOTE ON EMBEDDINGS
  Anthropic does not provide a text embedding model.
  Real options: OpenAITextEmbedding, AzureTextEmbedding, or a local model.
  In this lesson we build a *SimpleEmbeddingService* using numpy (already
  installed) so the lesson runs with just your Anthropic key.
  The SK memory API is identical regardless of which embedding service you use —
  swap the service class and nothing else changes.

This lesson covers:
  1. Building a custom EmbeddingGeneratorBase (educational numpy version)
  2. VolatileMemoryStore — in-process vector store (lost on restart)
  3. SemanticTextMemory — save_information() and search()
  4. Memory collections — organise by topic (like database tables)
  5. Real-world pattern: agent knowledge base
  6. What the code looks like with OpenAI embeddings (one diff)
"""

import asyncio
import re
from typing import Annotated
from dotenv import load_dotenv
import os

import numpy as np
from numpy import ndarray

import semantic_kernel as sk
from semantic_kernel.connectors.ai.anthropic import AnthropicChatCompletion
from semantic_kernel.connectors.ai.embedding_generator_base import EmbeddingGeneratorBase
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.memory.volatile_memory_store import VolatileMemoryStore
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin
from semantic_kernel.functions import KernelArguments

load_dotenv()

print("=" * 62)
print("Lesson 05 — Semantic Memory & Embeddings")
print("=" * 62)


# ---------------------------------------------------------------------------
# 0. Custom Embedding Service  (numpy — no extra API key needed)
# ---------------------------------------------------------------------------
# We subclass EmbeddingGeneratorBase which expects:
#   async generate_embeddings(texts: list[str], ...) -> ndarray
#
# Our implementation: character tri-gram TF-IDF vectors + L2 normalisation.
# This is NOT production quality — it's a teaching tool.
# Swap with OpenAITextEmbedding(model="text-embedding-3-small") for real use.

class SimpleEmbeddingService(EmbeddingGeneratorBase):
    """
    Numpy-based embedding service for educational use.
    Uses character tri-gram frequency vectors with cosine-compatible normalisation.
    """

    # Required by AIServiceClientBase (Pydantic model)
    ai_model_id: str = "simple-ngram-embedder"
    service_id: str = "simple_embedder"

    # Vocabulary is built lazily from all texts seen so far
    _vocab: dict = {}
    _dim: int = 512  # fixed vector size (hash trick)

    def _text_to_vector(self, text: str) -> np.ndarray:
        """Character tri-gram hash embedding — deterministic and fast."""
        vec = np.zeros(self._dim, dtype=np.float32)
        text = text.lower()
        # Remove punctuation, split on whitespace for token n-grams
        tokens = re.findall(r"[a-z0-9]+", text)
        for token in tokens:
            # character tri-grams
            for i in range(len(token) - 2):
                ngram = token[i:i+3]
                idx = hash(ngram) % self._dim
                vec[idx] += 1.0
        # L2 normalise so cosine similarity = dot product
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    async def generate_embeddings(
        self,
        texts: list[str],
        settings: PromptExecutionSettings | None = None,
        **kwargs,
    ) -> ndarray:
        """Return a (len(texts), dim) float32 ndarray."""
        return np.stack([self._text_to_vector(t) for t in texts])


# ---------------------------------------------------------------------------
# Setup Kernel + Services
# ---------------------------------------------------------------------------
kernel = sk.Kernel()
kernel.add_service(
    AnthropicChatCompletion(
        ai_model_id="claude-sonnet-4-6",
        api_key=os.environ["ANTHROPIC_API_KEY"],
        service_id="anthropic_chat",
    )
)

# Create the embedding service and memory store
# NOTE: VolatileMemoryStore + SemanticTextMemory are deprecated in SK 1.40.
# The replacement (InMemoryStore + InMemoryCollection + @vectorstoremodel) is
# marked @release_candidate in 1.40 and has known deserialization issues.
# We deliberately use the deprecated API here because it is stable, well-tested,
# and far simpler to read for learning purposes. Migrate when the new API
# reaches a stable release.
embedder = SimpleEmbeddingService()
memory_store = VolatileMemoryStore()       # in-process, lost on restart
memory = SemanticTextMemory(
    storage=memory_store,
    embeddings_generator=embedder,
)


# ---------------------------------------------------------------------------
# 1. Save information & search  (basic)
# ---------------------------------------------------------------------------
# memory.save_information(collection, text, id, description)  ← text before id!
# memory.search(collection, query, limit, min_relevance_score)
#
# IMPORTANT: SK 1.x positional order is (collection, text, id).
# Always use keyword arguments to avoid silent swapped-arg bugs.
# collection = a named bucket (like a DB table)
# id         = unique key within the collection
# text       = the content that gets embedded and returned
# description = optional metadata (not embedded)

async def basic_save_and_search():
    print("\n--- 1. Basic Save & Search ---")

    COLLECTION = "facts"

    # Save some facts
    facts = [
        ("fact01", "The Eiffel Tower is located in Paris, France."),
        ("fact02", "Mount Fuji is the highest mountain in Japan."),
        ("fact03", "The Amazon River is the largest river by discharge volume."),
        ("fact04", "Python was created by Guido van Rossum in 1991."),
        ("fact05", "Semantic Kernel is an open-source SDK from Microsoft."),
        ("fact06", "Tokyo is the capital city of Japan."),
        ("fact07", "The speed of light is approximately 299,792 km/s."),
    ]

    for fact_id, text in facts:
        await memory.save_information(
            collection=COLLECTION,
            id=fact_id,
            text=text,
        )

    print(f"Saved {len(facts)} facts.\n")

    # Search by meaning — not exact words
    queries = [
        "Japanese geography",
        "programming languages history",
        "Microsoft AI tools",
    ]

    for query in queries:
        results = await memory.search(
            collection=COLLECTION,
            query=query,
            limit=2,
            min_relevance_score=0.0,  # 0.0 = return all, sorted by score
        )
        print(f"Query: '{query}'")
        for r in results:
            print(f"  [{r.relevance:.3f}] {r.text}")
        print()


# ---------------------------------------------------------------------------
# 2. Memory Collections  (organise by topic)
# ---------------------------------------------------------------------------
# Collections are independent namespaces.
# Think of them like tables in a database — each has its own vector index.

async def collections_example():
    print("\n--- 2. Memory Collections ---")

    # Save to two separate collections
    sk_facts = [
        ("sk01", "The Kernel is the central orchestrator in Semantic Kernel."),
        ("sk02", "Plugins group related @kernel_function methods."),
        ("sk03", "SemanticTextMemory wraps a vector store and an embedding service."),
    ]
    python_facts = [
        ("py01", "Async functions are defined with async def."),
        ("py02", "Pydantic is a data validation library for Python."),
        ("py03", "Decorators wrap functions to modify their behaviour."),
    ]

    for fid, text in sk_facts:
        await memory.save_information(collection="sk_knowledge", id=fid, text=text)
    for fid, text in python_facts:
        await memory.save_information(collection="python_knowledge", id=fid, text=text)

    # Search only within one collection
    results = await memory.search("sk_knowledge", "how does SK orchestrate functions?", limit=2)
    print("Search in 'sk_knowledge' for 'how does SK orchestrate functions?':")
    for r in results:
        print(f"  [{r.relevance:.3f}] {r.text}")

    results = await memory.search("python_knowledge", "validating data structures", limit=2)
    print("\nSearch in 'python_knowledge' for 'validating data structures':")
    for r in results:
        print(f"  [{r.relevance:.3f}] {r.text}")


# ---------------------------------------------------------------------------
# 3. Updating & removing memories
# ---------------------------------------------------------------------------

async def update_remove_example():
    print("\n--- 3. Update & Remove Memories ---")

    await memory.save_information(collection="updates", id="item01", text="The sky is blue.")
    print("Saved: 'The sky is blue.'")

    # Saving with the same ID overwrites (upsert behaviour)
    await memory.save_information(collection="updates", id="item01", text="The sky appears blue due to Rayleigh scattering.")
    updated = await memory.get("updates", "item01")
    print(f"After update: '{updated.text}'")

    # Remove a specific record
    # SemanticTextMemory has no .remove() — call the underlying store directly
    await memory_store.remove(collection_name="updates", key="item01")
    try:
        gone = await memory.get("updates", "item01")
        print(f"After remove: {gone}")
    except Exception:
        print("After remove: None (record deleted)")  # expected


# ---------------------------------------------------------------------------
# 4. TextMemoryPlugin — let the LLM query memory directly
# ---------------------------------------------------------------------------
# TextMemoryPlugin is a built-in SK plugin that exposes memory.search()
# as a @kernel_function, so the LLM can call it as a tool.
# This is the bridge between memory and the agent loop.
#
# It exposes: recall(input, collection, relevance, limit)
#             remember(input, collection)

async def memory_plugin_example():
    print("\n--- 4. TextMemoryPlugin (LLM calls memory as a tool) ---")

    # Register the memory plugin with the kernel
    kernel.add_plugin(
        TextMemoryPlugin(memory),
        plugin_name="memory",
    )

    # Pre-load some memories
    context_facts = [
        ("ctx01", "The user's name is Srini."),
        ("ctx02", "Srini is learning Semantic Kernel as part of the agentic-learnings project."),
        ("ctx03", "Srini has already completed LangGraph and CrewAI modules."),
        ("ctx04", "Srini prefers concise answers with comparisons to LangGraph and CrewAI."),
    ]
    for fid, text in context_facts:
        await memory.save_information(collection="user_context", id=fid, text=text)

    # Register the memory recall prompt
    kernel.add_function(
        plugin_name="agent",
        function_name="respond_with_memory",
        prompt=(
            "Use the following retrieved context to personalise your answer.\n\n"
            "Context: {{memory.recall ask=$query collection='user_context' relevance='0.0' limit='3'}}\n\n"
            "Question: {{$query}}\n\n"
            "Answer concisely, addressing the user by name if appropriate."
        ),
        prompt_execution_settings=PromptExecutionSettings(max_tokens=200),
    )

    questions = [
        "What frameworks have I already finished?",
        "What am I currently learning?",
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        result = await kernel.invoke(
            plugin_name="agent",
            function_name="respond_with_memory",
            arguments=KernelArguments(query=q),
        )
        print(f"Claude: {result}")


# ---------------------------------------------------------------------------
# 5. How this looks with OpenAI embeddings  (reference only — not run)
# ---------------------------------------------------------------------------
# To use real embeddings, replace SimpleEmbeddingService with:
#
#   from semantic_kernel.connectors.ai.open_ai import OpenAITextEmbedding
#
#   embedder = OpenAITextEmbedding(
#       ai_model_id="text-embedding-3-small",
#       api_key=os.environ["OPENAI_API_KEY"],
#   )
#
# Then pass `embedder` to SemanticTextMemory exactly as in this lesson.
# Everything else — save_information, search, TextMemoryPlugin — is unchanged.
#
# For persistent storage replace VolatileMemoryStore with:
#   from semantic_kernel.connectors.memory.chroma import ChromaMemoryStore
#   from semantic_kernel.connectors.memory.azure_cognitive_search import AzureCognitiveSearchMemoryStore
#   ...dozens of connectors available

def print_production_note():
    print("\n--- 5. Production Embedding Options (reference) ---")
    print("  Embedder swap:  OpenAITextEmbedding('text-embedding-3-small')")
    print("  Persistent store: ChromaMemoryStore, AzureCognitiveSearchMemoryStore,")
    print("                    QdrantMemoryStore, WeaviateMemoryStore, ...")
    print("  SK memory API (save/search/remove) stays identical.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main():
    await basic_save_and_search()
    await collections_example()
    await update_remove_example()
    await memory_plugin_example()
    print_production_note()

    print("\n" + "=" * 62)
    print("Key Takeaways:")
    print("  • Semantic memory retrieves by meaning, not exact keyword")
    print("  • EmbeddingGeneratorBase: implement generate_embeddings() → ndarray")
    print("  • VolatileMemoryStore = in-process store; swap for Chroma/Azure etc.")
    print("  • SemanticTextMemory: save_information(), search(), get(), remove()")
    print("  • Collections = named namespaces — like DB tables per topic")
    print("  • TextMemoryPlugin bridges memory and the LLM tool loop")
    print("  • Swap SimpleEmbeddingService → OpenAITextEmbedding for production")
    print("  • Next: Lesson 06 — Planners (auto function selection)")
    print("=" * 62)


if __name__ == "__main__":
    asyncio.run(main())
