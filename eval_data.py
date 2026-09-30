"""
Shared evaluation data and metrics.

The 12-question ground-truth set and the three metrics are copied from
findoc_qa.py, so the OpenSearch experiments are scored exactly like the
original ChromaDB experiments. (findoc_qa.py runs its whole pipeline on
import, so it cannot be imported directly.)
"""

import re

GROUND_TRUTH = [
    {"question": "What was Apple's total net sales in fiscal year 2024?",
     "gold_answer": "$391,035 million", "keywords": ["391,035", "391"], "category": "numerical"},
    {"question": "How much did Apple spend on R&D in 2024?",
     "gold_answer": "$31,370 million", "keywords": ["31,370", "31.4"], "category": "numerical"},
    {"question": "What was Apple's net income in 2024?",
     "gold_answer": "$93,736 million", "keywords": ["93,736", "93.7"], "category": "numerical"},
    {"question": "What is Apple's total number of employees?",
     "gold_answer": "Approximately 164,000", "keywords": ["164,000", "164"], "category": "numerical"},
    {"question": "Who is Apple's CEO?",
     "gold_answer": "Tim Cook", "keywords": ["Tim Cook", "Cook"], "category": "factual"},
    {"question": "What was Apple's gross margin in 2024?",
     "gold_answer": "46.2%", "keywords": ["46.2", "46%", "180,683"], "category": "numerical"},
    {"question": "How much cash did Apple return to shareholders in 2024?",
     "gold_answer": "Over $100 billion through dividends and share repurchases",
     "keywords": ["dividend", "repurchas", "buyback", "100 billion", "94,949", "15,234"],
     "category": "numerical"},
    {"question": "What are Apple's main product categories?",
     "gold_answer": "iPhone, Mac, iPad, Wearables/Home/Accessories, Services",
     "keywords": ["iPhone", "Mac", "iPad", "Wearables", "Services"], "category": "factual"},
    {"question": "In which countries does Apple have retail stores?",
     "gold_answer": "Multiple countries including the U.S., with stores in 26 countries",
     "keywords": ["retail", "store", "country", "countries"], "category": "factual"},
    {"question": "What was iPhone revenue in 2024?",
     "gold_answer": "$201,183 million", "keywords": ["201,183", "201"], "category": "numerical"},
    {"question": "What was Services revenue in 2024?",
     "gold_answer": "$96,169 million", "keywords": ["96,169", "96"], "category": "numerical"},
    {"question": "What credit rating does Apple have?",
     "gold_answer": "AA+ from S&P / Aa1 from Moody's",
     "keywords": ["AA", "Aa1", "credit", "rating"], "category": "factual"},
]

SYSTEM_PROMPT = """You are a financial document analyst. Answer the user's question based ONLY on the provided context from the document.
Rules:
- Cite specific sources using [Source N] format
- If the context doesn't contain enough information, say so clearly
- For numerical data, quote exact figures from the document
- Be concise but thorough"""


def eval_keyword_hit(answer: str, keywords: list[str]) -> bool:
    """True if the answer contains at least one expected keyword."""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in keywords)


def eval_faithfulness(answer: str, source_texts: list[str]) -> float:
    """Share of numbers in the answer that also appear in the retrieved sources."""
    numbers_in_answer = set(re.findall(r'[\d,]+\.?\d*', answer))
    if not numbers_in_answer:
        return 1.0
    source_numbers = set(re.findall(r'[\d,]+\.?\d*', " ".join(source_texts).lower()))
    return len(numbers_in_answer & source_numbers) / len(numbers_in_answer)


def eval_relevance(cosine_similarities: list[float]) -> float:
    """Mean cosine similarity between the query and the retrieved chunks."""
    return sum(cosine_similarities) / len(cosine_similarities) if cosine_similarities else 0.0


def build_context(documents: list[str], metadatas: list[dict], distances: list[float]) -> str:
    """Same context format as the original rag_answer()."""
    parts = []
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        parts.append(f"[Source {i+1}, Page {meta['page_num']}] (relevance: {1-dist:.2f})\n{doc}")
    return "\n\n---\n\n".join(parts)
