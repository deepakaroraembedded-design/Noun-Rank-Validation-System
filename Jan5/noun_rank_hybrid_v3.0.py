"""
Noun Rank Hybrid Evaluator v3.0
- Hallucination (H): Compare LLM to DOCUMENT (source truth)
- Coverage/Entity/Semantic (C, E, S): Compare LLM to EXPERT (gold standard)
- Readability (R): LLM summary only
- Baseline: Expert summary score for comparison
- Accuracy: LLM score / Baseline score * 100%

NEW in v3.0:
- Prints hallucinated words per document
- Shows hallucination summary by category (entity, proper, nouns, temporal, numeric)
- Aggregates all hallucinated terms in final report

This properly measures:
- H: Did LLM invent facts not in source document?
- C/E/S: How close is LLM to expert quality?
- Accuracy: Overall performance vs expert baseline

Usage:
    python noun_rank_hybrid_v3.0.py --limit 100
    python noun_rank_hybrid_v3.0.py --model mistralai/Mistral-7B-Instruct-v0.3 --context 8192

Author: Deepak Arora
Date: January 2026
"""

import os
import sqlite3
import time
import json
import logging
import requests
import numpy as np
import threading
import gc
import re
from queue import Queue
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # === LLM Settings ===
    'LLM_HOST': 'http://localhost:8000',
    'LLM_MODEL': 'mistralai/Mistral-7B-Instruct-v0.3',
    'LLM_API_KEY': None,
    
    # === Generation Settings ===
    'MAX_TOKENS': 1000,
    'TEMPERATURE': 0.1,
    'TOP_P': 0.9,
    'TIMEOUT': 180,
    'MAX_INPUT_WORDS': 2500,
    'CHUNK_SIZE': 2000,
    
    # === Database ===
    'DB_PATH': 'noun_rank_hybrid_v3.db',
    
    # === NLP Models ===
    'SPACY_MODEL': 'en_core_web_sm',
    'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
    
    # === Algorithm ===
    'HALLUCINATION_THRESHOLD': 0.001,
    
    # === Readability Settings ===
    'OPTIMAL_SENTENCE_LENGTH_MIN': 15,
    'OPTIMAL_SENTENCE_LENGTH_MAX': 20,
    
    # === Version ===
    'ALGORITHM_VERSION': 'v3.0-hybrid',
}

WEIGHTS = {
    'H': {'entity': 0.30, 'proper': 0.25, 'nouns': 0.20, 'numeric': 0.15, 'temporal': 0.10},
    'C': {'key': 0.35, 'entity': 0.30, 'numeric': 0.20, 'temporal': 0.15},
    'R': {'flesch': 0.30, 'variance': 0.25, 'transitions': 0.25, 'avg_length': 0.20},
    'FINAL': {'H': 0.20, 'C': 0.20, 'F': 0.15, 'E': 0.15, 'S': 0.10, 'R': 0.20},
    'FINAL_NO_H': {'C': 0.24, 'F': 0.19, 'E': 0.19, 'S': 0.14, 'R': 0.24},
    'ENTITY': {'PERSON': 1.0, 'ORG': 1.0, 'GPE': 1.0, 'DATE': 0.8, 'MONEY': 0.8}
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# GPU Memory Management
# =============================================================================

def clear_context():
    """Clear GPU memory and Python garbage after each run."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractedTerms:
    nouns: Set[str] = field(default_factory=set)
    proper: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    temporal: Set[str] = field(default_factory=set)
    numeric: Set[str] = field(default_factory=set)
    freq: Dict[str, int] = field(default_factory=dict)
    
    @property
    def all_terms(self) -> Set[str]:
        return self.nouns | self.proper | self.entities | self.temporal | self.numeric
    
    def get_key_terms(self, min_freq: int = 2) -> Set[str]:
        return {t for t in self.all_terms if self.freq.get(t, 0) >= min_freq}


@dataclass
class ComponentScores:
    # Main components
    H: float = 0.0   # Hallucination: LLM vs Document
    C: float = 0.0   # Coverage: LLM vs Expert
    F: float = 0.0   # Frequency: LLM vs Expert
    E: float = 0.0   # Entity: LLM vs Expert
    S: float = 0.0   # Semantic: LLM vs Expert
    R: float = 0.0   # Readability: LLM only
    
    # Hallucination sub-components (vs Document)
    H_entity: float = 0.0
    H_proper: float = 0.0
    H_nouns: float = 0.0
    H_temporal: float = 0.0
    H_numeric: float = 0.0
    
    # Coverage sub-components (vs Expert)
    C_key: float = 0.0
    C_entity: float = 0.0
    C_temporal: float = 0.0
    C_numeric: float = 0.0
    
    # Readability sub-components
    R_flesch: float = 0.0
    R_variance: float = 0.0
    R_transitions: float = 0.0
    R_avg_length: float = 0.0
    
    final_score: float = 0.0


@dataclass
class EvalResult:
    doc_id: str
    baseline_score: float  # Expert baseline score
    llm_score: float       # LLM score
    accuracy_pct: float    # LLM / Baseline * 100
    H: float               # Hallucination (vs Document)
    C: float               # Coverage (vs Expert)
    S: float               # Semantic (vs Expert)
    expert_R: float        # Expert readability
    llm_R: float           # LLM readability
    status: str = ""
    llm_summary: str = ""
    expert_summary: str = ""
    hallucinated_words: Dict = field(default_factory=dict)  # Hallucinated words by category


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """Client for LLM via vLLM OpenAI-compatible API."""
    
    SUMMARIZE_PROMPT = """You are an expert government document summarizer. Create a comprehensive, factual summary of the following report.

CRITICAL RULES:
1. ONLY include information explicitly stated in the document
2. NEVER invent or fabricate facts, names, dates, or numbers
3. Preserve exact names of organizations, people, and places
4. Include specific dates, monetary amounts, and statistics verbatim
5. If uncertain about a detail, omit it rather than guess

FORMAT:
- Write in clear, professional prose
- Target length: approximately 550 words
- Focus on key findings, recommendations, and conclusions

DOCUMENT:
{document}

SUMMARY:"""

    def __init__(self, host: str, model: str, api_key: str = None):
        self.host = host.rstrip('/')
        self.model = model
        self.api_key = api_key
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to vLLM server."""
        try:
            response = requests.get(f"{self.host}/v1/models", timeout=10)
            if response.status_code == 200:
                logger.info(f"✓ Connected to vLLM at {self.host}")
                models = response.json().get('data', [])
                if models:
                    logger.info(f"  Available models: {[m['id'] for m in models]}")
            else:
                logger.warning(f"vLLM returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Cannot connect to vLLM: {e}")
            raise ConnectionError(f"Cannot connect to vLLM at {self.host}")
    
    def summarize(self, document: str, max_tokens: int = 1000) -> str:
        """Generate summary with chunked processing for long docs."""
        words = document.split()
        max_input = CONFIG['MAX_INPUT_WORDS']
        chunk_size = CONFIG['CHUNK_SIZE']
        
        if len(words) <= max_input:
            return self._call_llm(document, max_tokens)
        
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        logger.info(f"Document split into {len(chunks)} chunks ({len(words)} words)")
        
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  Summarizing chunk {i+1}/{len(chunks)}")
            summary = self._call_llm(chunk, max_tokens=400)
            chunk_summaries.append(summary)
        
        combined = "\n\n".join(chunk_summaries)
        
        if len(combined.split()) <= 800:
            return combined
        
        final_prompt = f"Combine these summaries into one coherent 550-word summary:\n\n{combined}"
        return self._call_llm(final_prompt, max_tokens=800, use_template=False)
    
    def _call_llm(self, text: str, max_tokens: int = 1000, use_template: bool = True) -> str:
        """Make LLM API call."""
        if use_template:
            prompt = self.SUMMARIZE_PROMPT.format(document=text)
        else:
            prompt = text
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": CONFIG['TEMPERATURE'],
            "top_p": CONFIG.get('TOP_P', 0.9),
        }
        
        try:
            response = requests.post(
                f"{self.host}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=CONFIG['TIMEOUT']
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            raise
        except Exception as e:
            logger.error(f"LLM error: {e}")
            raise


# =============================================================================
# NLP Models (Shared between threads)
# =============================================================================

class NLPModels:
    """Thread-safe NLP models container."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def load(self):
        """Load models (singleton)."""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            import spacy
            from sentence_transformers import SentenceTransformer
            
            logger.info("Loading spaCy...")
            self.nlp = spacy.load(CONFIG['SPACY_MODEL'])
            
            logger.info("Loading sentence transformer...")
            self.embedder = SentenceTransformer(CONFIG['EMBEDDING_MODEL'])
            
            self._initialized = True
            logger.info("✓ NLP models loaded")


# =============================================================================
# Term Extraction
# =============================================================================

def extract_terms(text: str, nlp) -> ExtractedTerms:
    """Extract nouns, entities, and other terms from text."""
    doc = nlp(text)
    terms = ExtractedTerms()
    
    for token in doc:
        if token.is_alpha and not token.is_stop:
            terms.freq[token.text.lower()] = terms.freq.get(token.text.lower(), 0) + 1
    
    for token in doc:
        if token.pos_ == 'NOUN' and not token.is_stop:
            terms.nouns.add(token.text.lower())
        elif token.pos_ == 'PROPN':
            terms.proper.add(token.text.lower())
    
    for ent in doc.ents:
        terms.entities.add(ent.text.lower())
        if ent.label_ in ('DATE', 'TIME'):
            terms.temporal.add(ent.text.lower())
        elif ent.label_ in ('MONEY', 'CARDINAL', 'PERCENT', 'QUANTITY'):
            terms.numeric.add(ent.text.lower())
    
    return terms


# =============================================================================
# Embedding Functions
# =============================================================================

def get_embedding(text: str, embedder, chunk_size: int = 512) -> np.ndarray:
    """Get embedding using 512-word chunking + averaging."""
    words = text.split()
    
    if len(words) <= chunk_size:
        return embedder.encode(text, convert_to_numpy=True)
    
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    return np.mean(embeddings, axis=0)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# HYBRID Noun Rank Scoring
# =============================================================================

def compute_hallucination_vs_document(doc_terms: ExtractedTerms, llm_terms: ExtractedTerms) -> Tuple[float, Dict, Dict]:
    """
    Compute hallucination score (H) - terms in LLM NOT in DOCUMENT.
    
    This measures: Did LLM invent facts not in the source document?
    
    Returns:
        H: hallucination score
        components: component scores
        hallucinated_words: dict of hallucinated words by category
    """
    components = {}
    hallucinated_words = {}
    
    def hallucination_ratio(llm_set: Set, doc_set: Set, category: str) -> float:
        if not llm_set:
            return 0.0
        hallucinated = llm_set - doc_set
        hallucinated_words[category] = hallucinated
        return len(hallucinated) / len(llm_set)
    
    components['entity'] = hallucination_ratio(llm_terms.entities, doc_terms.entities, 'entity')
    components['proper'] = hallucination_ratio(llm_terms.proper, doc_terms.proper, 'proper')
    components['nouns'] = hallucination_ratio(llm_terms.nouns, doc_terms.nouns, 'nouns')
    components['temporal'] = hallucination_ratio(llm_terms.temporal, doc_terms.temporal, 'temporal')
    components['numeric'] = hallucination_ratio(llm_terms.numeric, doc_terms.numeric, 'numeric')
    
    w = WEIGHTS['H']
    H = (components['entity'] * w['entity'] +
         components['proper'] * w['proper'] +
         components['nouns'] * w['nouns'] +
         components['temporal'] * w['temporal'] +
         components['numeric'] * w['numeric'])
    
    return H, components, hallucinated_words


def compute_coverage_vs_expert(expert_terms: ExtractedTerms, llm_terms: ExtractedTerms) -> Tuple[float, Dict]:
    """
    Compute coverage score (C) - how well LLM covers EXPERT terms.
    
    This measures: Did LLM capture the same key information as expert?
    """
    components = {}
    
    def coverage_ratio(llm_set: Set, expert_set: Set) -> float:
        if not expert_set:
            return 1.0
        covered = llm_set & expert_set
        return len(covered) / len(expert_set)
    
    key_terms = expert_terms.get_key_terms(min_freq=1)
    components['key'] = coverage_ratio(llm_terms.all_terms, key_terms) if key_terms else 1.0
    components['entity'] = coverage_ratio(llm_terms.entities, expert_terms.entities)
    components['temporal'] = coverage_ratio(llm_terms.temporal, expert_terms.temporal)
    components['numeric'] = coverage_ratio(llm_terms.numeric, expert_terms.numeric)
    
    w = WEIGHTS['C']
    C = (components['key'] * w['key'] +
         components['entity'] * w['entity'] +
         components['temporal'] * w['temporal'] +
         components['numeric'] * w['numeric'])
    
    return C, components


def compute_frequency_vs_expert(expert_terms: ExtractedTerms, llm_terms: ExtractedTerms) -> float:
    """Compute frequency alignment score (F) between LLM and EXPERT."""
    expert_freq = expert_terms.freq
    llm_freq = llm_terms.freq
    
    if not expert_freq or not llm_freq:
        return 0.5
    
    expert_total = sum(expert_freq.values())
    llm_total = sum(llm_freq.values())
    
    common_terms = set(expert_freq.keys()) & set(llm_freq.keys())
    if not common_terms:
        return 0.0
    
    alignment = 0.0
    for term in common_terms:
        expert_norm = expert_freq[term] / expert_total
        llm_norm = llm_freq[term] / llm_total
        alignment += min(expert_norm, llm_norm)
    
    return min(alignment * 2, 1.0)


def compute_entity_vs_expert(expert_terms: ExtractedTerms, llm_terms: ExtractedTerms) -> float:
    """Compute entity preservation score (E) - LLM preserves EXPERT entities."""
    if not expert_terms.entities:
        return 1.0
    
    preserved = llm_terms.entities & expert_terms.entities
    return len(preserved) / len(expert_terms.entities)


def compute_semantic_vs_expert(expert_summary: str, llm_summary: str, embedder) -> float:
    """Compute semantic similarity score (S) between LLM and EXPERT."""
    expert_emb = get_embedding(expert_summary, embedder)
    llm_emb = get_embedding(llm_summary, embedder)
    return cosine_similarity(expert_emb, llm_emb)


# =============================================================================
# Readability Scoring (R)
# =============================================================================

TRANSITION_WORDS = {
    'however', 'although', 'nevertheless', 'despite', 'whereas',
    'conversely', 'yet', 'still', 'nonetheless',
    'additionally', 'furthermore', 'moreover', 'also', 'besides',
    'therefore', 'consequently', 'thus', 'hence', 'accordingly',
    'first', 'second', 'third', 'then', 'next', 'finally', 'subsequently',
    'meanwhile', 'afterward', 'previously',
    'specifically', 'notably', 'particularly',
    'overall', 'ultimately',
}


def compute_flesch_score(summary: str) -> float:
    """Compute normalized Flesch Reading Ease score (0-1)."""
    words = summary.split()
    sentences = re.split(r'[.!?]+', summary)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not words or not sentences:
        return 0.5
    
    def count_syllables(word):
        word = word.lower()
        vowels = 'aeiouy'
        count = 0
        prev_vowel = False
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        return max(count, 1)
    
    total_syllables = sum(count_syllables(w) for w in words)
    
    asl = len(words) / len(sentences)
    asw = total_syllables / len(words)
    
    flesch = 206.835 - (1.015 * asl) - (84.6 * asw)
    
    if flesch < 40:
        return max(flesch / 40, 0.0)
    elif flesch > 80:
        return max(1 - (flesch - 80) / 40, 0.5)
    else:
        return 1.0


def compute_sentence_variance(summary: str) -> float:
    """Compute sentence length variance score (0-1)."""
    sentences = re.split(r'[.!?]+', summary)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if len(sentences) < 2:
        return 0.5
    
    lengths = [len(s.split()) for s in sentences]
    mean_len = np.mean(lengths)
    std_len = np.std(lengths)
    
    if mean_len == 0:
        return 0.0
    
    cv = std_len / mean_len
    return min(cv / 0.6, 1.0)


def compute_transition_score(summary: str) -> float:
    """Compute transition word density score (0-1)."""
    summary_lower = summary.lower()
    words = summary.split()
    
    if not words:
        return 0.0
    
    transition_count = sum(1 for t in TRANSITION_WORDS if t in summary_lower)
    density = (transition_count / len(words)) * 100
    
    return min(density / 5, 1.0)


def compute_avg_sentence_length_score(summary: str) -> float:
    """Compute average sentence length score (0-1, optimal: 15-20 words)."""
    sentences = re.split(r'[.!?]+', summary)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return 0.5
    
    lengths = [len(s.split()) for s in sentences]
    avg_len = np.mean(lengths)
    
    opt_min = CONFIG['OPTIMAL_SENTENCE_LENGTH_MIN']
    opt_max = CONFIG['OPTIMAL_SENTENCE_LENGTH_MAX']
    
    if opt_min <= avg_len <= opt_max:
        return 1.0
    elif avg_len < opt_min:
        return max(avg_len / opt_min, 0.3)
    else:
        return max(1 - (avg_len - opt_max) / 20, 0.3)


def compute_readability_score(summary: str) -> Tuple[float, Dict]:
    """Compute combined readability score (R)."""
    components = {}
    
    components['flesch'] = compute_flesch_score(summary)
    components['variance'] = compute_sentence_variance(summary)
    components['transitions'] = compute_transition_score(summary)
    components['avg_length'] = compute_avg_sentence_length_score(summary)
    
    w = WEIGHTS['R']
    R = (components['flesch'] * w['flesch'] +
         components['variance'] * w['variance'] +
         components['transitions'] * w['transitions'] +
         components['avg_length'] * w['avg_length'])
    
    return R, components


def compute_final_score(H: float, C: float, F: float, E: float, S: float, R: float) -> float:
    """Compute final score with adaptive weights."""
    if H > CONFIG['HALLUCINATION_THRESHOLD']:
        w = WEIGHTS['FINAL']
        return (1 - H) * w['H'] + C * w['C'] + F * w['F'] + E * w['E'] + S * w['S'] + R * w['R']
    else:
        w = WEIGHTS['FINAL_NO_H']
        return C * w['C'] + F * w['F'] + E * w['E'] + S * w['S'] + R * w['R']


def compute_hybrid_score(doc_text: str, expert_summary: str, llm_summary: str, 
                         nlp, embedder) -> Tuple[ComponentScores, Dict]:
    """
    HYBRID Noun Rank scoring:
    - H (Hallucination): LLM vs DOCUMENT (source truth)
    - C (Coverage): LLM vs EXPERT (gold standard)
    - F (Frequency): LLM vs EXPERT
    - E (Entity): LLM vs EXPERT
    - S (Semantic): LLM vs EXPERT
    - R (Readability): LLM only
    
    Returns:
        scores: ComponentScores object
        hallucinated_words: dict of hallucinated words by category
    """
    scores = ComponentScores()
    
    # Extract terms from all three texts
    doc_terms = extract_terms(doc_text, nlp)
    expert_terms = extract_terms(expert_summary, nlp)
    llm_terms = extract_terms(llm_summary, nlp)
    
    # H: Hallucination vs DOCUMENT (did LLM invent facts?)
    scores.H, h_comp, hallucinated_words = compute_hallucination_vs_document(doc_terms, llm_terms)
    scores.H_entity = h_comp['entity']
    scores.H_proper = h_comp['proper']
    scores.H_nouns = h_comp['nouns']
    scores.H_temporal = h_comp['temporal']
    scores.H_numeric = h_comp['numeric']
    
    # C: Coverage vs EXPERT (did LLM capture same info as expert?)
    scores.C, c_comp = compute_coverage_vs_expert(expert_terms, llm_terms)
    scores.C_key = c_comp['key']
    scores.C_entity = c_comp['entity']
    scores.C_temporal = c_comp['temporal']
    scores.C_numeric = c_comp['numeric']
    
    # F: Frequency alignment vs EXPERT
    scores.F = compute_frequency_vs_expert(expert_terms, llm_terms)
    
    # E: Entity preservation vs EXPERT
    scores.E = compute_entity_vs_expert(expert_terms, llm_terms)
    
    # S: Semantic similarity vs EXPERT
    scores.S = compute_semantic_vs_expert(expert_summary, llm_summary, embedder)
    
    # R: Readability of LLM summary
    scores.R, r_comp = compute_readability_score(llm_summary)
    scores.R_flesch = r_comp['flesch']
    scores.R_variance = r_comp['variance']
    scores.R_transitions = r_comp['transitions']
    scores.R_avg_length = r_comp['avg_length']
    
    # Final score
    scores.final_score = compute_final_score(scores.H, scores.C, scores.F, scores.E, scores.S, scores.R)
    
    return scores, hallucinated_words


def compute_baseline_score(doc_text: str, expert_summary: str, nlp, embedder) -> ComponentScores:
    """
    Compute BASELINE score for expert summary.
    
    - H: Expert vs Document (should be ~0, expert uses doc facts)
    - C: Expert vs Expert = 1.0 (perfect coverage)
    - F: Expert vs Expert = 1.0 (perfect frequency)
    - E: Expert vs Expert = 1.0 (perfect entity)
    - S: Expert vs Expert = 1.0 (perfect semantic)
    - R: Expert readability
    """
    scores = ComponentScores()
    
    doc_terms = extract_terms(doc_text, nlp)
    expert_terms = extract_terms(expert_summary, nlp)
    
    # H: Expert vs Document (how much did expert "hallucinate"?)
    scores.H, h_comp, _ = compute_hallucination_vs_document(doc_terms, expert_terms)
    scores.H_entity = h_comp['entity']
    scores.H_proper = h_comp['proper']
    scores.H_nouns = h_comp['nouns']
    scores.H_temporal = h_comp['temporal']
    scores.H_numeric = h_comp['numeric']
    
    # C, F, E, S: Expert vs Expert = perfect scores
    scores.C = 1.0
    scores.C_key = 1.0
    scores.C_entity = 1.0
    scores.C_temporal = 1.0
    scores.C_numeric = 1.0
    
    scores.F = 1.0
    scores.E = 1.0
    scores.S = 1.0
    
    # R: Expert readability
    scores.R, r_comp = compute_readability_score(expert_summary)
    scores.R_flesch = r_comp['flesch']
    scores.R_variance = r_comp['variance']
    scores.R_transitions = r_comp['transitions']
    scores.R_avg_length = r_comp['avg_length']
    
    # Final score
    scores.final_score = compute_final_score(scores.H, scores.C, scores.F, scores.E, scores.S, scores.R)
    
    return scores


# =============================================================================
# Database Manager (Thread-Safe)
# =============================================================================

class DatabaseManager:
    """Thread-safe database manager for hybrid evaluation."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_database()
        self._ensure_model_entry()
    
    def _init_database(self):
        """Create tables."""
        cursor = self.conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_length INTEGER,
                expert_summary_length INTEGER,
                source TEXT DEFAULT 'govreport',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_models (
                model_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                model_version TEXT,
                provider TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_name, model_version)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hybrid_evaluations (
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                model_id INTEGER,
                
                -- Main Score
                score REAL,
                
                -- Component Scores
                H REAL,  -- Hallucination (vs Document)
                C REAL,  -- Coverage (vs Expert)
                F REAL,  -- Frequency (vs Expert)
                E REAL,  -- Entity (vs Expert)
                S REAL,  -- Semantic (vs Expert)
                R REAL,  -- Readability (LLM)
                
                -- Hallucination Sub-components (vs Document)
                H_entity REAL,
                H_proper REAL,
                H_nouns REAL,
                H_temporal REAL,
                H_numeric REAL,
                
                -- Coverage Sub-components (vs Expert)
                C_key REAL,
                C_entity REAL,
                C_temporal REAL,
                C_numeric REAL,
                
                -- Readability Sub-components
                R_flesch REAL,
                R_variance REAL,
                R_transitions REAL,
                R_avg_length REAL,
                
                -- Expert Readability (for comparison)
                expert_R REAL,
                
                -- Status
                status TEXT,
                
                -- Metadata
                algorithm_version TEXT,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_time_ms REAL,
                
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
                FOREIGN KEY (model_id) REFERENCES llm_models(model_id)
            )
        ''')
        
        self.conn.commit()
    
    def _ensure_model_entry(self):
        """Ensure LLM model is registered."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO llm_models (model_name, model_version, provider)
                VALUES (?, ?, ?)
            ''', (CONFIG['LLM_MODEL'], 'v1', 'vast.ai'))
            self.conn.commit()
            
            cursor.execute('SELECT model_id FROM llm_models WHERE model_name = ?',
                          (CONFIG['LLM_MODEL'],))
            self.model_id = cursor.fetchone()['model_id']
    
    def save_document(self, doc_id: str, doc_length: int, expert_summary_length: int):
        """Save document metadata."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO documents (doc_id, doc_length, expert_summary_length)
                VALUES (?, ?, ?)
            ''', (doc_id, doc_length, expert_summary_length))
            self.conn.commit()
    
    def save_hybrid_evaluation(self, doc_id: str, scores: ComponentScores,
                                expert_R: float, status: str, processing_time_ms: float):
        """Save hybrid evaluation."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO hybrid_evaluations (
                    doc_id, model_id, score,
                    H, C, F, E, S, R,
                    H_entity, H_proper, H_nouns, H_temporal, H_numeric,
                    C_key, C_entity, C_temporal, C_numeric,
                    R_flesch, R_variance, R_transitions, R_avg_length,
                    expert_R, status,
                    algorithm_version, processing_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_id, self.model_id, scores.final_score,
                scores.H, scores.C, scores.F, scores.E, scores.S, scores.R,
                scores.H_entity, scores.H_proper, scores.H_nouns,
                scores.H_temporal, scores.H_numeric,
                scores.C_key, scores.C_entity, scores.C_temporal, scores.C_numeric,
                scores.R_flesch, scores.R_variance, scores.R_transitions, scores.R_avg_length,
                expert_R, status,
                CONFIG['ALGORITHM_VERSION'], processing_time_ms
            ))
            self.conn.commit()


# =============================================================================
# Hybrid Evaluator
# =============================================================================

class HybridEvaluator:
    """
    Hybrid evaluator:
    - H (Hallucination): Compare LLM to DOCUMENT
    - C/F/E/S: Compare LLM to EXPERT
    - R: LLM readability
    """
    
    def __init__(self, db_path: str, llm_host: str, llm_api_key: str = None):
        self.db = DatabaseManager(db_path)
        self.llm = LLMClient(llm_host, CONFIG['LLM_MODEL'], llm_api_key)
        self.models = NLPModels()
        self.models.load()
    
    def load_govreport(self, split: str = "test", limit: int = None) -> List[Dict]:
        """Load GovReport dataset."""
        from datasets import load_dataset
        
        logger.info(f"Loading GovReport (split={split})...")
        dataset = load_dataset("ccdv/govreport-summarization", split=split)
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        records = []
        for idx, item in enumerate(dataset):
            records.append({
                'doc_id': f"doc_{idx:05d}",
                'document': item['report'],
                'expert_summary': item['summary']
            })
        
        logger.info(f"Loaded {len(records)} documents")
        return records
    
    def evaluate_document(self, record: Dict) -> EvalResult:
        """Evaluate single document using hybrid approach."""
        doc_id = record['doc_id']
        doc_text = record['document']
        expert_summary = record['expert_summary']
        
        start_time = time.time()
        
        self.db.save_document(
            doc_id, 
            len(doc_text.split()), 
            len(expert_summary.split())
        )
        
        # Compute baseline score (expert summary)
        try:
            baseline_scores = compute_baseline_score(
                doc_text, expert_summary,
                self.models.nlp, self.models.embedder
            )
        except Exception as e:
            logger.error(f"{doc_id}: Baseline error: {e}")
            baseline_scores = ComponentScores()
            baseline_scores.final_score = 0.85  # Default fallback
        
        # Generate LLM summary
        try:
            llm_summary = self.llm.summarize(doc_text)
        except Exception as e:
            logger.error(f"{doc_id}: LLM error: {e}")
            return EvalResult(
                doc_id=doc_id,
                baseline_score=baseline_scores.final_score,
                llm_score=0.0,
                accuracy_pct=0.0,
                H=1.0,
                C=0.0,
                S=0.0,
                expert_R=baseline_scores.R,
                llm_R=0.0,
                status="ERROR",
                llm_summary="",
                expert_summary=expert_summary
            )
        
        # Compute hybrid score for LLM
        try:
            llm_scores, hallucinated_words = compute_hybrid_score(
                doc_text, expert_summary, llm_summary,
                self.models.nlp, self.models.embedder
            )
            
            # Print hallucinated words
            if any(hallucinated_words.values()):
                print(f"\n{'='*60}")
                print(f"📛 HALLUCINATED WORDS for {doc_id}:")
                print(f"{'='*60}")
                for category, words in hallucinated_words.items():
                    if words:
                        print(f"  [{category.upper()}] ({len(words)} words):")
                        # Sort and limit to top 20 for readability
                        sorted_words = sorted(words)[:20]
                        print(f"    {', '.join(sorted_words)}")
                        if len(words) > 20:
                            print(f"    ... and {len(words) - 20} more")
                print(f"{'='*60}\n")
            
        except Exception as e:
            logger.error(f"{doc_id}: Scoring error: {e}")
            return EvalResult(
                doc_id=doc_id,
                baseline_score=baseline_scores.final_score,
                llm_score=0.0,
                accuracy_pct=0.0,
                H=1.0,
                C=0.0,
                S=0.0,
                expert_R=baseline_scores.R,
                llm_R=0.0,
                status="ERROR",
                llm_summary=llm_summary,
                expert_summary=expert_summary
            )
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Calculate accuracy
        accuracy_pct = (llm_scores.final_score / baseline_scores.final_score * 100) if baseline_scores.final_score > 0 else 0
        
        # Determine status based on accuracy
        if llm_scores.H >= 0.35:
            status = "REJECT_HALLUCINATION"
        elif accuracy_pct >= 95:
            status = "EXCELLENT"
        elif accuracy_pct >= 85:
            status = "GOOD"
        elif accuracy_pct >= 70:
            status = "ACCEPTABLE"
        else:
            status = "POOR"
        
        self.db.save_hybrid_evaluation(
            doc_id, llm_scores, baseline_scores.R, status, elapsed_ms
        )
        
        return EvalResult(
            doc_id=doc_id,
            baseline_score=baseline_scores.final_score,
            llm_score=llm_scores.final_score,
            accuracy_pct=accuracy_pct,
            H=llm_scores.H,
            C=llm_scores.C,
            S=llm_scores.S,
            expert_R=baseline_scores.R,
            llm_R=llm_scores.R,
            status=status,
            llm_summary=llm_summary,
            expert_summary=expert_summary,
            hallucinated_words=hallucinated_words
        )
    
    def evaluate(self, split: str = "test", limit: int = None) -> Dict:
        """Run hybrid evaluation on all documents."""
        records = self.load_govreport(split=split, limit=limit)
        
        print("\n" + "=" * 80)
        print("HYBRID NOUN RANK EVALUATION")
        print("=" * 80)
        print(f"Model:     {CONFIG['LLM_MODEL']}")
        print(f"Documents: {len(records)}")
        print(f"Algorithm: {CONFIG['ALGORITHM_VERSION']}")
        print(f"Mode:      H vs Document, C/E/S vs Expert")
        print("=" * 80 + "\n")
        
        results = []
        start_time = time.time()
        
        BOLD = '\033[1m'
        RED = '\033[91m'
        RESET = '\033[0m'
        
        for i, record in enumerate(records):
            result = self.evaluate_document(record)
            results.append(result)
            
            status_icon = {
                'EXCELLENT': '★',
                'GOOD': '✓',
                'ACCEPTABLE': '○',
                'POOR': '↓',
                'REJECT_HALLUCINATION': '✗H',
                'ERROR': '✗E'
            }.get(result.status, '?')
            
            if result.status == 'REJECT_HALLUCINATION':
                logger.info(
                    f"{BOLD}{RED}[{i+1}/{len(records)}] {result.doc_id}: "
                    f"Baseline={result.baseline_score:.4f} LLM={result.llm_score:.4f} "
                    f"Acc={result.accuracy_pct:.1f}% "
                    f"H={result.H:.4f} C={result.C:.4f} "
                    f"[{status_icon}] ⚠️ HALLUCINATION{RESET}"
                )
            else:
                logger.info(
                    f"[{i+1}/{len(records)}] {result.doc_id}: "
                    f"Baseline={result.baseline_score:.4f} LLM={result.llm_score:.4f} "
                    f"Acc={result.accuracy_pct:.1f}% "
                    f"H={result.H:.4f} C={result.C:.4f} "
                    f"[{status_icon}]"
                )
            
            clear_context()
        
        valid_results = [r for r in results if r.status != 'ERROR']
        
        stats = {
            'total_docs': len(records),
            'evaluated': len(valid_results),
            'mean_baseline': np.mean([r.baseline_score for r in valid_results]) if valid_results else 0,
            'mean_llm': np.mean([r.llm_score for r in valid_results]) if valid_results else 0,
            'mean_accuracy': np.mean([r.accuracy_pct for r in valid_results]) if valid_results else 0,
            'std_accuracy': np.std([r.accuracy_pct for r in valid_results]) if valid_results else 0,
            'min_accuracy': np.min([r.accuracy_pct for r in valid_results]) if valid_results else 0,
            'max_accuracy': np.max([r.accuracy_pct for r in valid_results]) if valid_results else 0,
            'mean_H': np.mean([r.H for r in valid_results]) if valid_results else 0,
            'mean_C': np.mean([r.C for r in valid_results]) if valid_results else 0,
            'mean_S': np.mean([r.S for r in valid_results]) if valid_results else 0,
            'mean_expert_R': np.mean([r.expert_R for r in valid_results]) if valid_results else 0,
            'mean_llm_R': np.mean([r.llm_R for r in valid_results]) if valid_results else 0,
            'excellent': sum(1 for r in results if r.status == 'EXCELLENT'),
            'good': sum(1 for r in results if r.status == 'GOOD'),
            'acceptable': sum(1 for r in results if r.status == 'ACCEPTABLE'),
            'poor': sum(1 for r in results if r.status == 'POOR'),
            'rejected_hallucination': sum(1 for r in results if r.status == 'REJECT_HALLUCINATION'),
            'errors': sum(1 for r in results if r.status == 'ERROR'),
            'total_time': time.time() - start_time
        }
        
        self._print_report(stats)
        return stats
    
    def _print_report(self, stats: Dict):
        """Print final evaluation report."""
        print("\n" + "=" * 80)
        print("EVALUATION COMPLETE")
        print("=" * 80)
        
        print(f"\n📊 SCORES:")
        print(f"   Mean Baseline:  {stats['mean_baseline']:.4f}")
        print(f"   Mean LLM:       {stats['mean_llm']:.4f}")
        
        print(f"\n📈 ACCURACY (LLM vs Baseline):")
        print(f"   Mean:    {stats['mean_accuracy']:.1f}%")
        print(f"   Std:     {stats['std_accuracy']:.1f}%")
        print(f"   Range:   {stats['min_accuracy']:.1f}% - {stats['max_accuracy']:.1f}%")
        
        print(f"\n🔍 COMPONENT AVERAGES:")
        print(f"   H (Hallucination vs Doc):  {stats['mean_H']:.4f} (lower is better)")
        print(f"   C (Coverage vs Expert):    {stats['mean_C']:.4f}")
        print(f"   S (Semantic vs Expert):    {stats['mean_S']:.4f}")
        
        print(f"\n📖 READABILITY COMPARISON:")
        print(f"   Expert R:  {stats['mean_expert_R']:.4f}")
        print(f"   LLM R:     {stats['mean_llm_R']:.4f}")
        r_diff = stats['mean_expert_R'] - stats['mean_llm_R']
        if r_diff > 0.05:
            print(f"   → Expert more readable by {r_diff:.2f}")
        elif r_diff < -0.05:
            print(f"   → LLM more readable by {-r_diff:.2f}")
        else:
            print(f"   → Readability comparable")
        
        print(f"\n🏷️  CLASSIFICATION:")
        print(f"   ★ Excellent (≥95%):       {stats['excellent']}")
        print(f"   ✓ Good (85-95%):          {stats['good']}")
        print(f"   ○ Acceptable (70-85%):    {stats['acceptable']}")
        print(f"   ↓ Poor (<70%):            {stats['poor']}")
        print(f"   ✗H Hallucination:         {stats['rejected_hallucination']}")
        print(f"   ✗E Errors:                {stats['errors']}")
        
        print(f"\n⏱️  TIMING:")
        print(f"   Total Time: {stats['total_time']:.1f}s")
        print(f"   Per Doc:    {stats['total_time']/max(stats['evaluated'],1):.1f}s")
        
        print("\n" + "=" * 80)
        
        if stats['mean_accuracy'] >= 85:
            print("✅ PASS: Mean accuracy ≥ 85%")
        else:
            print("❌ FAIL: Mean accuracy < 85%")
        
        print("=" * 80 + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Noun Rank Hybrid Evaluator v3.0")
    parser.add_argument("--host", type=str, default="http://localhost:8000",
                       help="vLLM host URL")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for vLLM")
    parser.add_argument("--db", type=str, default="noun_rank_hybrid_v3.db",
                       help="Database path")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit documents to evaluate")
    parser.add_argument("--model", type=str, default=None,
                       help="Override LLM model name")
    parser.add_argument("--context", type=int, default=4096,
                       help="Model context length in tokens (default: 4096)")
    
    args = parser.parse_args()
    
    if args.model:
        CONFIG['LLM_MODEL'] = args.model
    
    CONFIG['MAX_INPUT_WORDS'] = int((args.context - 800) * 0.75)
    CONFIG['CHUNK_SIZE'] = int(CONFIG['MAX_INPUT_WORDS'] * 0.8)
    
    logger.info(f"Context: {args.context} tokens → Max input: {CONFIG['MAX_INPUT_WORDS']} words")
    
    evaluator = HybridEvaluator(
        db_path=args.db,
        llm_host=args.host,
        llm_api_key=args.api_key
    )
    
    stats = evaluator.evaluate(split=args.split, limit=args.limit)
    return stats


if __name__ == "__main__":
    main()
