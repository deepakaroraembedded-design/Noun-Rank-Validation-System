"""
Noun Rank Combined Evaluator v13.0
Parallel baseline calibration + Mistral LLM evaluation

Two parallel threads:
  T1: Baseline calculation using human summaries (Noun Rank)
  T2: LLM summarization using Mistral 7B + Noun Rank scoring

Saves baseline score, LLM score, and delta to database.

Usage:
    python noun_rank_combined_evaluator.py --limit 100
    python noun_rank_combined_evaluator.py --limit 100 --host http://localhost:8000

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
    'MAX_INPUT_WORDS': 2500,  # ~3200 tokens, fits 4K context (adjust per model)
    'CHUNK_SIZE': 2000,       # Words per chunk for long docs
    
    # === Database ===
    'DB_PATH': 'noun_rank_combined.db',
    
    # === NLP Models ===
    'SPACY_MODEL': 'en_core_web_sm',
    'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
    
    # === Algorithm ===
    'HALLUCINATION_THRESHOLD': 0.05,  # Below this, use adaptive weights
    
    # === Version ===
    'ALGORITHM_VERSION': 'v13.0-combined',
}

WEIGHTS = {
    'H': {'entity': 0.30, 'proper': 0.25, 'nouns': 0.20, 'numeric': 0.15, 'temporal': 0.10},
    'C': {'key': 0.35, 'entity': 0.30, 'numeric': 0.20, 'temporal': 0.15},
    'FINAL': {'H': 0.25, 'C': 0.25, 'F': 0.20, 'E': 0.20, 'S': 0.10},
    'FINAL_NO_H': {'C': 0.35, 'E': 0.30, 'F': 0.25, 'S': 0.10},  # Adaptive weights when H≈0
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
        pass  # torch not available, skip GPU clearing


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
    H: float = 0.0
    C: float = 0.0
    F: float = 0.0
    E: float = 0.0
    S: float = 0.0
    H_entity: float = 0.0
    H_proper: float = 0.0
    H_nouns: float = 0.0
    H_temporal: float = 0.0
    H_numeric: float = 0.0
    C_key: float = 0.0
    C_entity: float = 0.0
    C_temporal: float = 0.0
    C_numeric: float = 0.0
    final_score: float = 0.0


@dataclass
class EvalResult:
    doc_id: str
    baseline_score: float
    llm_score: float
    delta: float
    accuracy_pct: float
    baseline_H: float
    llm_H: float
    status: str
    llm_summary: str = ""
    expert_summary: str = ""


# =============================================================================
# LLM Client
# =============================================================================

class LLMClient:
    """Client for Mistral 7B via vLLM OpenAI-compatible API."""
    
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
        
        # If document fits context, summarize directly
        if len(words) <= max_input:
            return self._call_llm(document, max_tokens)
        
        # Chunk long documents
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        logger.info(f"Document split into {len(chunks)} chunks ({len(words)} words)")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"  Summarizing chunk {i+1}/{len(chunks)}")
            summary = self._call_llm(chunk, max_tokens=400)
            chunk_summaries.append(summary)
        
        # Combine chunk summaries
        combined = "\n\n".join(chunk_summaries)
        
        if len(combined.split()) <= 800:
            return combined
        
        # Summarize combined summaries
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
    
    # Count all tokens for frequency
    for token in doc:
        if token.is_alpha and not token.is_stop:
            terms.freq[token.text.lower()] = terms.freq.get(token.text.lower(), 0) + 1
    
    # Extract nouns
    for token in doc:
        if token.pos_ == 'NOUN' and not token.is_stop:
            terms.nouns.add(token.text.lower())
        elif token.pos_ == 'PROPN':
            terms.proper.add(token.text.lower())
    
    # Extract entities
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
    """Get embedding using 512-word chunking + averaging (matches calibration)."""
    words = text.split()
    
    if len(words) <= chunk_size:
        return embedder.encode(text, convert_to_numpy=True)
    
    # Chunk and average embeddings
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
# Noun Rank Scoring
# =============================================================================

def compute_hallucination_score(doc_terms: ExtractedTerms, sum_terms: ExtractedTerms) -> Tuple[float, Dict]:
    """Compute hallucination score (H) - terms in summary not in document."""
    components = {}
    
    def hallucination_ratio(sum_set: Set, doc_set: Set) -> float:
        if not sum_set:
            return 0.0
        hallucinated = sum_set - doc_set
        return len(hallucinated) / len(sum_set)
    
    components['entity'] = hallucination_ratio(sum_terms.entities, doc_terms.entities)
    components['proper'] = hallucination_ratio(sum_terms.proper, doc_terms.proper)
    components['nouns'] = hallucination_ratio(sum_terms.nouns, doc_terms.nouns)
    components['temporal'] = hallucination_ratio(sum_terms.temporal, doc_terms.temporal)
    components['numeric'] = hallucination_ratio(sum_terms.numeric, doc_terms.numeric)
    
    w = WEIGHTS['H']
    H = (components['entity'] * w['entity'] +
         components['proper'] * w['proper'] +
         components['nouns'] * w['nouns'] +
         components['temporal'] * w['temporal'] +
         components['numeric'] * w['numeric'])
    
    return H, components


def compute_coverage_score(doc_terms: ExtractedTerms, sum_terms: ExtractedTerms) -> Tuple[float, Dict]:
    """Compute coverage score (C) - how well summary covers key document terms."""
    components = {}
    
    def coverage_ratio(sum_set: Set, doc_set: Set) -> float:
        if not doc_set:
            return 1.0
        covered = sum_set & doc_set
        return len(covered) / len(doc_set)
    
    key_terms = doc_terms.get_key_terms(min_freq=2)
    components['key'] = coverage_ratio(sum_terms.all_terms, key_terms) if key_terms else 1.0
    components['entity'] = coverage_ratio(sum_terms.entities, doc_terms.entities)
    components['temporal'] = coverage_ratio(sum_terms.temporal, doc_terms.temporal)
    components['numeric'] = coverage_ratio(sum_terms.numeric, doc_terms.numeric)
    
    w = WEIGHTS['C']
    C = (components['key'] * w['key'] +
         components['entity'] * w['entity'] +
         components['temporal'] * w['temporal'] +
         components['numeric'] * w['numeric'])
    
    return C, components


def compute_frequency_alignment(doc_terms: ExtractedTerms, sum_terms: ExtractedTerms) -> float:
    """Compute frequency alignment score (F)."""
    doc_freq = doc_terms.freq
    sum_freq = sum_terms.freq
    
    if not doc_freq or not sum_freq:
        return 0.5
    
    # Normalize frequencies
    doc_total = sum(doc_freq.values())
    sum_total = sum(sum_freq.values())
    
    common_terms = set(doc_freq.keys()) & set(sum_freq.keys())
    if not common_terms:
        return 0.0
    
    alignment = 0.0
    for term in common_terms:
        doc_norm = doc_freq[term] / doc_total
        sum_norm = sum_freq[term] / sum_total
        alignment += min(doc_norm, sum_norm)
    
    return min(alignment * 2, 1.0)


def compute_entity_score(doc_terms: ExtractedTerms, sum_terms: ExtractedTerms, nlp) -> float:
    """Compute weighted entity preservation score (E)."""
    if not doc_terms.entities:
        return 1.0
    
    preserved = sum_terms.entities & doc_terms.entities
    return len(preserved) / len(doc_terms.entities)


def compute_semantic_similarity(doc_text: str, summary: str, embedder) -> float:
    """Compute semantic similarity score (S)."""
    doc_emb = get_embedding(doc_text, embedder)
    sum_emb = get_embedding(summary, embedder)
    return cosine_similarity(doc_emb, sum_emb)


def compute_final_score(H: float, C: float, F: float, E: float, S: float) -> float:
    """Compute final score with adaptive weights based on observed hallucination."""
    if H > CONFIG['HALLUCINATION_THRESHOLD']:
        # Use standard weights (hallucination detected)
        w = WEIGHTS['FINAL']
        return (1 - H) * w['H'] + C * w['C'] + F * w['F'] + E * w['E'] + S * w['S']
    else:
        # Use redistributed weights (H ≈ 0)
        w = WEIGHTS['FINAL_NO_H']
        return C * w['C'] + F * w['F'] + E * w['E'] + S * w['S']


def compute_noun_rank(doc_text: str, summary: str, nlp, embedder) -> ComponentScores:
    """Compute full Noun Rank score for a document-summary pair."""
    scores = ComponentScores()
    
    # Extract terms
    doc_terms = extract_terms(doc_text, nlp)
    sum_terms = extract_terms(summary, nlp)
    
    # Compute components
    scores.H, h_comp = compute_hallucination_score(doc_terms, sum_terms)
    scores.H_entity = h_comp['entity']
    scores.H_proper = h_comp['proper']
    scores.H_nouns = h_comp['nouns']
    scores.H_temporal = h_comp['temporal']
    scores.H_numeric = h_comp['numeric']
    
    scores.C, c_comp = compute_coverage_score(doc_terms, sum_terms)
    scores.C_key = c_comp['key']
    scores.C_entity = c_comp['entity']
    scores.C_temporal = c_comp['temporal']
    scores.C_numeric = c_comp['numeric']
    
    scores.F = compute_frequency_alignment(doc_terms, sum_terms)
    scores.E = compute_entity_score(doc_terms, sum_terms, nlp)
    scores.S = compute_semantic_similarity(doc_text, summary, embedder)
    
    # Final score (adaptive)
    scores.final_score = compute_final_score(scores.H, scores.C, scores.F, scores.E, scores.S)
    
    return scores


# =============================================================================
# Database Manager (Thread-Safe)
# =============================================================================

class DatabaseManager:
    """Thread-safe database manager for combined evaluation."""
    
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
                summary_length INTEGER,
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
            CREATE TABLE IF NOT EXISTS combined_evaluations (
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                model_id INTEGER,
                
                -- Baseline (Expert Summary) Scores
                baseline_score REAL,
                baseline_H REAL,
                baseline_C REAL,
                baseline_F REAL,
                baseline_E REAL,
                baseline_S REAL,
                
                -- LLM Summary Scores
                llm_score REAL,
                llm_H REAL,
                llm_C REAL,
                llm_F REAL,
                llm_E REAL,
                llm_S REAL,
                
                -- Performance Metrics
                delta REAL,
                accuracy_pct REAL,
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
            ''', (CONFIG['LLM_MODEL'], 'v0.3', 'vast.ai'))
            self.conn.commit()
            
            cursor.execute('SELECT model_id FROM llm_models WHERE model_name = ?',
                          (CONFIG['LLM_MODEL'],))
            self.model_id = cursor.fetchone()['model_id']
    
    def save_document(self, doc_id: str, doc_length: int, summary_length: int):
        """Save document metadata."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO documents (doc_id, doc_length, summary_length)
                VALUES (?, ?, ?)
            ''', (doc_id, doc_length, summary_length))
            self.conn.commit()
    
    def save_combined_evaluation(self, doc_id: str, 
                                  baseline_scores: ComponentScores,
                                  llm_scores: ComponentScores,
                                  delta: float, accuracy_pct: float,
                                  status: str, processing_time_ms: float):
        """Save combined baseline + LLM evaluation."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO combined_evaluations (
                    doc_id, model_id,
                    baseline_score, baseline_H, baseline_C, baseline_F, baseline_E, baseline_S,
                    llm_score, llm_H, llm_C, llm_F, llm_E, llm_S,
                    delta, accuracy_pct, status,
                    algorithm_version, processing_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_id, self.model_id,
                baseline_scores.final_score, baseline_scores.H, baseline_scores.C,
                baseline_scores.F, baseline_scores.E, baseline_scores.S,
                llm_scores.final_score, llm_scores.H, llm_scores.C,
                llm_scores.F, llm_scores.E, llm_scores.S,
                delta, accuracy_pct, status,
                CONFIG['ALGORITHM_VERSION'], processing_time_ms
            ))
            self.conn.commit()
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics from database."""
        with self._lock:
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    AVG(baseline_score) as avg_baseline,
                    AVG(llm_score) as avg_llm,
                    AVG(delta) as avg_delta,
                    AVG(accuracy_pct) as avg_accuracy,
                    MIN(accuracy_pct) as min_accuracy,
                    MAX(accuracy_pct) as max_accuracy
                FROM combined_evaluations
            ''')
            row = cursor.fetchone()
            return dict(row) if row else {}


# =============================================================================
# Combined Evaluator
# =============================================================================

class CombinedEvaluator:
    """
    Parallel evaluator with two threads:
      T1: Baseline calculation (Noun Rank on expert summaries)
      T2: LLM summarization + Noun Rank scoring
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
        """
        Evaluate single document with parallel baseline + LLM scoring.
        
        T1: Compute baseline (expert summary)
        T2: Generate LLM summary + score
        """
        doc_id = record['doc_id']
        doc_text = record['document']
        expert_summary = record['expert_summary']
        
        start_time = time.time()
        
        # Save document metadata
        self.db.save_document(
            doc_id, 
            len(doc_text.split()), 
            len(expert_summary.split())
        )
        
        # Results containers
        results = {'baseline': None, 'llm': None, 'llm_summary': None, 'error': None}
        
        def compute_baseline():
            """T1: Compute baseline score on expert summary."""
            try:
                results['baseline'] = compute_noun_rank(
                    doc_text, expert_summary,
                    self.models.nlp, self.models.embedder
                )
            except Exception as e:
                results['error'] = f"Baseline error: {e}"
        
        def compute_llm():
            """T2: Generate LLM summary and score it."""
            try:
                llm_summary = self.llm.summarize(doc_text)
                results['llm_summary'] = llm_summary
                results['llm'] = compute_noun_rank(
                    doc_text, llm_summary,
                    self.models.nlp, self.models.embedder
                )
            except Exception as e:
                results['error'] = f"LLM error: {e}"
        
        # Run T1 and T2 in parallel
        t1 = threading.Thread(target=compute_baseline)
        t2 = threading.Thread(target=compute_llm)
        
        t1.start()
        t2.start()
        
        t1.join()
        t2.join()
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Handle errors
        if results['error'] or results['baseline'] is None or results['llm'] is None:
            logger.error(f"{doc_id}: {results['error']}")
            return EvalResult(
                doc_id=doc_id,
                baseline_score=0.0,
                llm_score=0.0,
                delta=0.0,
                accuracy_pct=0.0,
                baseline_H=0.0,
                llm_H=0.0,
                status="ERROR",
                llm_summary="",
                expert_summary=expert_summary
            )
        
        baseline_scores = results['baseline']
        llm_scores = results['llm']
        
        # Calculate delta and accuracy
        delta = llm_scores.final_score - baseline_scores.final_score
        accuracy_pct = (llm_scores.final_score / baseline_scores.final_score * 100) if baseline_scores.final_score > 0 else 0
        
        # Determine status
        if llm_scores.H >= 0.35:
            status = "REJECT_HALLUCINATION"
        elif accuracy_pct >= 95:
            status = "EXCELLENT"
        elif accuracy_pct >= 85:
            status = "GOOD"
        elif accuracy_pct >= 70:
            status = "ACCEPTABLE"
        else:
            status = "BELOW_BASELINE"
        
        # Save to database
        self.db.save_combined_evaluation(
            doc_id, baseline_scores, llm_scores,
            delta, accuracy_pct, status, elapsed_ms
        )
        
        return EvalResult(
            doc_id=doc_id,
            baseline_score=baseline_scores.final_score,
            llm_score=llm_scores.final_score,
            delta=delta,
            accuracy_pct=accuracy_pct,
            baseline_H=baseline_scores.H,
            llm_H=llm_scores.H,
            status=status,
            llm_summary=results['llm_summary'],
            expert_summary=expert_summary
        )
    
    def evaluate(self, split: str = "test", limit: int = None) -> Dict:
        """Run combined evaluation on all documents."""
        records = self.load_govreport(split=split, limit=limit)
        
        print("\n" + "=" * 80)
        print("COMBINED NOUN RANK EVALUATION")
        print("=" * 80)
        print(f"Model:     {CONFIG['LLM_MODEL']}")
        print(f"Documents: {len(records)}")
        print(f"Algorithm: {CONFIG['ALGORITHM_VERSION']}")
        print(f"Mode:      Parallel (T1: Baseline, T2: LLM)")
        print("=" * 80 + "\n")
        
        results = []
        start_time = time.time()
        
        for i, record in enumerate(records):
            result = self.evaluate_document(record)
            results.append(result)
            
            # ANSI codes for bold/red
            BOLD = '\033[1m'
            RED = '\033[91m'
            RESET = '\033[0m'
            
            # Progress log
            status_icon = {
                'EXCELLENT': '★',
                'GOOD': '✓',
                'ACCEPTABLE': '○',
                'BELOW_BASELINE': '↓',
                'REJECT_HALLUCINATION': '✗H',
                'ERROR': '✗E'
            }.get(result.status, '?')
            
            delta_str = f"+{result.delta:.4f}" if result.delta >= 0 else f"{result.delta:.4f}"
            
            # Bold + Red if hallucination detected
            if result.status == 'REJECT_HALLUCINATION':
                logger.info(
                    f"{BOLD}{RED}[{i+1}/{len(records)}] {result.doc_id}: "
                    f"Baseline={result.baseline_score:.4f} "
                    f"LLM={result.llm_score:.4f} "
                    f"Δ={delta_str} "
                    f"Acc={result.accuracy_pct:.1f}% "
                    f"H={result.llm_H:.4f} "
                    f"[{status_icon}] ⚠️ HALLUCINATION DETECTED{RESET}"
                )
            else:
                logger.info(
                    f"[{i+1}/{len(records)}] {result.doc_id}: "
                    f"Baseline={result.baseline_score:.4f} "
                    f"LLM={result.llm_score:.4f} "
                    f"Δ={delta_str} "
                    f"Acc={result.accuracy_pct:.1f}% "
                    f"[{status_icon}]"
                )
            
            # Clear GPU memory after each document
            clear_context()
        
        # Calculate statistics
        valid_results = [r for r in results if r.status != 'ERROR']
        
        stats = {
            'total_docs': len(records),
            'evaluated': len(valid_results),
            'mean_baseline': np.mean([r.baseline_score for r in valid_results]) if valid_results else 0,
            'mean_llm': np.mean([r.llm_score for r in valid_results]) if valid_results else 0,
            'mean_delta': np.mean([r.delta for r in valid_results]) if valid_results else 0,
            'mean_accuracy': np.mean([r.accuracy_pct for r in valid_results]) if valid_results else 0,
            'std_accuracy': np.std([r.accuracy_pct for r in valid_results]) if valid_results else 0,
            'min_accuracy': np.min([r.accuracy_pct for r in valid_results]) if valid_results else 0,
            'max_accuracy': np.max([r.accuracy_pct for r in valid_results]) if valid_results else 0,
            'excellent': sum(1 for r in results if r.status == 'EXCELLENT'),
            'good': sum(1 for r in results if r.status == 'GOOD'),
            'acceptable': sum(1 for r in results if r.status == 'ACCEPTABLE'),
            'below_baseline': sum(1 for r in results if r.status == 'BELOW_BASELINE'),
            'rejected_hallucination': sum(1 for r in results if r.status == 'REJECT_HALLUCINATION'),
            'errors': sum(1 for r in results if r.status == 'ERROR'),
            'mean_baseline_H': np.mean([r.baseline_H for r in valid_results]) if valid_results else 0,
            'mean_llm_H': np.mean([r.llm_H for r in valid_results]) if valid_results else 0,
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
        print(f"   Mean Baseline Score:  {stats['mean_baseline']:.4f}")
        print(f"   Mean LLM Score:       {stats['mean_llm']:.4f}")
        print(f"   Mean Delta:           {stats['mean_delta']:+.4f}")
        
        print(f"\n📈 ACCURACY:")
        print(f"   Mean:    {stats['mean_accuracy']:.1f}%")
        print(f"   Std:     {stats['std_accuracy']:.1f}%")
        print(f"   Range:   {stats['min_accuracy']:.1f}% - {stats['max_accuracy']:.1f}%")
        
        print(f"\n🏷️  CLASSIFICATION:")
        print(f"   ★ Excellent (≥95%):       {stats['excellent']}")
        print(f"   ✓ Good (85-95%):          {stats['good']}")
        print(f"   ○ Acceptable (70-85%):    {stats['acceptable']}")
        print(f"   ↓ Below Baseline (<70%):  {stats['below_baseline']}")
        print(f"   ✗H Hallucination:         {stats['rejected_hallucination']}")
        print(f"   ✗E Errors:                {stats['errors']}")
        
        print(f"\n🔍 HALLUCINATION CHECK:")
        print(f"   Mean Baseline H:  {stats['mean_baseline_H']:.4f}")
        print(f"   Mean LLM H:       {stats['mean_llm_H']:.4f}")
        
        if stats['mean_llm_H'] < CONFIG['HALLUCINATION_THRESHOLD']:
            print(f"   → Using adaptive weights (H < {CONFIG['HALLUCINATION_THRESHOLD']})")
        else:
            print(f"   → Using standard weights (H ≥ {CONFIG['HALLUCINATION_THRESHOLD']})")
        
        print(f"\n⏱️  TIMING:")
        print(f"   Total Time: {stats['total_time']:.1f}s")
        print(f"   Per Doc:    {stats['total_time']/max(stats['evaluated'],1):.1f}s")
        
        print("\n" + "=" * 80)
        
        # Pass/Fail
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
    
    parser = argparse.ArgumentParser(description="Noun Rank Combined Evaluator")
    parser.add_argument("--host", type=str, default="http://localhost:8000",
                       help="vLLM host URL")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for vLLM")
    parser.add_argument("--db", type=str, default="noun_rank_combined.db",
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
    
    # Adjust input words based on context length
    # Reserve ~800 tokens for prompt + output
    CONFIG['MAX_INPUT_WORDS'] = int((args.context - 800) * 0.75)  # ~0.75 words per token
    CONFIG['CHUNK_SIZE'] = int(CONFIG['MAX_INPUT_WORDS'] * 0.8)
    
    logger.info(f"Context: {args.context} tokens → Max input: {CONFIG['MAX_INPUT_WORDS']} words")
    
    evaluator = CombinedEvaluator(
        db_path=args.db,
        llm_host=args.host,
        llm_api_key=args.api_key
    )
    
    stats = evaluator.evaluate(split=args.split, limit=args.limit)
    return stats


if __name__ == "__main__":
    main()
