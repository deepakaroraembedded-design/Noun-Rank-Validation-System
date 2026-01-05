"""
Noun Rank Calibration Engine v12.1 (GPU Optimized for vast.ai A100)
High-Performance Baseline Calibration with SQLite Storage

Optimizations for A100 GPU:
- CUDA-accelerated embeddings with FP16
- Large batch processing (A100 has 40-80GB VRAM)
- High parallelism for document processing
- Batched embedding computation across documents
- No text truncation (plenty of memory)
- Multi-GPU support if available

Author: Deepak Arora
Date: January 2026

Usage on vast.ai:
    python noun_rank_calibration_engine_gpu.py --split test --parallel 16 --batch-size 64
"""

import os
import sqlite3
import math
import threading
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
from collections import Counter
from datetime import datetime
from queue import Queue

# =============================================================================
# Configuration (Optimized for A100 GPU)
# =============================================================================

CONFIG = {
    'PARALLEL_DOCS': 16,          # High parallelism - A100 can handle it
    'SPACY_MODEL': 'en_core_web_sm',
    'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
    'DB_PATH': 'noun_rank.db',
    'LOG_LEVEL': logging.INFO,
    'ALGORITHM_VERSION': 'v12.1-gpu-a100',
    'EMBEDDING_BATCH_SIZE': 64,   # Large batches for GPU efficiency
    'MAX_TEXT_LENGTH': None,      # No truncation - A100 has plenty of memory
    'USE_FP16': True,             # Half precision for 2x speed
    'DEVICE': 'cuda',             # GPU acceleration
}

# Weights from v12.0 spec
WEIGHTS = {
    'H': {'entity': 0.30, 'proper': 0.25, 'nouns': 0.20, 'numeric': 0.15, 'temporal': 0.10},
    'C': {'key': 0.35, 'entity': 0.30, 'numeric': 0.20, 'temporal': 0.15},
    'FINAL': {'H': 0.25, 'C': 0.25, 'F': 0.20, 'E': 0.20, 'S': 0.10},
    'ENTITY': {'PERSON': 1.0, 'ORG': 1.0, 'GPE': 1.0, 'DATE': 0.8, 'MONEY': 0.8}
}

# =============================================================================
# Logging Setup
# =============================================================================

logging.basicConfig(
    level=CONFIG['LOG_LEVEL'],
    format='%(asctime)s [%(threadName)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# GPU Setup and Diagnostics
# =============================================================================

def setup_gpu():
    """Setup and diagnose GPU environment."""
    import torch
    
    print("=" * 60)
    print("GPU DIAGNOSTICS")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available! Falling back to CPU.")
        CONFIG['DEVICE'] = 'cpu'
        CONFIG['USE_FP16'] = False
        CONFIG['EMBEDDING_BATCH_SIZE'] = 8
        CONFIG['PARALLEL_DOCS'] = 4
        return
    
    # GPU info
    gpu_count = torch.cuda.device_count()
    print(f"CUDA Available: Yes")
    print(f"GPU Count: {gpu_count}")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / (1024**3)
        print(f"\nGPU {i}: {props.name}")
        print(f"  - Memory: {memory_gb:.1f} GB")
        print(f"  - Compute Capability: {props.major}.{props.minor}")
        print(f"  - Multi-Processor Count: {props.multi_processor_count}")
    
    # Use first GPU
    torch.cuda.set_device(0)
    CONFIG['DEVICE'] = 'cuda:0'
    
    # Optimize batch size based on GPU memory
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if gpu_memory >= 70:  # A100 80GB
        CONFIG['EMBEDDING_BATCH_SIZE'] = 128
        CONFIG['PARALLEL_DOCS'] = 32
    elif gpu_memory >= 35:  # A100 40GB
        CONFIG['EMBEDDING_BATCH_SIZE'] = 64
        CONFIG['PARALLEL_DOCS'] = 16
    elif gpu_memory >= 20:  # A10, RTX 3090
        CONFIG['EMBEDDING_BATCH_SIZE'] = 32
        CONFIG['PARALLEL_DOCS'] = 8
    else:  # Smaller GPUs
        CONFIG['EMBEDDING_BATCH_SIZE'] = 16
        CONFIG['PARALLEL_DOCS'] = 4
    
    print(f"\nOptimized Settings:")
    print(f"  - Batch Size: {CONFIG['EMBEDDING_BATCH_SIZE']}")
    print(f"  - Parallel Docs: {CONFIG['PARALLEL_DOCS']}")
    print(f"  - FP16: {CONFIG['USE_FP16']}")
    print("=" * 60)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExtractedTerms:
    """Container for extracted terms from a document or summary."""
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
    """Container for individual component scores."""
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


# =============================================================================
# Database Manager
# =============================================================================

class DatabaseManager:
    """Thread-safe SQLite database manager with batch insert support."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._batch_queue = []
        self._batch_size = 100
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            # Performance optimizations
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=10000")
        return self._local.conn
    
    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
                parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_name, model_version)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS evaluations (
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT NOT NULL,
                model_id INTEGER,
                is_baseline BOOLEAN DEFAULT FALSE,
                final_score REAL NOT NULL,
                H REAL, C REAL, F REAL, E REAL, S REAL,
                H_entity REAL, H_proper REAL, H_nouns REAL,
                H_temporal REAL, H_numeric REAL,
                C_key REAL, C_entity REAL,
                C_temporal REAL, C_numeric REAL,
                accuracy REAL,
                algorithm_version TEXT,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processing_time_ms REAL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
                FOREIGN KEY (model_id) REFERENCES llm_models(model_id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS calibration_stats (
                stat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                doc_count INTEGER,
                mean_score REAL,
                std_score REAL,
                min_score REAL,
                max_score REAL,
                p50_score REAL,
                p95_score REAL,
                total_time_sec REAL,
                algorithm_version TEXT
            )
        ''')
        
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_eval_doc ON evaluations(doc_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluations(model_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_eval_baseline ON evaluations(is_baseline)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_eval_score ON evaluations(final_score)')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def save_document(self, doc_id: str, doc_length: int, summary_length: int):
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR IGNORE INTO documents (doc_id, doc_length, summary_length)
                VALUES (?, ?, ?)
            ''', (doc_id, doc_length, summary_length))
            conn.commit()
    
    def save_evaluation(self, doc_id: str, scores: ComponentScores, 
                        is_baseline: bool = True, model_id: int = None,
                        processing_time_ms: float = 0):
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO evaluations (
                    doc_id, model_id, is_baseline, final_score,
                    H, C, F, E, S,
                    H_entity, H_proper, H_nouns, H_temporal, H_numeric,
                    C_key, C_entity, C_temporal, C_numeric,
                    algorithm_version, processing_time_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                doc_id, model_id, is_baseline, scores.final_score,
                scores.H, scores.C, scores.F, scores.E, scores.S,
                scores.H_entity, scores.H_proper, scores.H_nouns,
                scores.H_temporal, scores.H_numeric,
                scores.C_key, scores.C_entity, scores.C_temporal, scores.C_numeric,
                CONFIG['ALGORITHM_VERSION'], processing_time_ms
            ))
            conn.commit()
    
    def save_statistics(self, stats: Dict):
        with self._lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO calibration_stats (
                    doc_count, mean_score, std_score, min_score, max_score,
                    p50_score, p95_score, total_time_sec, algorithm_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                stats['count'], stats['mean'], stats['std'],
                stats['min'], stats['max'], stats['p50'], stats['p95'],
                stats['total_time'], CONFIG['ALGORITHM_VERSION']
            ))
            conn.commit()
    
    def get_baseline(self, doc_id: str) -> Optional[float]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT final_score FROM evaluations
            WHERE doc_id = ? AND is_baseline = TRUE
            ORDER BY run_date DESC LIMIT 1
        ''', (doc_id,))
        row = cursor.fetchone()
        return row['final_score'] if row else None


# =============================================================================
# Model Manager (Singleton with GPU Support)
# =============================================================================

class ModelManager:
    """
    Singleton that loads models ONCE on GPU and shares them across threads.
    Optimized for A100 with FP16 and large batch processing.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._nlp = None
        self._embedder = None
        self._embed_lock = threading.Lock()
        self._initialized = True
    
    def load_models(self):
        """Load all models on GPU."""
        import torch
        
        logger.info("=" * 50)
        logger.info("Loading models on GPU...")
        logger.info("=" * 50)
        
        # Load spaCy (CPU - it's fast enough)
        start = time.time()
        import spacy
        self._nlp = spacy.load(CONFIG['SPACY_MODEL'])
        logger.info(f"✓ spaCy loaded: {CONFIG['SPACY_MODEL']} ({time.time()-start:.1f}s)")
        
        # Load SentenceTransformer on GPU with FP16
        start = time.time()
        from sentence_transformers import SentenceTransformer
        
        self._embedder = SentenceTransformer(
            CONFIG['EMBEDDING_MODEL'],
            device=CONFIG['DEVICE']
        )
        
        # Convert to FP16 for faster inference on A100
        if CONFIG['USE_FP16'] and CONFIG['DEVICE'].startswith('cuda'):
            self._embedder.half()
            logger.info("✓ Model converted to FP16 for faster inference")
        
        logger.info(f"✓ SentenceTransformer loaded on {CONFIG['DEVICE']}: {CONFIG['EMBEDDING_MODEL']} ({time.time()-start:.1f}s)")
        
        # Warm up GPU
        logger.info("Warming up GPU...")
        _ = self._embedder.encode(["warmup text"], show_progress_bar=False)
        
        # Show GPU memory usage
        if CONFIG['DEVICE'].startswith('cuda'):
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        logger.info("=" * 50)
        logger.info("All models loaded and ready!")
        logger.info("=" * 50)
    
    @property
    def nlp(self):
        if self._nlp is None:
            self.load_models()
        return self._nlp
    
    @property
    def embedder(self):
        if self._embedder is None:
            self.load_models()
        return self._embedder
    
    def encode(self, texts: List[str], show_progress: bool = False) -> np.ndarray:
        """Thread-safe batch encoding on GPU."""
        if isinstance(texts, str):
            texts = [texts]
        
        with self._embed_lock:
            return self._embedder.encode(
                texts, 
                show_progress_bar=show_progress,
                batch_size=CONFIG['EMBEDDING_BATCH_SIZE'],
                convert_to_numpy=True
            )
    
    def encode_batch_async(self, texts: List[str]) -> np.ndarray:
        """Encode large batches efficiently."""
        return self.encode(texts, show_progress=len(texts) > 100)


# =============================================================================
# Term Extractor
# =============================================================================

class TermExtractor:
    """Extract terms using shared spaCy model."""
    
    def __init__(self):
        self.models = ModelManager()
    
    def extract(self, text: str) -> ExtractedTerms:
        """Extract all term types from text."""
        # Optional truncation for very extreme cases
        if CONFIG['MAX_TEXT_LENGTH'] and len(text) > CONFIG['MAX_TEXT_LENGTH']:
            text = text[:CONFIG['MAX_TEXT_LENGTH']]
        
        doc = self.models.nlp(text)
        
        nouns, proper, entities, temporal, numeric = set(), set(), set(), set(), set()
        all_terms = []
        
        entity_types = {"ORG", "PERSON", "GPE"}
        temporal_types = {"DATE", "TIME"}
        numeric_types = {"MONEY", "CARDINAL", "PERCENT", "QUANTITY"}
        
        for token in doc:
            if token.pos_ == "NOUN":
                lemma = token.lemma_.lower()
                nouns.add(lemma)
                all_terms.append(lemma)
            elif token.pos_ == "PROPN":
                proper.add(token.text)
                all_terms.append(token.text)
        
        for ent in doc.ents:
            ent_text = ent.text.strip()
            if ent.label_ in entity_types:
                entities.add(ent_text)
                all_terms.append(ent_text)
            elif ent.label_ in temporal_types:
                temporal.add(ent_text)
                all_terms.append(ent_text)
            elif ent.label_ in numeric_types:
                numeric.add(ent_text)
                all_terms.append(ent_text)
        
        return ExtractedTerms(
            nouns=nouns, proper=proper, entities=entities,
            temporal=temporal, numeric=numeric, freq=dict(Counter(all_terms))
        )


# =============================================================================
# Embedding Calculator (GPU Optimized)
# =============================================================================

class EmbeddingCalculator:
    """Calculate embeddings using GPU-accelerated shared model."""
    
    def __init__(self):
        self.models = ModelManager()
    
    def get_embedding(self, text: str, max_words: int = 512) -> np.ndarray:
        """Get embedding, chunking if necessary."""
        words = text.split()
        
        if len(words) <= max_words:
            result = self.models.encode(text)
            return result[0] if len(result.shape) > 1 else result
        
        # Chunk long texts and batch encode
        chunks = []
        for i in range(0, len(words), max_words):
            chunk = " ".join(words[i:i + max_words])
            chunks.append(chunk)
        
        # Batch encode all chunks at once (GPU efficient)
        embeddings = self.models.encode(chunks)
        return np.mean(embeddings, axis=0)
    
    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for multiple texts at once - most efficient for GPU."""
        return self.models.encode(texts)
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot = np.dot(emb1, emb2)
        norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        return float(dot / (norm1 * norm2)) if norm1 and norm2 else 0.0


# =============================================================================
# Score Calculators
# =============================================================================

def safe_divide(num: float, denom: float, default: float = 0.0) -> float:
    return num / denom if denom else default


def calculate_hallucination(D: ExtractedTerms, S: ExtractedTerms) -> Tuple[float, Dict[str, float]]:
    H_nouns = safe_divide(len(S.nouns - D.nouns), len(S.nouns), 0.0)
    H_proper = safe_divide(len(S.proper - D.proper), len(S.proper), 0.0)
    H_entity = safe_divide(len(S.entities - D.entities), len(S.entities), 0.0)
    H_temporal = safe_divide(len(S.temporal - D.temporal), len(S.temporal), 0.0)
    H_numeric = safe_divide(len(S.numeric - D.numeric), len(S.numeric), 0.0)
    
    w = WEIGHTS['H']
    H = (H_entity * w['entity'] + H_proper * w['proper'] + H_nouns * w['nouns'] +
         H_numeric * w['numeric'] + H_temporal * w['temporal'])
    
    components = {
        'H_entity': H_entity, 'H_proper': H_proper, 'H_nouns': H_nouns,
        'H_temporal': H_temporal, 'H_numeric': H_numeric
    }
    return H, components


def calculate_coverage(D: ExtractedTerms, S: ExtractedTerms) -> Tuple[float, Dict[str, float]]:
    D_key = D.get_key_terms(min_freq=2)
    
    C_key = safe_divide(len(S.all_terms & D_key), len(D_key), 1.0)
    C_entity = safe_divide(len(S.entities & D.entities), len(D.entities), 1.0)
    C_temporal = safe_divide(len(S.temporal & D.temporal), len(D.temporal), 1.0)
    C_numeric = safe_divide(len(S.numeric & D.numeric), len(D.numeric), 1.0)
    
    w = WEIGHTS['C']
    C = (C_key * w['key'] + C_entity * w['entity'] +
         C_numeric * w['numeric'] + C_temporal * w['temporal'])
    
    components = {
        'C_key': C_key, 'C_entity': C_entity,
        'C_temporal': C_temporal, 'C_numeric': C_numeric
    }
    return C, components


def calculate_frequency(D: ExtractedTerms, S: ExtractedTerms) -> float:
    D_key = D.get_key_terms(min_freq=2)
    F_sum = sum(math.log(1 + D.freq.get(t, 0)) for t in S.all_terms if t in D.all_terms)
    max_F = sum(math.log(1 + D.freq.get(t, 0)) for t in D_key)
    return min(safe_divide(F_sum, max_F, 1.0), 1.0)


def calculate_entity(D: ExtractedTerms, S: ExtractedTerms) -> float:
    if not (D.entities or D.temporal or D.numeric):
        return 1.0
    
    E_matched, E_total = 0.0, 0.0
    
    for ent in D.entities:
        E_total += 1.0
        if ent in S.entities:
            E_matched += 1.0
    
    for ent in D.temporal:
        E_total += 0.8
        if ent in S.temporal:
            E_matched += 0.8
    
    for ent in D.numeric:
        E_total += 0.8
        if ent in S.numeric:
            E_matched += 0.8
    
    return safe_divide(E_matched, E_total, 1.0)


# =============================================================================
# Document Processor
# =============================================================================

class DocumentProcessor:
    """Process documents with GPU-accelerated embeddings."""
    
    def __init__(self):
        self.extractor = TermExtractor()
        self.embedder = EmbeddingCalculator()
    
    def process(self, doc_id: str, doc_text: str, summary_text: str) -> Tuple[ComponentScores, float]:
        """Process document and return scores."""
        start_time = time.time()
        
        # Extract terms
        D = self.extractor.extract(doc_text)
        S = self.extractor.extract(summary_text)
        
        # Calculate all components
        H, H_components = calculate_hallucination(D, S)
        C, C_components = calculate_coverage(D, S)
        F = calculate_frequency(D, S)
        E = calculate_entity(D, S)
        
        # Semantic similarity (GPU accelerated)
        emb_D = self.embedder.get_embedding(doc_text)
        emb_S = self.embedder.get_embedding(summary_text)
        S_score = self.embedder.cosine_similarity(emb_D, emb_S)
        
        # Compute final score
        w = WEIGHTS['FINAL']
        final_score = (
            (1 - H) * w['H'] +
            C * w['C'] +
            F * w['F'] +
            E * w['E'] +
            S_score * w['S']
        )
        
        scores = ComponentScores(
            H=H, C=C, F=F, E=E, S=S_score,
            H_entity=H_components['H_entity'],
            H_proper=H_components['H_proper'],
            H_nouns=H_components['H_nouns'],
            H_temporal=H_components['H_temporal'],
            H_numeric=H_components['H_numeric'],
            C_key=C_components['C_key'],
            C_entity=C_components['C_entity'],
            C_temporal=C_components['C_temporal'],
            C_numeric=C_components['C_numeric'],
            final_score=final_score
        )
        
        elapsed = (time.time() - start_time) * 1000
        return scores, elapsed


# =============================================================================
# Batch Processor (Optimized for GPU)
# =============================================================================

class BatchProcessor:
    """
    Process documents in batches for optimal GPU utilization.
    Batches embedding computations across multiple documents.
    """
    
    def __init__(self):
        self.extractor = TermExtractor()
        self.embedder = EmbeddingCalculator()
    
    def process_batch(self, records: List[Dict]) -> List[Tuple[str, ComponentScores, float]]:
        """Process a batch of documents with batched GPU embeddings."""
        start_time = time.time()
        
        # Extract terms for all documents (CPU - parallelizable)
        extracted = []
        for rec in records:
            D = self.extractor.extract(rec['document'])
            S = self.extractor.extract(rec['summary'])
            extracted.append((rec['doc_id'], rec['document'], rec['summary'], D, S))
        
        # Batch compute all embeddings at once (GPU efficient)
        all_texts = []
        for doc_id, doc_text, summary_text, D, S in extracted:
            all_texts.append(doc_text)
            all_texts.append(summary_text)
        
        # Single GPU batch call for all embeddings
        all_embeddings = self.embedder.get_embeddings_batch(all_texts)
        
        # Compute scores
        results = []
        for i, (doc_id, doc_text, summary_text, D, S) in enumerate(extracted):
            emb_D = all_embeddings[i * 2]
            emb_S = all_embeddings[i * 2 + 1]
            
            H, H_components = calculate_hallucination(D, S)
            C, C_components = calculate_coverage(D, S)
            F = calculate_frequency(D, S)
            E = calculate_entity(D, S)
            S_score = self.embedder.cosine_similarity(emb_D, emb_S)
            
            w = WEIGHTS['FINAL']
            final_score = (
                (1 - H) * w['H'] +
                C * w['C'] +
                F * w['F'] +
                E * w['E'] +
                S_score * w['S']
            )
            
            scores = ComponentScores(
                H=H, C=C, F=F, E=E, S=S_score,
                H_entity=H_components['H_entity'],
                H_proper=H_components['H_proper'],
                H_nouns=H_components['H_nouns'],
                H_temporal=H_components['H_temporal'],
                H_numeric=H_components['H_numeric'],
                C_key=C_components['C_key'],
                C_entity=C_components['C_entity'],
                C_temporal=C_components['C_temporal'],
                C_numeric=C_components['C_numeric'],
                final_score=final_score
            )
            
            elapsed = (time.time() - start_time) * 1000 / len(records)
            results.append((doc_id, scores, elapsed))
        
        return results


# =============================================================================
# Calibration Engine (GPU Optimized)
# =============================================================================

class CalibrationEngine:
    """Master controller for GPU-accelerated calibration."""
    
    def __init__(self, db_path: str = None):
        # Setup GPU first
        setup_gpu()
        
        self.db = DatabaseManager(db_path or CONFIG['DB_PATH'])
        self.processor = DocumentProcessor()
        self.batch_processor = BatchProcessor()
        
        # Preload models on GPU
        logger.info("Preloading models on GPU...")
        ModelManager().load_models()
    
    def load_govreport(self, split: str = "test", limit: int = None) -> List[Dict]:
        """Load GovReport dataset."""
        from datasets import load_dataset
        
        logger.info(f"Loading GovReport dataset (split={split}, limit={limit})...")
        dataset = load_dataset("ccdv/govreport-summarization", split=split)
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        records = []
        for idx, item in enumerate(dataset):
            records.append({
                'doc_id': f"doc_{idx:05d}",
                'document': item['report'],
                'summary': item['summary']
            })
        
        logger.info(f"Loaded {len(records)} documents")
        return records
    
    def process_document(self, record: Dict) -> Tuple[str, ComponentScores, float]:
        """Process single document."""
        doc_id = record['doc_id']
        doc_text = record['document']
        summary_text = record['summary']
        
        self.db.save_document(doc_id, len(doc_text.split()), len(summary_text.split()))
        scores, elapsed = self.processor.process(doc_id, doc_text, summary_text)
        self.db.save_evaluation(doc_id, scores, is_baseline=True, processing_time_ms=elapsed)
        
        return doc_id, scores, elapsed
    
    def calibrate(self, split: str = "test", limit: int = None, 
                  parallel_docs: int = None, use_batching: bool = True) -> Dict:
        """Run calibration on GovReport dataset."""
        parallel = parallel_docs or CONFIG['PARALLEL_DOCS']
        records = self.load_govreport(split=split, limit=limit)
        
        total_docs = len(records)
        
        logger.info("=" * 60)
        logger.info(f"CALIBRATION STARTED (GPU Mode)")
        logger.info(f"Total documents: {total_docs}")
        logger.info(f"Parallel processing: {parallel} documents at a time")
        logger.info(f"Batch size: {CONFIG['EMBEDDING_BATCH_SIZE']}")
        logger.info(f"Device: {CONFIG['DEVICE']}")
        logger.info(f"FP16: {CONFIG['USE_FP16']}")
        logger.info("=" * 60)
        
        start_time = time.time()
        all_scores = []
        completed = 0
        
        if use_batching and total_docs > 10:
            # Batch processing mode (more GPU efficient for large runs)
            batch_size = min(CONFIG['EMBEDDING_BATCH_SIZE'], total_docs)
            
            for i in range(0, total_docs, batch_size):
                batch = records[i:i + batch_size]
                
                try:
                    results = self.batch_processor.process_batch(batch)
                    
                    for doc_id, scores, elapsed in results:
                        # Save to DB
                        rec = next(r for r in batch if r['doc_id'] == doc_id)
                        self.db.save_document(doc_id, len(rec['document'].split()), len(rec['summary'].split()))
                        self.db.save_evaluation(doc_id, scores, is_baseline=True, processing_time_ms=elapsed)
                        
                        all_scores.append(scores.final_score)
                        completed += 1
                    
                    progress = (completed / total_docs) * 100
                    elapsed_time = time.time() - start_time
                    rate = completed / elapsed_time
                    eta = (total_docs - completed) / rate if rate > 0 else 0
                    logger.info(f"Progress: {completed}/{total_docs} ({progress:.1f}%) - {rate:.1f} docs/sec - ETA: {eta:.0f}s")
                    
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
        else:
            # Individual processing with thread pool
            with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="DocWorker") as executor:
                futures = {executor.submit(self.process_document, rec): rec for rec in records}
                
                for future in as_completed(futures):
                    try:
                        doc_id, scores, elapsed = future.result()
                        all_scores.append(scores.final_score)
                        completed += 1
                        
                        if completed % 10 == 0 or completed == total_docs:
                            progress = (completed / total_docs) * 100
                            elapsed_time = time.time() - start_time
                            rate = completed / elapsed_time
                            eta = (total_docs - completed) / rate if rate > 0 else 0
                            logger.info(f"Progress: {completed}/{total_docs} ({progress:.1f}%) - {rate:.1f} docs/sec - ETA: {eta:.0f}s")
                            
                    except Exception as e:
                        rec = futures[future]
                        logger.error(f"Error processing {rec['doc_id']}: {e}")
        
        # Calculate statistics
        scores_arr = np.array(all_scores)
        total_time = time.time() - start_time
        
        stats = {
            'count': len(all_scores),
            'mean': float(np.mean(scores_arr)),
            'std': float(np.std(scores_arr)),
            'min': float(np.min(scores_arr)),
            'max': float(np.max(scores_arr)),
            'p50': float(np.percentile(scores_arr, 50)),
            'p95': float(np.percentile(scores_arr, 95)),
            'total_time': total_time,
            'docs_per_sec': len(all_scores) / total_time
        }
        
        self.db.save_statistics(stats)
        self._print_summary(stats)
        
        return stats
    
    def _print_summary(self, stats: Dict):
        """Print calibration summary."""
        print("\n" + "=" * 60)
        print("CALIBRATION COMPLETE (GPU Mode)")
        print("=" * 60)
        print(f"{'Metric':<35} {'Value':>15}")
        print("-" * 60)
        print(f"{'Documents Processed':<35} {stats['count']:>15}")
        print(f"{'Mean Baseline Score':<35} {stats['mean']:>15.4f}")
        print(f"{'Standard Deviation (σ)':<35} {stats['std']:>15.4f}")
        print(f"{'Minimum Score':<35} {stats['min']:>15.4f}")
        print(f"{'Maximum Score':<35} {stats['max']:>15.4f}")
        print(f"{'Median (50th percentile)':<35} {stats['p50']:>15.4f}")
        print(f"{'95th Percentile':<35} {stats['p95']:>15.4f}")
        print(f"{'Total Time (seconds)':<35} {stats['total_time']:>15.1f}")
        print(f"{'Throughput (docs/sec)':<35} {stats['docs_per_sec']:>15.2f}")
        print(f"{'Avg Time per Document (ms)':<35} {(stats['total_time']/stats['count'])*1000:>15.1f}")
        print("=" * 60)
        
        print("\nSUCCESS CRITERIA:")
        if stats['std'] < 0.10:
            print(f"  ✓ PASS: Baseline consistency σ = {stats['std']:.4f} < 0.10")
        else:
            print(f"  ✗ FAIL: Baseline consistency σ = {stats['std']:.4f} >= 0.10")
        
        print(f"\nDatabase saved to: {self.db.db_path}")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run calibration from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Noun Rank Calibration Engine v12.1 (GPU Optimized for A100)"
    )
    parser.add_argument(
        "--split", type=str, default="test",
        choices=["train", "validation", "test"],
        help="Dataset split (default: test)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of documents (default: all)"
    )
    parser.add_argument(
        "--parallel", type=int, default=None,
        help="Documents to process in parallel (default: auto based on GPU)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Embedding batch size (default: auto based on GPU memory)"
    )
    parser.add_argument(
        "--db", type=str, default="noun_rank.db",
        help="SQLite database path (default: noun_rank.db)"
    )
    parser.add_argument(
        "--no-fp16", action="store_true",
        help="Disable FP16 (use FP32)"
    )
    parser.add_argument(
        "--no-batch", action="store_true",
        help="Disable batch processing (use individual document processing)"
    )
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['DB_PATH'] = args.db
    if args.parallel:
        CONFIG['PARALLEL_DOCS'] = args.parallel
    if args.batch_size:
        CONFIG['EMBEDDING_BATCH_SIZE'] = args.batch_size
    if args.no_fp16:
        CONFIG['USE_FP16'] = False
    
    # Run calibration
    engine = CalibrationEngine(db_path=args.db)
    stats = engine.calibrate(
        split=args.split,
        limit=args.limit,
        parallel_docs=args.parallel,
        use_batching=not args.no_batch
    )
    
    return stats


if __name__ == "__main__":
    main()
