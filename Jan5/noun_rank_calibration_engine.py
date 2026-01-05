"""
Noun Rank Calibration Engine v12.0
Multi-threaded Baseline Calibration with SQLite Storage

Architecture:
- Master Thread: Coordinates processing, computes final score, saves to DB
- Helper Threads: Calculate individual components (H, C, F, E, S) in parallel
- Parallel Document Processing: Process N documents concurrently

Author: Deepak Arora
Date: January 2026
"""

import sqlite3
import math
import threading
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Set, List, Tuple, Optional
from collections import Counter
from queue import Queue
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    'PARALLEL_DOCS': 1,          # Documents processed in parallel
    'SPACY_MODEL': 'en_core_web_sm',
    'EMBEDDING_MODEL': 'all-MiniLM-L6-v2',
    'DB_PATH': 'noun_rank.db',
    'LOG_LEVEL': logging.INFO,
    'ALGORITHM_VERSION': 'v12.0'
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
    
    # Sub-components
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
    """Thread-safe SQLite database manager."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
        self._lock = threading.Lock()
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_length INTEGER,
                summary_length INTEGER,
                source TEXT DEFAULT 'govreport',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # LLM Models table
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
        
        # Evaluations table
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
        
        # Statistics table
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
        
        # Indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_eval_doc ON evaluations(doc_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluations(model_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_eval_baseline ON evaluations(is_baseline)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_eval_score ON evaluations(final_score)')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized: {self.db_path}")
    
    def save_document(self, doc_id: str, doc_length: int, summary_length: int):
        """Save document metadata."""
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
        """Save evaluation scores."""
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
        """Save calibration run statistics."""
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
        """Get baseline score for a document."""
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
# Term Extractor (Thread-Safe)
# =============================================================================

class TermExtractor:
    """Extract terms using spaCy. Thread-safe with instance per thread."""
    
    _local = threading.local()
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or CONFIG['SPACY_MODEL']
    
    def _get_nlp(self):
        """Get thread-local spaCy model."""
        if not hasattr(self._local, 'nlp'):
            import spacy
            self._local.nlp = spacy.load(self.model_name)
        return self._local.nlp
    
    def extract(self, text: str) -> ExtractedTerms:
        """Extract all term types from text."""
        nlp = self._get_nlp()
        doc = nlp(text)
        
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
# Embedding Calculator (Thread-Safe)
# =============================================================================

class EmbeddingCalculator:
    """Calculate embeddings. Thread-safe with instance per thread."""
    
    _local = threading.local()
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or CONFIG['EMBEDDING_MODEL']
    
    def _get_model(self):
        """Get thread-local sentence transformer model."""
        if not hasattr(self._local, 'model'):
            from sentence_transformers import SentenceTransformer
            self._local.model = SentenceTransformer(self.model_name)
        return self._local.model
    
    def get_embedding(self, text: str, max_length: int = 512):
        """Get embedding, chunking if necessary."""
        import numpy as np
        model = self._get_model()
        words = text.split()
        
        if len(words) <= max_length:
            return model.encode(text)
        
        chunks = []
        for i in range(0, len(words), max_length):
            chunk = " ".join(words[i:i + max_length])
            chunks.append(model.encode(chunk))
        
        return np.mean(chunks, axis=0)
    
    def cosine_similarity(self, emb1, emb2) -> float:
        """Calculate cosine similarity."""
        import numpy as np
        dot = np.dot(emb1, emb2)
        norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
        return float(dot / (norm1 * norm2)) if norm1 and norm2 else 0.0


# =============================================================================
# Score Calculators (Run in Helper Threads)
# =============================================================================

def safe_divide(num: float, denom: float, default: float = 0.0) -> float:
    """Safe division with default for zero denominator."""
    return num / denom if denom else default


class HallucinationCalculator:
    """Calculate Hallucination Score (H) - Runs in Helper Thread."""
    
    @staticmethod
    def calculate(D: ExtractedTerms, S: ExtractedTerms) -> Tuple[float, Dict[str, float]]:
        """
        H = (H_entity × 0.30) + (H_proper × 0.25) + (H_nouns × 0.20) 
            + (H_numeric × 0.15) + (H_temporal × 0.10)
        """
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
        
        logger.debug(f"Hallucination calculated: H={H:.4f}")
        return H, components


class CoverageCalculator:
    """Calculate Coverage Score (C) - Runs in Helper Thread."""
    
    @staticmethod
    def calculate(D: ExtractedTerms, S: ExtractedTerms) -> Tuple[float, Dict[str, float]]:
        """
        C = (C_key × 0.35) + (C_entity × 0.30) + (C_numeric × 0.20) + (C_temporal × 0.15)
        """
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
        
        logger.debug(f"Coverage calculated: C={C:.4f}")
        return C, components


class FrequencyCalculator:
    """Calculate Frequency Score (F) - Runs in Helper Thread."""
    
    @staticmethod
    def calculate(D: ExtractedTerms, S: ExtractedTerms) -> float:
        """
        F = F_sum / max_F
        F_sum = Σ log(1 + D_freq[t]) for t in S_all where t in D_all
        max_F = Σ log(1 + D_freq[t]) for all t in D_key
        """
        D_key = D.get_key_terms(min_freq=2)
        
        F_sum = sum(math.log(1 + D.freq.get(t, 0)) for t in S.all_terms if t in D.all_terms)
        max_F = sum(math.log(1 + D.freq.get(t, 0)) for t in D_key)
        
        F = min(safe_divide(F_sum, max_F, 1.0), 1.0)
        
        logger.debug(f"Frequency calculated: F={F:.4f}")
        return F


class EntityCalculator:
    """Calculate Entity Score (E) - Runs in Helper Thread."""
    
    @staticmethod
    def calculate(D: ExtractedTerms, S: ExtractedTerms) -> float:
        """
        E = E_matched / E_total
        Weights: PERSON/ORG/GPE = 1.0, DATE/MONEY = 0.8
        """
        if not (D.entities or D.temporal or D.numeric):
            return 1.0
        
        E_matched, E_total = 0.0, 0.0
        
        # Entities (PERSON, ORG, GPE) weight 1.0
        for ent in D.entities:
            E_total += 1.0
            if ent in S.entities:
                E_matched += 1.0
        
        # Temporal (DATE, TIME) weight 0.8
        for ent in D.temporal:
            E_total += 0.8
            if ent in S.temporal:
                E_matched += 0.8
        
        # Numeric (MONEY) weight 0.8
        for ent in D.numeric:
            E_total += 0.8
            if ent in S.numeric:
                E_matched += 0.8
        
        E = safe_divide(E_matched, E_total, 1.0)
        
        logger.debug(f"Entity calculated: E={E:.4f}")
        return E


class SemanticCalculator:
    """Calculate Semantic Score (S) - Runs in Helper Thread."""
    
    def __init__(self, embedder: EmbeddingCalculator):
        self.embedder = embedder
    
    def calculate(self, doc_text: str, summary_text: str) -> float:
        """
        S = cosine_similarity(embed(D), embed(S))
        """
        emb_D = self.embedder.get_embedding(doc_text)
        emb_S = self.embedder.get_embedding(summary_text)
        S = self.embedder.cosine_similarity(emb_D, emb_S)
        
        logger.debug(f"Semantic calculated: S={S:.4f}")
        return S


# =============================================================================
# Document Processor (Coordinates Helper Threads)
# =============================================================================

class DocumentProcessor:
    """Process a single document with parallel helper threads."""
    
    def __init__(self, extractor: TermExtractor, embedder: EmbeddingCalculator):
        self.extractor = extractor
        self.embedder = embedder
        self.semantic_calc = SemanticCalculator(embedder)
    
    def process(self, doc_id: str, doc_text: str, summary_text: str) -> ComponentScores:
        """
        Process document with 5 helper threads (one per component).
        Master thread waits for all, computes final score.
        """
        start_time = time.time()
        logger.info(f"Processing {doc_id}...")
        
        # Extract terms (shared by H, C, F, E)
        D = self.extractor.extract(doc_text)
        S = self.extractor.extract(summary_text)
        
        # Results container
        results = {}
        
        # Helper thread functions
        def calc_hallucination():
            H, comps = HallucinationCalculator.calculate(D, S)
            results['H'] = H
            results['H_components'] = comps
        
        def calc_coverage():
            C, comps = CoverageCalculator.calculate(D, S)
            results['C'] = C
            results['C_components'] = comps
        
        def calc_frequency():
            results['F'] = FrequencyCalculator.calculate(D, S)
        
        def calc_entity():
            results['E'] = EntityCalculator.calculate(D, S)
        
        def calc_semantic():
            results['S'] = self.semantic_calc.calculate(doc_text, summary_text)
        
        # Spawn helper threads
        threads = [
            threading.Thread(target=calc_hallucination, name=f"{doc_id}-H"),
            threading.Thread(target=calc_coverage, name=f"{doc_id}-C"),
            threading.Thread(target=calc_frequency, name=f"{doc_id}-F"),
            threading.Thread(target=calc_entity, name=f"{doc_id}-E"),
            threading.Thread(target=calc_semantic, name=f"{doc_id}-S"),
        ]
        
        # Start all helper threads
        for t in threads:
            t.start()
        
        # Wait for all helper threads to complete
        for t in threads:
            t.join()
        
        # Master thread computes final score
        w = WEIGHTS['FINAL']
        final_score = (
            (1 - results['H']) * w['H'] +
            results['C'] * w['C'] +
            results['F'] * w['F'] +
            results['E'] * w['E'] +
            results['S'] * w['S']
        )
        
        # Build scores object
        scores = ComponentScores(
            H=results['H'], C=results['C'], F=results['F'],
            E=results['E'], S=results['S'],
            H_entity=results['H_components']['H_entity'],
            H_proper=results['H_components']['H_proper'],
            H_nouns=results['H_components']['H_nouns'],
            H_temporal=results['H_components']['H_temporal'],
            H_numeric=results['H_components']['H_numeric'],
            C_key=results['C_components']['C_key'],
            C_entity=results['C_components']['C_entity'],
            C_temporal=results['C_components']['C_temporal'],
            C_numeric=results['C_components']['C_numeric'],
            final_score=final_score
        )
        
        elapsed = (time.time() - start_time) * 1000
        logger.info(f"Completed {doc_id}: score={final_score:.4f} ({elapsed:.0f}ms)")
        
        return scores, elapsed


# =============================================================================
# Calibration Engine (Master Controller)
# =============================================================================

class CalibrationEngine:
    """
    Master controller for calibration.
    Processes documents in parallel, saves to SQLite.
    """
    
    def __init__(self, db_path: str = None):
        self.db = DatabaseManager(db_path or CONFIG['DB_PATH'])
        self.extractor = TermExtractor()
        self.embedder = EmbeddingCalculator()
        self.processor = DocumentProcessor(self.extractor, self.embedder)
    
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
        """Process single document (runs in thread pool)."""
        doc_id = record['doc_id']
        doc_text = record['document']
        summary_text = record['summary']
        
        # Save document metadata
        self.db.save_document(doc_id, len(doc_text.split()), len(summary_text.split()))
        
        # Process with helper threads
        scores, elapsed = self.processor.process(doc_id, doc_text, summary_text)
        
        # Save to database
        self.db.save_evaluation(doc_id, scores, is_baseline=True, processing_time_ms=elapsed)
        
        return doc_id, scores, elapsed
    
    def calibrate(self, split: str = "test", limit: int = None, 
                  parallel_docs: int = None) -> Dict:
        """
        Run calibration on GovReport dataset.
        
        Args:
            split: Dataset split ('train', 'validation', 'test')
            limit: Limit number of documents (None for all)
            parallel_docs: Number of documents to process in parallel
        
        Returns:
            Dict with statistics
        """
        parallel = parallel_docs or CONFIG['PARALLEL_DOCS']
        records = self.load_govreport(split=split, limit=limit)
        
        total_docs = len(records)
        total_runs = math.ceil(total_docs / parallel)
        
        logger.info("=" * 60)
        logger.info(f"CALIBRATION STARTED")
        logger.info(f"Total documents: {total_docs}")
        logger.info(f"Parallel processing: {parallel} documents at a time")
        logger.info(f"Total runs: {total_runs}")
        logger.info("=" * 60)
        
        start_time = time.time()
        all_scores = []
        completed = 0
        
        # Process in parallel batches
        with ThreadPoolExecutor(max_workers=parallel, thread_name_prefix="DocWorker") as executor:
            futures = {executor.submit(self.process_document, rec): rec for rec in records}
            
            for future in as_completed(futures):
                try:
                    doc_id, scores, elapsed = future.result()
                    all_scores.append(scores.final_score)
                    completed += 1
                    
                    if completed % 10 == 0 or completed == total_docs:
                        progress = (completed / total_docs) * 100
                        logger.info(f"Progress: {completed}/{total_docs} ({progress:.1f}%)")
                        
                except Exception as e:
                    rec = futures[future]
                    logger.error(f"Error processing {rec['doc_id']}: {e}")
        
        # Calculate statistics
        import numpy as np
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
            'total_time': total_time
        }
        
        # Save statistics to database
        self.db.save_statistics(stats)
        
        # Print summary
        self._print_summary(stats)
        
        return stats
    
    def _print_summary(self, stats: Dict):
        """Print calibration summary."""
        print("\n" + "=" * 60)
        print("CALIBRATION COMPLETE")
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
        print(f"{'Avg Time per Document (ms)':<35} {(stats['total_time']/stats['count'])*1000:>15.1f}")
        print("=" * 60)
        
        # Success criteria check
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
        description="Noun Rank Calibration Engine v12.0"
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
        "--parallel", type=int, default=10,
        help="Documents to process in parallel (default: 10)"
    )
    parser.add_argument(
        "--db", type=str, default="noun_rank.db",
        help="SQLite database path (default: noun_rank.db)"
    )
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['DB_PATH'] = args.db
    CONFIG['PARALLEL_DOCS'] = args.parallel
    
    # Run calibration
    engine = CalibrationEngine(db_path=args.db)
    stats = engine.calibrate(
        split=args.split,
        limit=args.limit,
        parallel_docs=args.parallel
    )
    
    return stats


if __name__ == "__main__":
    main()
