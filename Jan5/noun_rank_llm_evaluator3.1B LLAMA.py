"""
Noun Rank LLM Evaluator v12.1
Evaluate LLaMA 3.1 8B summaries against expert baselines

Usage:
    python noun_rank_llm_evaluator.py --host http://localhost:8000 --limit 10
    
With SSH tunnel:
    Terminal 1: ssh -p PORT root@HOST -L 8000:localhost:8000
    Terminal 2: python noun_rank_llm_evaluator.py --limit 10

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
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from collections import Counter
import math

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    # === LLM Settings ===
    'LLM_HOST': 'http://localhost:8000',  # vLLM endpoint
    'LLM_MODEL': 'meta-llama/Meta-Llama-3.1-8B-Instruct',  # Fits 24GB GPU
    'LLM_API_KEY': None,  # Set if using API key
    
    # === Generation Settings ===
    'MAX_TOKENS': 1000,       # ~550-700 words for summary generation
    'TEMPERATURE': 0.1,       # Low = more deterministic, factual (better for summarization)
    'TOP_P': 0.9,             # Nucleus sampling
    'TIMEOUT': 180,           # 3 minutes (8B is faster than 70B)
    
    # === Database ===
    'DB_PATH': 'noun_rank_8B.db',
    
    # === NLP Models ===
    'SPACY_MODEL': 'en_core_web_sm',      # Fast, good for NER
    'EMBEDDING_MODEL': 'all-MiniLM-L6-v2', # Fast, 384-dim embeddings
    'MAX_TEXT_LENGTH': None,              # No truncation (matches calibration engine)
    
    # === Version ===
    'ALGORITHM_VERSION': 'v12.1-llm-eval-8b',
}

WEIGHTS = {
    'H': {'entity': 0.30, 'proper': 0.25, 'nouns': 0.20, 'numeric': 0.15, 'temporal': 0.10},
    'C': {'key': 0.35, 'entity': 0.30, 'numeric': 0.20, 'temporal': 0.15},
    'FINAL': {'H': 0.25, 'C': 0.25, 'F': 0.20, 'E': 0.20, 'S': 0.10},
    'ENTITY': {'PERSON': 1.0, 'ORG': 1.0, 'GPE': 1.0, 'DATE': 0.8, 'MONEY': 0.8}
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


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
    accuracy: float
    scores: ComponentScores
    status: str
    llm_summary: str = ""


# =============================================================================
# LLM Client
# =============================================================================

class LLaMAClient:
    """Client for LLaMA 3.1 8B via vLLM OpenAI-compatible API."""
    
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
    
    def summarize(self, document: str, max_tokens: int = 2048) -> str:
        """Generate summary using LLaMA 3.1 8B."""
        # Truncate very long documents to fit context
        words = document.split()
        if len(words) > 12000:  # ~16k tokens approx
            document = " ".join(words[:12000])
            logger.warning(f"Document truncated to 12000 words")
        
        prompt = self.SUMMARIZE_PROMPT.format(document=document)
        
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
# NLP Models
# =============================================================================

class NLPModels:
    """Singleton for spaCy and sentence-transformers."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._nlp = None
            cls._instance._embedder = None
        return cls._instance
    
    def load(self):
        if self._nlp is None:
            import spacy
            logger.info("Loading spaCy...")
            self._nlp = spacy.load(CONFIG['SPACY_MODEL'])
        
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading sentence-transformers...")
            self._embedder = SentenceTransformer(CONFIG['EMBEDDING_MODEL'])
        
        logger.info("✓ NLP models loaded")
    
    @property
    def nlp(self):
        if self._nlp is None:
            self.load()
        return self._nlp
    
    @property
    def embedder(self):
        if self._embedder is None:
            self.load()
        return self._embedder


# =============================================================================
# Term Extractor & Score Calculators
# =============================================================================

def extract_terms(text: str, nlp) -> ExtractedTerms:
    """Extract terms from text using spaCy."""
    # Optional truncation for very extreme cases (matches calibration engine)
    if CONFIG['MAX_TEXT_LENGTH'] and len(text) > CONFIG['MAX_TEXT_LENGTH']:
        text = text[:CONFIG['MAX_TEXT_LENGTH']]
    
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


def safe_divide(num: float, denom: float, default: float = 0.0) -> float:
    return num / denom if denom else default


def calculate_hallucination(D: ExtractedTerms, S: ExtractedTerms) -> Tuple[float, Dict]:
    H_nouns = safe_divide(len(S.nouns - D.nouns), len(S.nouns), 0.0)
    H_proper = safe_divide(len(S.proper - D.proper), len(S.proper), 0.0)
    H_entity = safe_divide(len(S.entities - D.entities), len(S.entities), 0.0)
    H_temporal = safe_divide(len(S.temporal - D.temporal), len(S.temporal), 0.0)
    H_numeric = safe_divide(len(S.numeric - D.numeric), len(S.numeric), 0.0)
    
    w = WEIGHTS['H']
    H = (H_entity * w['entity'] + H_proper * w['proper'] + H_nouns * w['nouns'] +
         H_numeric * w['numeric'] + H_temporal * w['temporal'])
    
    return H, {'H_entity': H_entity, 'H_proper': H_proper, 'H_nouns': H_nouns,
               'H_temporal': H_temporal, 'H_numeric': H_numeric}


def calculate_coverage(D: ExtractedTerms, S: ExtractedTerms) -> Tuple[float, Dict]:
    D_key = D.get_key_terms(min_freq=2)
    
    C_key = safe_divide(len(S.all_terms & D_key), len(D_key), 1.0)
    C_entity = safe_divide(len(S.entities & D.entities), len(D.entities), 1.0)
    C_temporal = safe_divide(len(S.temporal & D.temporal), len(D.temporal), 1.0)
    C_numeric = safe_divide(len(S.numeric & D.numeric), len(D.numeric), 1.0)
    
    w = WEIGHTS['C']
    C = (C_key * w['key'] + C_entity * w['entity'] +
         C_numeric * w['numeric'] + C_temporal * w['temporal'])
    
    return C, {'C_key': C_key, 'C_entity': C_entity,
               'C_temporal': C_temporal, 'C_numeric': C_numeric}


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


def get_embedding(text: str, embedder, max_words: int = 512) -> np.ndarray:
    """Get embedding, chunking if necessary (matches calibration engine)."""
    words = text.split()
    
    if len(words) <= max_words:
        result = embedder.encode(text)
        return result[0] if len(result.shape) > 1 else result
    
    # Chunk long texts and batch encode
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i + max_words])
        chunks.append(chunk)
    
    # Batch encode all chunks at once
    embeddings = embedder.encode(chunks)
    return np.mean(embeddings, axis=0)


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calculate cosine similarity."""
    dot = np.dot(emb1, emb2)
    norm1, norm2 = np.linalg.norm(emb1), np.linalg.norm(emb2)
    return float(dot / (norm1 * norm2)) if norm1 and norm2 else 0.0


def calculate_semantic(doc_text: str, summary_text: str, embedder) -> float:
    """Calculate semantic similarity (matches calibration engine)."""
    emb_D = get_embedding(doc_text, embedder)
    emb_S = get_embedding(summary_text, embedder)
    return cosine_similarity(emb_D, emb_S)


def compute_noun_rank(doc_text: str, summary_text: str, nlp, embedder) -> ComponentScores:
    """Compute full Noun Rank score."""
    D = extract_terms(doc_text, nlp)
    S = extract_terms(summary_text, nlp)
    
    H, H_comp = calculate_hallucination(D, S)
    C, C_comp = calculate_coverage(D, S)
    F = calculate_frequency(D, S)
    E = calculate_entity(D, S)
    S_score = calculate_semantic(doc_text, summary_text, embedder)
    
    w = WEIGHTS['FINAL']
    final_score = ((1 - H) * w['H'] + C * w['C'] + F * w['F'] + E * w['E'] + S_score * w['S'])
    
    return ComponentScores(
        H=H, C=C, F=F, E=E, S=S_score,
        H_entity=H_comp['H_entity'], H_proper=H_comp['H_proper'],
        H_nouns=H_comp['H_nouns'], H_temporal=H_comp['H_temporal'],
        H_numeric=H_comp['H_numeric'],
        C_key=C_comp['C_key'], C_entity=C_comp['C_entity'],
        C_temporal=C_comp['C_temporal'], C_numeric=C_comp['C_numeric'],
        final_score=final_score
    )


# =============================================================================
# Database Manager
# =============================================================================

class DatabaseManager:
    """Read baselines and save LLM evaluations."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._ensure_llm_table()
    
    def _ensure_llm_table(self):
        """Ensure LLM model is registered."""
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR IGNORE INTO llm_models (model_name, model_version, provider)
            VALUES (?, ?, ?)
        ''', (CONFIG['LLM_MODEL'], 'INT8', 'vast.ai'))
        self.conn.commit()
        
        cursor.execute('SELECT model_id FROM llm_models WHERE model_name = ?', 
                      (CONFIG['LLM_MODEL'],))
        self.model_id = cursor.fetchone()['model_id']
    
    def get_baseline(self, doc_id: str) -> Optional[float]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT final_score FROM evaluations
            WHERE doc_id = ? AND is_baseline = 1
            ORDER BY run_date DESC LIMIT 1
        ''', (doc_id,))
        row = cursor.fetchone()
        return row['final_score'] if row else None
    
    def get_all_baselines(self) -> Dict[str, float]:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT doc_id, final_score FROM evaluations
            WHERE is_baseline = 1
        ''')
        return {row['doc_id']: row['final_score'] for row in cursor.fetchall()}
    
    def save_llm_evaluation(self, doc_id: str, scores: ComponentScores, 
                           accuracy: float, processing_time_ms: float):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO evaluations (
                doc_id, model_id, is_baseline, final_score,
                H, C, F, E, S,
                H_entity, H_proper, H_nouns, H_temporal, H_numeric,
                C_key, C_entity, C_temporal, C_numeric,
                accuracy, algorithm_version, processing_time_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            doc_id, self.model_id, False, scores.final_score,
            scores.H, scores.C, scores.F, scores.E, scores.S,
            scores.H_entity, scores.H_proper, scores.H_nouns,
            scores.H_temporal, scores.H_numeric,
            scores.C_key, scores.C_entity, scores.C_temporal, scores.C_numeric,
            accuracy, CONFIG['ALGORITHM_VERSION'], processing_time_ms
        ))
        self.conn.commit()


# =============================================================================
# LLM Evaluator
# =============================================================================

class LLMEvaluator:
    """Main evaluator class."""
    
    def __init__(self, db_path: str, llm_host: str, llm_api_key: str = None):
        self.db = DatabaseManager(db_path)
        self.llm = LLaMAClient(llm_host, CONFIG['LLM_MODEL'], llm_api_key)
        self.models = NLPModels()
        self.models.load()
        self.baselines = self.db.get_all_baselines()
        logger.info(f"Loaded {len(self.baselines)} baseline scores from database")
    
    def load_govreport(self, split: str = "test", limit: int = None) -> List[Dict]:
        """Load GovReport dataset."""
        from datasets import load_dataset
        
        logger.info(f"Loading GovReport (split={split})...")
        dataset = load_dataset("ccdv/govreport-summarization", split=split)
        
        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))
        
        records = []
        for idx, item in enumerate(dataset):
            doc_id = f"doc_{idx:05d}"
            if doc_id in self.baselines:  # Only include docs with baselines
                records.append({
                    'doc_id': doc_id,
                    'document': item['report'],
                    'expert_summary': item['summary']
                })
        
        logger.info(f"Loaded {len(records)} documents with baselines")
        return records
    
    def evaluate_document(self, record: Dict) -> EvalResult:
        """Evaluate single document."""
        doc_id = record['doc_id']
        doc_text = record['document']
        baseline = self.baselines.get(doc_id, 0.4264)  # Use mean if missing
        
        start_time = time.time()
        
        # Generate LLM summary
        try:
            llm_summary = self.llm.summarize(doc_text)
        except Exception as e:
            logger.error(f"{doc_id}: LLM failed - {e}")
            return EvalResult(
                doc_id=doc_id, baseline_score=baseline, llm_score=0.0,
                accuracy=0.0, scores=ComponentScores(), status="LLM_ERROR"
            )
        
        # Score LLM summary
        scores = compute_noun_rank(
            doc_text, llm_summary, 
            self.models.nlp, self.models.embedder
        )
        
        # Calculate accuracy
        accuracy = (scores.final_score / baseline) * 100 if baseline > 0 else 0
        
        # Determine status
        if scores.H >= 0.35:
            status = "REJECT_HALLUCINATION"
        elif accuracy >= 95:
            status = "EXCELLENT"
        elif accuracy >= 85:
            status = "GOOD"
        elif accuracy >= 70:
            status = "ACCEPTABLE"
        else:
            status = "REJECT_LOW_ACCURACY"
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Save to database
        self.db.save_llm_evaluation(doc_id, scores, accuracy, elapsed_ms)
        
        return EvalResult(
            doc_id=doc_id,
            baseline_score=baseline,
            llm_score=scores.final_score,
            accuracy=accuracy,
            scores=scores,
            status=status,
            llm_summary=llm_summary
        )
    
    def evaluate(self, split: str = "test", limit: int = None) -> Dict:
        """Run full evaluation."""
        records = self.load_govreport(split=split, limit=limit)
        
        print("\n" + "=" * 70)
        print("LLM EVALUATION STARTED")
        print("=" * 70)
        print(f"Model: {CONFIG['LLM_MODEL']}")
        print(f"Documents: {len(records)}")
        print("=" * 70 + "\n")
        
        results = []
        start_time = time.time()
        
        for i, record in enumerate(records):
            result = self.evaluate_document(record)
            results.append(result)
            
            # Progress log
            status_icon = {
                'EXCELLENT': '★',
                'GOOD': '✓',
                'ACCEPTABLE': '○',
                'REJECT_HALLUCINATION': '✗H',
                'REJECT_LOW_ACCURACY': '✗L',
                'LLM_ERROR': '✗E'
            }.get(result.status, '?')
            
            logger.info(
                f"[{i+1}/{len(records)}] {result.doc_id}: "
                f"Score={result.llm_score:.4f} "
                f"Baseline={result.baseline_score:.4f} "
                f"Accuracy={result.accuracy:.1f}% "
                f"[{status_icon}]"
            )
        
        # Calculate statistics
        accuracies = [r.accuracy for r in results if r.status != 'LLM_ERROR']
        scores = [r.llm_score for r in results if r.status != 'LLM_ERROR']
        
        stats = {
            'total_docs': len(records),
            'evaluated': len(accuracies),
            'mean_accuracy': np.mean(accuracies) if accuracies else 0,
            'std_accuracy': np.std(accuracies) if accuracies else 0,
            'min_accuracy': np.min(accuracies) if accuracies else 0,
            'max_accuracy': np.max(accuracies) if accuracies else 0,
            'mean_score': np.mean(scores) if scores else 0,
            'excellent': sum(1 for r in results if r.status == 'EXCELLENT'),
            'good': sum(1 for r in results if r.status == 'GOOD'),
            'acceptable': sum(1 for r in results if r.status == 'ACCEPTABLE'),
            'rejected_hallucination': sum(1 for r in results if r.status == 'REJECT_HALLUCINATION'),
            'rejected_low': sum(1 for r in results if r.status == 'REJECT_LOW_ACCURACY'),
            'errors': sum(1 for r in results if r.status == 'LLM_ERROR'),
            'total_time': time.time() - start_time
        }
        
        self._print_summary(stats)
        return stats
    
    def _print_summary(self, stats: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 70)
        print("LLM EVALUATION COMPLETE")
        print("=" * 70)
        print(f"{'Model':<35} {CONFIG['LLM_MODEL']}")
        print("-" * 70)
        print(f"{'Documents Evaluated':<35} {stats['evaluated']:>15}")
        print(f"{'Mean Accuracy':<35} {stats['mean_accuracy']:>14.1f}%")
        print(f"{'Std Deviation':<35} {stats['std_accuracy']:>14.1f}%")
        print(f"{'Min Accuracy':<35} {stats['min_accuracy']:>14.1f}%")
        print(f"{'Max Accuracy':<35} {stats['max_accuracy']:>14.1f}%")
        print(f"{'Mean LLM Score':<35} {stats['mean_score']:>15.4f}")
        print("-" * 70)
        print("STATUS BREAKDOWN:")
        print(f"  ★ Excellent (≥95%)             {stats['excellent']:>10}")
        print(f"  ✓ Good (85-94%)                {stats['good']:>10}")
        print(f"  ○ Acceptable (70-84%)          {stats['acceptable']:>10}")
        print(f"  ✗ Rejected (Hallucination)     {stats['rejected_hallucination']:>10}")
        print(f"  ✗ Rejected (Low Accuracy)      {stats['rejected_low']:>10}")
        print(f"  ✗ Errors                       {stats['errors']:>10}")
        print("-" * 70)
        print(f"{'Total Time (seconds)':<35} {stats['total_time']:>15.1f}")
        print(f"{'Avg Time per Doc (seconds)':<35} {stats['total_time']/max(stats['evaluated'],1):>15.1f}")
        print("=" * 70)
        
        # Pass/Fail
        if stats['mean_accuracy'] >= 85:
            print("\n✓ PASS: Mean accuracy ≥ 85%")
        else:
            print("\n✗ FAIL: Mean accuracy < 85%")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Noun Rank LLM Evaluator")
    parser.add_argument("--host", type=str, default="http://localhost:8000",
                       help="vLLM host URL")
    parser.add_argument("--api-key", type=str, default=None,
                       help="API key for vLLM")
    parser.add_argument("--db", type=str, default="noun_rank.db",
                       help="Database path with baselines")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit documents to evaluate")
    parser.add_argument("--model", type=str, default=None,
                       help="Override LLM model name")
    
    args = parser.parse_args()
    
    if args.model:
        CONFIG['LLM_MODEL'] = args.model
    
    evaluator = LLMEvaluator(
        db_path=args.db,
        llm_host=args.host,
        llm_api_key=args.api_key
    )
    
    stats = evaluator.evaluate(split=args.split, limit=args.limit)
    return stats


if __name__ == "__main__":
    main()
