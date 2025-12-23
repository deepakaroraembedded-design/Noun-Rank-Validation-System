   
import fitz  # PyMuPDF
import spacy
from collections import Counter

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def read_pdf(file_path):
    """Read text from PDF file"""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def extract_nouns(text):
    """Extract all noun data from text"""
    doc = nlp(text)
    
    # Common nouns (lemmatized)
    common_nouns = [token.lemma_.lower() for token in doc if token.pos_ == "NOUN"]
    
    # Proper nouns (original case)
    proper_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
    
    # Named entities
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # Frequency count
    noun_freq = Counter(common_nouns)
    proper_freq = Counter(proper_nouns)
    
    # Key nouns (appear 2+ times)
    key_nouns = {noun: count for noun, count in noun_freq.items() if count >= 2}
    
    return {
        "common_nouns": list(set(common_nouns)),
        "proper_nouns": list(set(proper_nouns)),
        "entities": list(set(entities)),
        "noun_frequency": noun_freq,
        "proper_frequency": proper_freq,
        "key_nouns": key_nouns,
        "total_common": len(common_nouns),
        "total_proper": len(proper_nouns),
        "unique_common": len(set(common_nouns)),
        "unique_proper": len(set(proper_nouns))
    }


"""
def print_results(results):
 
    
    print("=" * 60)
    print("PDF NOUN EXTRACTION RESULTS")
    print("=" * 60)
    
    # Stats
    print(f"\n📊 STATISTICS")
    print(f"   Total common nouns: {results['total_common']}")
    print(f"   Unique common nouns: {results['unique_common']}")
    print(f"   Total proper nouns: {results['total_proper']}")
    print(f"   Unique proper nouns: {results['unique_proper']}")
    
    # Key nouns
    print(f"\n🔑 KEY NOUNS (appear 2+ times)")
    print("-" * 40)
    sorted_key = sorted(results['key_nouns'].items(), key=lambda x: x[1], reverse=True)
    for noun, count in sorted_key[:20]:
        if(count > 5):  # Top 20
            print(f"   {noun:20} : {count}")
    
    # Proper nouns
    print(f"\n📌 PROPER NOUNS")
    print("-" * 40)
    sorted_proper = sorted(results['proper_frequency'].items(), key=lambda x: x[1], reverse=True)
    for noun, count in sorted_proper[:15]:  # Top 15
        print(f"   {noun:20} : {count}")
    
    # Named entities
    print(f"\n🏷️ NAMED ENTITIES")
    print("-" * 40)
    for entity, label in sorted(results['entities'], key=lambda x: x[1])[:20]:
        print(f"   {entity:25} → {label}")
    
    # All common nouns
    print(f"\n📝 ALL UNIQUE COMMON NOUNS")
    print("-" * 40)
    print(", ".join(sorted(results['common_nouns'])))
"""

def print_results(results):
 
    
    print("=" * 60)
    print("PDF NOUN EXTRACTION RESULTS")
    print("=" * 60)
    
    # Stats
 #   print(f"\n📊 STATISTICS")
 #   print(f"   Total common nouns: {results['total_common']}")
 #   print(f"   Unique common nouns: {results['unique_common']}")
 #   print(f"   Total proper nouns: {results['total_proper']}")
 #   print(f"   Unique proper nouns: {results['unique_proper']}")
    
    # Key nouns
 #   print(f"\n🔑 KEY NOUNS (appear 2+ times)")
    print("-" * 40)

 
#   sorted_key = sorted(results['key_nouns'].items(), key=lambda x: x[1], reverse=True)
#   for noun, count in sorted_key:
#       print(f"   {noun:20} : {count}")

    # Proper nouns
 #  print(f"\n📌 PROPER NOUNS")
 #  print("-" * 40)
 #  sorted_proper = sorted(results['proper_frequency'].items(), key=lambda x: x[1], reverse=True)
 #  for noun, count in sorted_proper:  # Top 15
 #      print(f"   {noun:20} : {count}")

    
    # Named entities
 #  print(f"\n🏷️ NAMED ENTITIES")
 #  print("-" * 40)
 #  for entity, label in sorted(results['entities'], key=lambda x: x[1])[:20]:
 #      print(f"   {entity:25} → {label}")
    
    # All common nouns
    print(f"\n📝 ALL UNIQUE COMMON NOUNS")
    print("-" * 40)
    print(", ".join(sorted(results['common_nouns'])))
    return

# PDF file path
pdf_path = r"C:\Users\arora\OneDrive\Documents\GitHub\Noun-Rank-Validation-System\OCI Technical Interview Prep - Technical.pdf"
    
print(f"Reading PDF: {pdf_path}")

    # Read PDF
text = read_pdf(pdf_path)
print(f"Extracted {len(text)} characters from PDF")
    
    # Extract nouns
results = extract_nouns(text)
    
#print(results)
print_results(results)
    #print_results(results)
    
   

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter

#nltk.download('punkt')
#nltk.download('stopwords')

def summarize_nltk(text, num_sentences=10):
    sentences = sent_tokenize(text)
    words = word_tokenize(text.lower())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w.isalnum() and w not in stop_words]
    
    # Word frequency
    word_freq = Counter(words)
    
    # Score sentences
    sentence_scores = {}
    for sentence in sentences:
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                sentence_scores[sentence] = sentence_scores.get(sentence, 0) + word_freq[word]
    
    # Top sentences
    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    
    # Keep original order
    summary = [s for s in sentences if s in top_sentences]
    
    return " ".join(summary)

# Usage
#text = """Your long document text here..."""
summary = summarize_nltk(text, num_sentences=3)
#rint(summary)
results = extract_nouns(summary)
#sleep(15)  
print_results(results)
