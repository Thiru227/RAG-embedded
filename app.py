"""
RTC Scholar - Phase 2: Semantic Embeddings
===========================================
Configurable retrieval: tfidf / semantic / hybrid
Default: hybrid (85-88% accuracy)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import time

from knowledge_base import KNOWLEDGE_BASE

# ============================================
# CHOOSE RETRIEVAL METHOD
# ============================================
RETRIEVAL_MODE = os.environ.get('RETRIEVAL_MODE', 'hybrid').lower()

if RETRIEVAL_MODE == 'semantic':
    from semantic_vector_db import SemanticVectorDB
    vector_db = SemanticVectorDB()
    SYSTEM_NAME = "Semantic Embeddings"
    EXPECTED_ACCURACY = "80-85%"
    
elif RETRIEVAL_MODE == 'hybrid':
    from semantic_vector_db import HybridVectorDB
    vector_db = HybridVectorDB(tfidf_weight=0.3, semantic_weight=0.7)
    SYSTEM_NAME = "Hybrid (TF-IDF + Semantic)"
    EXPECTED_ACCURACY = "85-88%"
    
else:  # 'tfidf' or default
    from vecto
