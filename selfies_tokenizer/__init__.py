"""
SELFIES Tokenizer - Fast tokenization for molecular SELFIES strings.

A super fast tokenizer for SELFIES (SELF-referencIng Embedded Strings)
molecular representations with ML-ready batch processing and vocabulary management.

Main classes:
    SELFIESTokenizer: Main tokenizer class for SELFIES strings

Utility functions:
    tokenize: Quick tokenization without creating a tokenizer instance
    save_encoded_dataset: Save encoded dataset with full metadata
    load_encoded_dataset: Load encoded dataset with metadata

Example:
    >>> from selfies_tokenizer import SELFIESTokenizer
    >>> tokenizer = SELFIESTokenizer()
    >>> tokenizer.fit(['[C][=O]', '[N][C]'])
    >>> encoded = tokenizer.encode('[C][=O]', max_len=10)
"""

__version__ = "0.1.0"
__author__ = "Rudra Sondhi"
__email__ = "rudra.sondhi2@gmail.com"

from selfies_tokenizer.tokenizer import SELFIESTokenizer, tokenize
from selfies_tokenizer.utils import save_encoded_dataset, load_encoded_dataset

__all__ = [
    'SELFIESTokenizer',
    'tokenize',
    'save_encoded_dataset',
    'load_encoded_dataset',
]
