"""
Utility functions for SELFIES tokenizer dataset management.

Provides tools for saving and loading encoded datasets with full metadata.
"""

import json
import os
from datetime import datetime
from typing import List, Union, Optional, Dict, Any
import numpy as np


def save_encoded_dataset(
    selfies_data: Union[str, List[str]],
    tokenizer: 'SELFIESTokenizer',
    save_dir: str,
    max_len: Optional[int] = None,
    padding: bool = True,
    truncation: bool = True,
    add_special_tokens: bool = True,
    show_progress: bool = False,
    vocab_filename: str = 'vocab.json',
    data_filename: str = 'encoded_data.npy',
    index_filename: str = 'index_mapping.json',
    metadata_filename: str = 'metadata.json'
) -> Dict[str, str]:
    """
    Save encoded SELFIES dataset with vocabulary, index mapping, and metadata.

    Creates a complete dataset package with:
    - Tokenizer vocabulary
    - Encoded data as numpy array
    - Index mapping (array index -> SELFIES string)
    - Metadata (timestamp, settings, statistics)

    Args:
        selfies_data: Single SELFIES string or list of SELFIES strings to encode
        tokenizer: Fitted SELFIESTokenizer instance with vocabulary
        save_dir: Directory path to save dataset (will be created if doesn't exist)
        max_len: Maximum sequence length for encoding
        padding: Pad sequences to max_len
        truncation: Truncate sequences longer than max_len
        add_special_tokens: Add <start> and <end> tokens
        show_progress: Show progress bar during encoding
        vocab_filename: Name for vocabulary file (default: 'vocab.json')
        data_filename: Name for encoded data file (default: 'encoded_data.npy')
        index_filename: Name for index mapping file (default: 'index_mapping.json')
        metadata_filename: Name for metadata file (default: 'metadata.json')

    Returns:
        Dictionary with paths to all saved files:
        {
            'save_dir': str,
            'vocab_path': str,
            'data_path': str,
            'index_path': str,
            'metadata_path': str
        }

    Examples:
        >>> from selfies_tokenizer import SELFIESTokenizer, save_encoded_dataset
        >>>
        >>> # Prepare tokenizer
        >>> tokenizer = SELFIESTokenizer()
        >>> tokenizer.fit(train_data)
        >>>
        >>> # Save dataset
        >>> paths = save_encoded_dataset(
        ...     selfies_data=train_data,
        ...     tokenizer=tokenizer,
        ...     save_dir='./datasets/my_dataset',
        ...     max_len=50,
        ...     show_progress=True
        ... )
        >>>
        >>> print(f"Dataset saved to: {paths['save_dir']}")

    Directory Structure:
        save_dir/
            ├── vocab.json              # Tokenizer vocabulary
            ├── encoded_data.npy        # Encoded sequences (shape: [N, max_len])
            ├── index_mapping.json      # Maps array index -> SELFIES string
            └── metadata.json           # Dataset metadata and settings
    """
    # Convert single string to list
    if isinstance(selfies_data, str):
        selfies_data = [selfies_data]

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Define file paths
    vocab_path = os.path.join(save_dir, vocab_filename)
    data_path = os.path.join(save_dir, data_filename)
    index_path = os.path.join(save_dir, index_filename)
    metadata_path = os.path.join(save_dir, metadata_filename)

    # 1. Save vocabulary
    tokenizer.save_vocab(vocab_path)

    # 2. Encode data
    print(f"\nEncoding {len(selfies_data)} sequences...")
    encoded_data = tokenizer.encode(
        selfies_data,
        max_len=max_len,
        padding=padding,
        truncation=truncation,
        add_special_tokens=add_special_tokens,
        show_progress=show_progress
    )

    # Convert to numpy array and save
    encoded_array = np.array(encoded_data, dtype=np.int32)
    np.save(data_path, encoded_array)
    print(f"✓ Saved encoded data: {data_path}")
    print(f"  Shape: {encoded_array.shape}")
    print(f"  Dtype: {encoded_array.dtype}")

    # 3. Create index mapping (array index -> SELFIES)
    # Maps each array index to its SELFIES string
    print(f"\nCreating index mapping...")
    index_mapping = {str(idx): selfies for idx, selfies in enumerate(selfies_data)}

    with open(index_path, 'w') as f:
        json.dump(index_mapping, f, indent=2)
    print(f"✓ Saved index mapping: {index_path}")
    print(f"  Total mappings: {len(index_mapping)}")

    # 4. Calculate statistics
    sequence_lengths = [len(tokenizer.tokenize(s)) for s in selfies_data]
    if add_special_tokens:
        sequence_lengths = [l + 2 for l in sequence_lengths]  # Account for <start> and <end>

    # Count truncated sequences
    if max_len is not None and truncation:
        num_truncated = sum(1 for l in sequence_lengths if l > max_len)
    else:
        num_truncated = 0

    # 5. Create metadata
    metadata = {
        'created_at': datetime.now().isoformat(),
        'dataset': {
            'num_sequences': len(selfies_data),
            'sequence_length': {
                'min': int(min(sequence_lengths)) if sequence_lengths else 0,
                'max': int(max(sequence_lengths)) if sequence_lengths else 0,
                'mean': float(np.mean(sequence_lengths)) if sequence_lengths else 0.0,
                'median': float(np.median(sequence_lengths)) if sequence_lengths else 0.0
            },
            'num_truncated': num_truncated,
            'coverage': 100.0 * (len(selfies_data) - num_truncated) / len(selfies_data) if selfies_data else 0.0
        },
        'encoding_settings': {
            'max_len': max_len,
            'padding': padding,
            'truncation': truncation,
            'add_special_tokens': add_special_tokens
        },
        'tokenizer': {
            'vocab_size': tokenizer.vocab_size,
            'special_tokens': tokenizer.special_tokens,
            'method': tokenizer.method,
            'tokenizer_max_len': tokenizer.max_len
        },
        'files': {
            'vocab': vocab_filename,
            'data': data_filename,
            'index_mapping': index_filename,
            'metadata': metadata_filename
        },
        'array_info': {
            'shape': list(encoded_array.shape),
            'dtype': str(encoded_array.dtype)
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {metadata_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("DATASET SAVED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nLocation: {save_dir}")
    print(f"\nFiles created:")
    print(f"  ✓ {vocab_filename:30s} - Tokenizer vocabulary")
    print(f"  ✓ {data_filename:30s} - Encoded data ({encoded_array.shape})")
    print(f"  ✓ {index_filename:30s} - Index mapping ({len(index_mapping)} entries)")
    print(f"  ✓ {metadata_filename:30s} - Dataset metadata")
    print(f"\nDataset Statistics:")
    print(f"  Sequences: {len(selfies_data)}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  Max length: {max_len}")
    print(f"  Truncated: {num_truncated} ({100*num_truncated/len(selfies_data):.1f}%)")
    print(f"  Coverage: {metadata['dataset']['coverage']:.1f}%")
    print("=" * 70)

    # Return all paths
    return {
        'save_dir': save_dir,
        'vocab_path': vocab_path,
        'data_path': data_path,
        'index_path': index_path,
        'metadata_path': metadata_path
    }


def load_encoded_dataset(
    load_dir: str,
    vocab_filename: str = 'vocab.json',
    data_filename: str = 'encoded_data.npy',
    index_filename: str = 'index_mapping.json',
    metadata_filename: str = 'metadata.json',
    load_tokenizer: bool = True
) -> Dict[str, Any]:
    """
    Load encoded SELFIES dataset with all metadata.

    Args:
        load_dir: Directory containing the dataset
        vocab_filename: Name of vocabulary file (default: 'vocab.json')
        data_filename: Name of encoded data file (default: 'encoded_data.npy')
        index_filename: Name of index mapping file (default: 'index_mapping.json')
        metadata_filename: Name of metadata file (default: 'metadata.json')
        load_tokenizer: If True, load and return tokenizer instance

    Returns:
        Dictionary containing:
        {
            'encoded_data': np.ndarray,          # Encoded sequences
            'index_mapping': dict,                # index -> SELFIES string
            'metadata': dict,                     # Dataset metadata
            'tokenizer': SELFIESTokenizer or None # Tokenizer instance (if load_tokenizer=True)
        }

    Examples:
        >>> from selfies_tokenizer import load_encoded_dataset
        >>>
        >>> # Load dataset
        >>> dataset = load_encoded_dataset('./datasets/my_dataset')
        >>>
        >>> # Access components
        >>> X = dataset['encoded_data']
        >>> tokenizer = dataset['tokenizer']
        >>> metadata = dataset['metadata']
    """
    # Define file paths
    vocab_path = os.path.join(load_dir, vocab_filename)
    data_path = os.path.join(load_dir, data_filename)
    index_path = os.path.join(load_dir, index_filename)
    metadata_path = os.path.join(load_dir, metadata_filename)

    # Verify all files exist
    for path, name in [
        (vocab_path, 'vocabulary'),
        (data_path, 'encoded data'),
        (index_path, 'index mapping'),
        (metadata_path, 'metadata')
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} file: {path}")

    print(f"\nLoading dataset from: {load_dir}")

    # Load encoded data
    encoded_data = np.load(data_path)
    print(f"✓ Loaded encoded data: {encoded_data.shape}")

    # Load index mapping
    with open(index_path, 'r') as f:
        index_mapping = json.load(f)
    print(f"✓ Loaded index mapping: {len(index_mapping)} entries")

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✓ Loaded metadata")

    # Load tokenizer if requested
    tokenizer = None
    if load_tokenizer:
        from selfies_tokenizer import SELFIESTokenizer
        tokenizer = SELFIESTokenizer(vocab_path=vocab_path)
        print(f"✓ Loaded tokenizer: vocab_size={tokenizer.vocab_size}")

    print(f"\n{'=' * 70}")
    print("DATASET LOADED SUCCESSFULLY")
    print(f"{'=' * 70}")
    print(f"\nDataset info:")
    print(f"  Created: {metadata['created_at']}")
    print(f"  Sequences: {metadata['dataset']['num_sequences']}")
    print(f"  Shape: {encoded_data.shape}")
    print(f"  Vocab size: {metadata['tokenizer']['vocab_size']}")
    print(f"  Max length: {metadata['encoding_settings']['max_len']}")
    print(f"  Coverage: {metadata['dataset']['coverage']:.1f}%")
    print(f"{'=' * 70}")

    return {
        'encoded_data': encoded_data,
        'index_mapping': index_mapping,
        'metadata': metadata,
        'tokenizer': tokenizer
    }
