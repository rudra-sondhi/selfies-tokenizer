# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2025-11-26

### Added
- `load_map` parameter to `load_encoded_dataset()` function
  - When `False` (default), index_mapping is not loaded (faster, uses less memory)
  - When `True`, index_mapping is loaded and returned
  - Improves performance for large datasets where index mapping is not needed

### Changed
- `load_encoded_dataset()` now returns `index_mapping=None` by default
- Index mapping file is only checked/loaded when `load_map=True`

## [0.1.0] - 2025-11-26

### Added
- Initial release of SELFIES Tokenizer
- Fast regex-based tokenization (~109k molecules/sec)
- ML-ready batch processing with padding and truncation
- Hard-coded special token indices: `<pad>=0`, `<unk>=1`, `<start>=2`, `<end>=3`
- Smart `suggest_len()` method for interactive max_len selection
- Complete dataset packaging with `save_encoded_dataset()` and `load_encoded_dataset()`
- Index mapping (array index → SELFIES string)
- Metadata management (timestamp, settings, statistics, coverage)
- Progress bar support with tqdm
- Perfect round-trip encoding/decoding
- Comprehensive documentation and examples
- MIT License

### Features
- `SELFIESTokenizer` - Main tokenizer class
- `tokenize()` - Quick tokenization function
- `save_encoded_dataset()` - Save datasets with full metadata
- `load_encoded_dataset()` - Load datasets with metadata
- Support for custom special tokens
- Vocabulary save/load functionality
- Batch processing for ML efficiency

[0.1.0]: https://github.com/yourusername/selfies-tokenizer/releases/tag/v0.1.0
