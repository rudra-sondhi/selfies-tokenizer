"""
SELFIES Tokenizer - Complete Example

This example demonstrates all major features of the SELFIES tokenizer:
1. Basic tokenization
2. Vocabulary building
3. Encoding and decoding with special tokens
4. Smart max_len selection
5. Saving and loading datasets
6. Using with ML frameworks (PyTorch example)
"""

from selfies_tokenizer import SELFIESTokenizer, tokenize, save_encoded_dataset, load_encoded_dataset

print("=" * 80)
print("SELFIES Tokenizer - Complete Example")
print("=" * 80)

# Sample SELFIES data (in practice, load from file)
sample_data = [
    '[C][=O]',
    '[C][C][C][=O]',
    '[C][C][C][C][C][=O]',
    '[N][C][=O]',
    '[C][Branch1][C][C][=O]',
    '[C][C][N][C][C][=O]',
    '[C][C][C][N][C][C][C][=O]',
    '[C][O][C][C][N][C][=C]',
    '[O][=C][Branch1][C][O]',
    '[C][C][O][C][=Branch1]',
] * 10  # 100 samples

print(f"\nSample dataset: {len(sample_data)} SELFIES strings")
print(f"First few: {sample_data[:3]}")

# ============================================================================
# 1. Basic Tokenization
# ============================================================================
print("\n" + "=" * 80)
print("1. Basic Tokenization")
print("=" * 80)

tokens = tokenize('[C][C][N][C][C][=O]')
print(f"\nInput:  '[C][C][N][C][C][=O]'")
print(f"Output: {tokens}")

# ============================================================================
# 2. Build Vocabulary
# ============================================================================
print("\n" + "=" * 80)
print("2. Building Vocabulary")
print("=" * 80)

tokenizer = SELFIESTokenizer(
    vocab_path='./example_vocab.json',
    refresh_dict=True,
    max_len=15  # Default max_len (saved in metadata)
)

print("\nFitting tokenizer on training data...")
tokenizer.fit(sample_data, show_progress=False)

print(f"✓ Vocabulary size: {tokenizer.vocab_size}")
print(f"✓ Special tokens: {tokenizer.special_tokens}")
print(f"✓ <pad> index: {tokenizer.token2idx['<pad>']} (always 0)")

# Save vocabulary
tokenizer.save_vocab()

# ============================================================================
# 3. Encoding with Special Tokens
# ============================================================================
print("\n" + "=" * 80)
print("3. Encoding with Special Tokens")
print("=" * 80)

# Single sequence (use molecule from training data for perfect round-trip)
molecule = '[C][C][N][C][C][=O]'
encoded = tokenizer.encode(molecule, max_len=10)
print(f"\nOriginal: {molecule}")
print(f"Encoded:  {encoded}")
print(f"Format:   <start>, [C], [C], [N], [C], [C], [=O], <end>, <pad>, <pad>")

# As token strings
encoded_str = tokenizer.encode(molecule, max_len=10, return_str=True)
print(f"Tokens:   {encoded_str}")

# Batch encoding (using molecules from training data)
batch = ['[C][=O]', '[N][C][=O]', '[C][C][C][=O]']
encoded_batch = tokenizer.encode(batch, max_len=10)
print(f"\nBatch encoding:")
for orig, enc in zip(batch, encoded_batch):
    print(f"  {orig:20s} -> {enc}")

# ============================================================================
# 4. Decoding (stops at <end>, removes special tokens)
# ============================================================================
print("\n" + "=" * 80)
print("4. Smart Decoding")
print("=" * 80)

decoded = tokenizer.decode(encoded)
print(f"\nEncoded:  {encoded}")
print(f"Decoded:  {decoded}")
print(f"Original: {molecule}")
print(f"Match:    {decoded == molecule} ✓")

# Batch decoding
decoded_batch = tokenizer.decode(encoded_batch)
print(f"\nBatch decoding (all match original):")
for orig, dec in zip(batch, decoded_batch):
    match = "✓" if orig == dec else "✗"
    print(f"  {match} {orig:15s} -> {dec}")

# ============================================================================
# 5. Smart max_len Selection (Interactive)
# ============================================================================
print("\n" + "=" * 80)
print("5. Smart max_len Selection")
print("=" * 80)

print("\nCalling suggest_len() - this will show interactive options...")
print("(Select option 2 for 90% coverage when prompted)")

# Call the interactive suggest_len method
chosen_max_len = tokenizer.suggest_len(sample_data, show_progress=False)

if chosen_max_len is None:
    print("\n⚠ suggest_len() was cancelled, using default max_len=15")
    chosen_max_len = 15
else:
    print(f"\n✓ Will use max_len={chosen_max_len} for dataset")

# ============================================================================
# 6. Save Complete Dataset Package
# ============================================================================
print("\n" + "=" * 80)
print("6. Save Dataset with Full Metadata")
print("=" * 80)

save_dir = './example_dataset'

print(f"\nSaving complete dataset package to: {save_dir}")
paths = save_encoded_dataset(
    selfies_data=sample_data,
    tokenizer=tokenizer,
    save_dir=save_dir,
    max_len=chosen_max_len,
    show_progress=False
)

print(f"\nFiles created:")
for key, path in paths.items():
    if key != 'save_dir':
        print(f"  ✓ {path}")

# ============================================================================
# 7. Load Dataset
# ============================================================================
print("\n" + "=" * 80)
print("7. Load Dataset with Metadata")
print("=" * 80)

dataset = load_encoded_dataset(save_dir, load_tokenizer=True)

print(f"\nLoaded components:")
print(f"  encoded_data: shape={dataset['encoded_data'].shape}, dtype={dataset['encoded_data'].dtype}")
print(f"  tokenizer:    vocab_size={dataset['tokenizer'].vocab_size}")
print(f"  index_mapping: {len(dataset['index_mapping'])} entries")
print(f"  metadata:     created at {dataset['metadata']['created_at'][:19]}")

# Access specific sequence
print(f"\nExample: Access sequence at index 0")
idx = "0"  # index_mapping keys are strings
selfies_at_0 = dataset['index_mapping'][idx]
encoded_at_0 = dataset['encoded_data'][0]
print(f"  Index 0 -> SELFIES: {selfies_at_0}")
print(f"  Encoded: {encoded_at_0}")
print(f"  Decoded: {dataset['tokenizer'].decode(encoded_at_0)}")

# ============================================================================
# 8. Using with PyTorch (Example)
# ============================================================================
print("\n" + "=" * 80)
print("8. Using with PyTorch")
print("=" * 80)

print("\nExample PyTorch integration:")
print("""
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load dataset
dataset = load_encoded_dataset('./example_dataset')
X = dataset['encoded_data']  # Shape: (100, 10)

# Convert to PyTorch tensor
X_tensor = torch.tensor(X, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Use in training loop
for batch in train_loader:
    sequences = batch[0]  # Shape: (batch_size, max_len)
    # Your model training here
    # outputs = model(sequences)
    # loss = criterion(outputs, targets)
""")

print("✓ Dataset ready for ML training!")

# ============================================================================
# 9. Summary
# ============================================================================
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)

print(f"""
✓ Tokenized {len(sample_data)} SELFIES strings
✓ Built vocabulary with {tokenizer.vocab_size} tokens
✓ Encoded sequences with max_len={chosen_max_len}
✓ Saved complete dataset package to {save_dir}/
✓ Dataset ready for ML training with PyTorch/TensorFlow

Files created:
  - example_vocab.json   (vocabulary)
  - {save_dir}/          (complete dataset package)

To use this dataset in your project:
  from utils import load_encoded_dataset
  dataset = load_encoded_dataset('{save_dir}')
  X = dataset['encoded_data']
""")

print("\n" + "=" * 80)
print("Example Complete!")
print("=" * 80)

# Cleanup instructions
print(f"""
To clean up example files:
  rm -rf {save_dir}
  rm example_vocab.json
""")
