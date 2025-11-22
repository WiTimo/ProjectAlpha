# Memory-Efficient Processing for Huge Datasets

This document explains how ProjectAlpha handles datasets that are too large to fit in RAM.

## Problem Statement

When processing or training on huge datasets (100GB+), loading everything into memory causes:
- System crashes due to RAM overflow
- Slow performance from excessive swapping
- Inability to process datasets larger than available RAM

## Solutions Implemented

### 1. Preprocessing: Streaming File Processing

**Location:** `preprocessing/src/pipeline/executor.rs`

#### How It Works

Instead of loading entire CSV files into a single `Vec<MarketEvent>`, the preprocessing pipeline now supports chunked processing:

```bash
# Process files in 50,000 event chunks to limit memory usage
npm run preprocessing -- --batch-size 50000

# Or use the full script with batching
cargo run --manifest-path preprocessing/Cargo.toml --bin preprocessing -- \
    --input data/raw \
    --output data/preprocessed \
    --resolutions fast,mid,slow \
    --batch-size 50000
```

#### Configuration

**CLI Parameter:**
- `--batch-size <SIZE>`: Maximum events to hold in memory (default: **35000** for 16GB RAM)
  - Set to 10000-25000 for systems with 8GB or less
  - Set to 50000-100000 for systems with 32GB+
  - Set to 0 for unlimited (only if you have abundant RAM)

**Example:**
```bash
# Default (optimized for 16GB RAM) - no flag needed
npm run preprocessing

# For systems with 8GB RAM
npm run preprocessing:memory-efficient

# For systems with 32GB+ RAM (can handle more)
cargo run -- --batch-size 75000 --input ../data/raw --output ../data/preprocessed

# Unlimited (legacy behavior, only for huge RAM systems)
cargo run -- --batch-size 0 --input ../data/raw --output ../data/preprocessed
```

#### Current Implementation Status

✅ **Implemented:**
- Chunked event reading from CSV files
- Batch-by-batch progress reporting
- Configuration via CLI and config files

⚠️ **TODO for Production:**
- Incremental Parquet writing (currently accumulates results)
- Streaming label computation without holding all events
- True zero-copy processing pipeline

### 2. Training: Streaming Data Loaders

**Location:** `training/streaming_dataset.py`

#### Two Approaches Provided

##### A. StreamingParquetDataset (IterableDataset)

Best for: Sequential training, truly massive datasets

```python
from streaming_dataset import StreamingParquetDataset

# This loads files ONE AT A TIME, never holding full dataset in RAM
dataset = StreamingParquetDataset(
    file_paths=train_files,
    feature_cols=feature_columns,
    target_cols=target_columns,
    seq_len=64,
    batch_size=128
)

# Use with DataLoader
loader = DataLoader(dataset, batch_size=None)  # batching handled by dataset
```

**Characteristics:**
- Constant memory usage regardless of dataset size
- Sequential access only (no shuffling)
- Loads one Parquet file at a time
- Ideal for datasets > 100GB

##### B. ChunkedParquetDataset (Map-Style Dataset)

Best for: Random access, datasets that partially fit in RAM

```python
from streaming_dataset import ChunkedParquetDataset

# Caches up to 5 files in memory using LRU eviction
dataset = ChunkedParquetDataset(
    file_paths=train_files,
    feature_cols=feature_columns,
    target_cols=target_columns,
    seq_len=64,
    cache_size=5  # Keep 5 files in RAM
)

# Use with regular DataLoader (supports shuffling)
loader = DataLoader(dataset, batch_size=128, shuffle=True)
```

**Characteristics:**
- Random access support (enables shuffling)
- LRU cache keeps recently-used files in RAM
- Memory usage = cache_size × avg_file_size
- Ideal for datasets 2-10x larger than RAM

#### Integration Example

```python
# In tcn.py or similar training script
from streaming_dataset import build_streaming_dataloaders

# Build datasets that load incrementally
train_ds, val_ds, test_ds = build_streaming_dataloaders(
    feature_root=Path("data/preprocessed"),
    resolution="fast",
    train_stems=train_file_stems,
    val_stems=val_file_stems,
    test_stems=test_file_stems,
    feature_cols=PHASE5_FEATURE_COLUMNS,
    target_cols=["target_t40"],
    seq_len=64,
    batch_size=128,
    streaming=True,  # True = IterableDataset, False = ChunkedDataset
)

# Use as normal
train_loader = DataLoader(train_ds, batch_size=None)  # if streaming
```

## Memory Usage Guidelines

### Preprocessing

| RAM Available | Recommended --batch-size | How to Use |
|---------------|--------------------------|------------|
| 4-8GB         | 10000-20000             | `npm run preprocessing:memory-efficient` |
| **16GB (default)** | **35000 (automatic)**  | `npm run preprocessing` |
| 32GB          | 75000                   | `--batch-size 75000` |
| 64GB+         | 100000                  | `--batch-size 100000` |
| Unlimited RAM | 0 (legacy)              | `npm run preprocessing:unlimited` |

### Training

| RAM Available | Dataset Type | Settings |
|---------------|--------------|----------|
| 8GB           | Streaming    | batch_size=32, no caching |
| 16GB          | Streaming    | batch_size=64, no caching |
| 32GB          | Chunked      | cache_size=3, batch_size=128 |
| 64GB+         | Chunked      | cache_size=5-10, batch_size=256 |

## Best Practices

### For Preprocessing

1. **Start with small batch sizes** and increase until you see good performance
2. **Monitor memory usage** during processing:
   ```bash
   watch -n 1 free -h
   ```
3. **Process in stages** if needed - preprocess a subset first to test

### For Training

1. **Use StreamingParquetDataset** for datasets > 100GB
2. **Use ChunkedParquetDataset** for datasets 2-10x your RAM
3. **Monitor GPU memory** separately - these solutions handle CPU RAM
4. **Reduce sequence_len** if running out of GPU memory
5. **Split dataset by days** rather than loading all at once

## Performance Tips

### Preprocessing

```bash
# Good: Process incrementally
cargo run -- --batch-size 50000 --skip-existing

# Avoid: Loading huge files all at once
cargo run -- --batch-size 0  # Don't do this with 100GB+ files
```

### Training

```python
# Good: Stream from disk
dataset = StreamingParquetDataset(files, ...)

# Avoid: Loading all data upfront
all_data = pd.concat([pd.read_parquet(f) for f in files])  # OOM!
```

## Monitoring Memory Usage

### Linux
```bash
# Watch memory in real-time
watch -n 1 free -h

# Check per-process memory
top -p $(pgrep -f preprocessing)
```

### Python
```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_gb = process.memory_info().rss / 1024**3
print(f"Current memory usage: {memory_gb:.2f} GB")
```

## Future Improvements

- [ ] Implement true streaming Parquet writers in Rust
- [ ] Add memory usage monitoring to pipeline
- [ ] Parallel processing with memory limits
- [ ] Distributed processing across multiple machines
- [ ] GPU-accelerated feature computation
- [ ] Memory-mapped file support for ultra-large files

## Troubleshooting

### "Out of Memory" Errors

**Preprocessing:**
```bash
# Reduce batch size
cargo run -- --batch-size 10000  # Try smaller values
```

**Training:**
```python
# Use streaming dataset
dataset = StreamingParquetDataset(...)  # Not ChunkedParquetDataset

# Reduce cache size
dataset = ChunkedParquetDataset(..., cache_size=2)

# Reduce batch size
loader = DataLoader(dataset, batch_size=32)  # Down from 128
```

### Slow Performance

**Preprocessing:**
- Increase batch size if memory allows
- Use SSD instead of HDD for data storage
- Process fewer resolutions at once

**Training:**
- Use ChunkedParquetDataset with larger cache
- Increase number of DataLoader workers
- Use faster storage (NVMe SSD)

### Data Corruption

- Always use `--skip-existing` to avoid reprocessing
- Check Parquet file integrity: `parquet-tools meta file.parquet`
- Clear corrupt outputs and reprocess

## Contact

For issues with memory-efficient processing, check:
1. This document
2. `preprocessing/README.md` for Rust-specific details
3. `training/README.md` for Python-specific details
