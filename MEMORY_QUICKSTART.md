# Memory-Efficient Processing Quick Start

## Overview

ProjectAlpha now supports processing and training on **datasets larger than your available RAM**. This is critical for production systems handling years of market data.

## Quick Commands

### Preprocessing (Memory-Efficient)

```bash
# Default: Optimized for 16GB RAM (batch-size 35000 automatic)
npm run preprocessing

# For systems with 8GB or less RAM
npm run preprocessing:memory-efficient

# Full reprocessing (still optimized for 16GB)
npm run preprocessing:full

# For legacy/unlimited RAM systems
npm run preprocessing:unlimited

# Custom batch size for your specific needs
cd preprocessing && cargo run --bin preprocessing -- \
  --input ../data/raw \
  --output ../data/preprocessed \
  --resolutions fast,mid,slow \
  --skip-existing \
  --batch-size 35000
```

### Training (Memory-Efficient)

```python
# In your training script
from streaming_dataset import build_streaming_dataloaders

# This loads data incrementally, not all at once
train_ds, val_ds, test_ds = build_streaming_dataloaders(
    feature_root=Path("data/preprocessed"),
    resolution="fast",
    train_stems=train_files,
    val_stems=val_files,
    test_stems=test_files,
    feature_cols=PHASE5_FEATURE_COLUMNS,
    target_cols=["target_t40"],
    seq_len=64,
    batch_size=128,
    streaming=True  # True for huge datasets, False for moderate
)
```

## What Changed?

### Before (Could Crash with Large Data)

**Preprocessing:**
```rust
// OLD: Load entire file into memory
let events = read_events(file, levels)?;  // ❌ Could OOM on huge files
process_all_events(&events)?;
```

**Training:**
```python
# OLD: Load everything at once
all_data = pd.concat([pd.read_parquet(f) for f in files])  # ❌ Could OOM
```

### After (Handles Unlimited Data)

**Preprocessing:**
```rust
// NEW: Process in configurable chunks
--batch-size 50000  // ✅ Processes 50k events at a time
```

**Training:**
```python
# NEW: Stream from disk incrementally
dataset = StreamingParquetDataset(files, ...)  # ✅ Constant memory usage
```

## Configuration Guide

### Preprocessing Batch Sizes

| Your RAM | Recommended --batch-size | Command |
|----------|--------------------------|---------||
| 4-8GB    | 10000-20000             | `npm run preprocessing:memory-efficient` |
| **16GB (Your System)** | **35000 (default)** | `npm run preprocessing` |
| 32GB     | 75000                   | `--batch-size 75000` |
| 64GB+    | 100000                  | `--batch-size 100000` |

### Training Strategies

| Your RAM | Dataset Size | Strategy |
|----------|--------------|----------|
| 8GB      | Any          | `streaming=True`, no caching |
| 16GB     | < 50GB       | `ChunkedParquetDataset`, `cache_size=2` |
| 32GB+    | < 100GB      | `ChunkedParquetDataset`, `cache_size=5` |
| 32GB+    | > 100GB      | `StreamingParquetDataset` |

## Package.json Scripts

```json
{
  "preprocessing": "with batching + skip existing (recommended)",
  "preprocessing:full": "full reprocessing with batching",
  "preprocessing:memory-efficient": "for systems with limited RAM",
  "training": "standard training (add streaming dataset manually)"
}
```

## Key Files

- **`MEMORY_EFFICIENCY.md`** - Comprehensive guide to memory-efficient features
- **`preprocessing/src/pipeline/executor.rs`** - Streaming preprocessing implementation
- **`training/streaming_dataset.py`** - Memory-efficient PyTorch datasets
- **`package.json`** - Ready-to-use npm scripts with batching

## Monitoring Memory

### Linux
```bash
# Real-time memory monitoring
watch -n 1 free -h

# Per-process memory
htop
```

### Python
```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024**3:.2f} GB")
```

## Troubleshooting

### Still Running Out of Memory?

**Preprocessing:**
1. Reduce `--batch-size` further
2. Process one resolution at a time
3. Close other applications

**Training:**
1. Use `StreamingParquetDataset` instead of `ChunkedParquetDataset`
2. Reduce `cache_size` in ChunkedParquetDataset
3. Reduce training `batch_size`
4. Reduce `sequence_len`

### Performance Is Slow?

**Preprocessing:**
1. Increase `--batch-size` if memory allows
2. Use SSD instead of HDD
3. Process in parallel (future feature)

**Training:**
1. Increase `cache_size` if memory allows
2. Use more DataLoader workers: `DataLoader(..., num_workers=4)`
3. Use NVMe SSD for faster I/O

## Examples

### Preprocess 1TB Dataset
```bash
# System: 16GB RAM, 1TB of raw CSV data
npm run preprocessing -- --batch-size 50000
# Processes indefinitely large datasets
```

### Train on 500GB Dataset
```python
# System: 32GB RAM, 500GB preprocessed Parquet
from streaming_dataset import StreamingParquetDataset

dataset = StreamingParquetDataset(
    file_paths=train_files,
    feature_cols=features,
    target_cols=targets,
    seq_len=64,
    batch_size=128
)
# Never loads more than one file into RAM
```

## For More Details

See **`MEMORY_EFFICIENCY.md`** for:
- Architecture details
- Implementation status
- Performance benchmarks
- Advanced configuration
- Future roadmap
