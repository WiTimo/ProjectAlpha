"""Memory-efficient streaming dataset for huge Parquet collections.

This module provides dataset classes that load data incrementally from disk
instead of loading everything into RAM at once. Essential for training on
datasets larger than available memory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset


class StreamingParquetDataset(IterableDataset):
    """Iterable dataset that streams from Parquet files without loading all data into RAM.
    
    This is critical for huge datasets - it reads files one at a time and yields
    batches, keeping memory usage constant regardless of dataset size.
    """

    def __init__(
        self,
        file_paths: List[Path],
        feature_cols: List[str],
        target_cols: List[str],
        seq_len: int,
        batch_size: int = 32,
        transform_fn: Optional[callable] = None,
    ):
        """Initialize streaming dataset.
        
        Args:
            file_paths: List of Parquet files to stream from
            feature_cols: Feature column names to extract
            target_cols: Target column names to extract
            seq_len: Sequence length for windowing
            batch_size: Number of samples to yield per batch
            transform_fn: Optional function to apply to (features, targets) before yielding
        """
        self.file_paths = sorted(file_paths)
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.transform_fn = transform_fn

    def __iter__(self):
        """Stream data from files incrementally."""
        for file_path in self.file_paths:
            try:
                # Read one file at a time
                table = pq.read_table(file_path, columns=self.feature_cols + self.target_cols)
                df = table.to_pandas()
                
                if len(df) < self.seq_len:
                    logging.warning(
                        f"Skipping {file_path.name} - only {len(df)} rows (need {self.seq_len})"
                    )
                    continue

                # Extract features and targets
                features = df[self.feature_cols].to_numpy(dtype=np.float32)
                targets = df[self.target_cols].to_numpy(dtype=np.int64)

                # Apply transform if provided (e.g., standardization)
                if self.transform_fn:
                    features, targets = self.transform_fn(features, targets)

                # Generate sequences from this file
                num_sequences = len(features) - self.seq_len + 1
                for start_idx in range(0, num_sequences, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, num_sequences)
                    batch_features = []
                    batch_targets = []
                    
                    for i in range(start_idx, end_idx):
                        seq = features[i : i + self.seq_len]
                        target = targets[i + self.seq_len - 1]
                        batch_features.append(seq)
                        batch_targets.append(target)
                    
                    if batch_features:
                        yield (
                            torch.from_numpy(np.array(batch_features, dtype=np.float32)).transpose(1, 2),
                            torch.from_numpy(np.array(batch_targets, dtype=np.int64)),
                        )

            except Exception as exc:
                logging.error(f"Error reading {file_path}: {exc}")
                continue


class ChunkedParquetDataset(Dataset):
    """Memory-efficient dataset that loads Parquet files in chunks.
    
    Unlike StreamingParquetDataset (IterableDataset), this is a map-style
    Dataset that loads files on-demand but caches the current file in memory.
    Better for random access patterns but still memory-efficient for large collections.
    """

    def __init__(
        self,
        file_paths: List[Path],
        feature_cols: List[str],
        target_cols: List[str],
        seq_len: int,
        cache_size: int = 5,
    ):
        """Initialize chunked dataset with LRU-style file caching.
        
        Args:
            file_paths: List of Parquet files to load
            feature_cols: Feature column names
            target_cols: Target column names  
            seq_len: Sequence length for windowing
            cache_size: Max number of files to keep in memory simultaneously
        """
        self.file_paths = sorted(file_paths)
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.seq_len = seq_len
        self.cache_size = max(1, cache_size)
        
        # Build index: maps global index -> (file_idx, local_idx)
        self.index_map: List[Tuple[int, int]] = []
        self.file_info: List[Dict[str, int]] = []
        
        for file_idx, file_path in enumerate(self.file_paths):
            try:
                # Just read metadata, not full file
                parquet_file = pq.ParquetFile(file_path)
                num_rows = parquet_file.metadata.num_rows
                
                if num_rows < seq_len:
                    logging.warning(f"Skipping {file_path.name} - insufficient rows")
                    continue
                
                num_sequences = num_rows - seq_len + 1
                for local_idx in range(num_sequences):
                    self.index_map.append((file_idx, local_idx))
                
                self.file_info.append({
                    'file_idx': file_idx,
                    'num_rows': num_rows,
                    'num_sequences': num_sequences,
                })
                
            except Exception as exc:
                logging.error(f"Error indexing {file_path}: {exc}")
                continue
        
        # Simple LRU cache for loaded files
        self._cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._access_order: List[int] = []
        
        logging.info(
            f"ChunkedParquetDataset: {len(self)} sequences from {len(self.file_info)} files"
        )

    def __len__(self) -> int:
        return len(self.index_map)

    def _load_file(self, file_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load a file into cache if not already loaded."""
        if file_idx in self._cache:
            # Move to end of access order (most recently used)
            self._access_order.remove(file_idx)
            self._access_order.append(file_idx)
            return self._cache[file_idx]
        
        # Load file
        file_path = self.file_paths[file_idx]
        table = pq.read_table(file_path, columns=self.feature_cols + self.target_cols)
        df = table.to_pandas()
        
        features = df[self.feature_cols].to_numpy(dtype=np.float32)
        targets = df[self.target_cols].to_numpy(dtype=np.int64)
        
        # Evict oldest if cache is full
        if len(self._cache) >= self.cache_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        # Add to cache
        self._cache[file_idx] = (features, targets)
        self._access_order.append(file_idx)
        
        return features, targets

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        file_idx, local_idx = self.index_map[idx]
        features, targets = self._load_file(file_idx)
        
        # Extract sequence
        seq_features = features[local_idx : local_idx + self.seq_len]
        seq_target = targets[local_idx + self.seq_len - 1]
        
        # Convert to tensors and transpose for Conv1d: (features, seq_len)
        return (
            torch.from_numpy(seq_features).transpose(0, 1),
            torch.from_numpy(seq_target),
        )


def discover_split_files(
    feature_root: Path,
    resolution: str,
    source_files: Iterable[str],
    limit: int = 0,
) -> List[Path]:
    """Discover Parquet files matching specific source file stems.
    
    Args:
        feature_root: Root directory with resolution subdirectories
        resolution: Resolution name (fast, mid, slow)
        source_files: Iterable of source file stems to include
        limit: Maximum number of files to return (0 = all)
    
    Returns:
        List of matching Parquet file paths
    """
    resolution_dir = feature_root / resolution
    source_set = set(source_files)
    
    matching_files = []
    for file_path in sorted(resolution_dir.glob("*.parquet")):
        if file_path.stem in source_set:
            matching_files.append(file_path)
    
    if limit > 0:
        matching_files = matching_files[:limit]
    
    logging.info(
        f"Discovered {len(matching_files)} files for {resolution} "
        f"(from {len(source_set)} source stems)"
    )
    
    return matching_files


def build_streaming_dataloaders(
    feature_root: Path,
    resolution: str,
    train_stems: List[str],
    val_stems: List[str],
    test_stems: List[str],
    feature_cols: List[str],
    target_cols: List[str],
    seq_len: int,
    batch_size: int,
    streaming: bool = True,
    transform_fn: Optional[callable] = None,
) -> Tuple[Dataset, Dataset, Dataset]:
    """Build memory-efficient datasets for training.
    
    Args:
        feature_root: Root directory for features
        resolution: Resolution to load
        train_stems: Training file stems
        val_stems: Validation file stems
        test_stems: Test file stems
        feature_cols: Feature columns to load
        target_cols: Target columns to load
        seq_len: Sequence length
        batch_size: Batch size
        streaming: Use streaming dataset (True) or chunked dataset (False)
        transform_fn: Optional transform function for data
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_files = discover_split_files(feature_root, resolution, train_stems)
    val_files = discover_split_files(feature_root, resolution, val_stems)
    test_files = discover_split_files(feature_root, resolution, test_stems)
    
    if streaming:
        train_ds = StreamingParquetDataset(
            train_files, feature_cols, target_cols, seq_len, batch_size, transform_fn
        )
        val_ds = StreamingParquetDataset(
            val_files, feature_cols, target_cols, seq_len, batch_size, transform_fn
        )
        test_ds = StreamingParquetDataset(
            test_files, feature_cols, target_cols, seq_len, batch_size, transform_fn
        )
    else:
        train_ds = ChunkedParquetDataset(
            train_files, feature_cols, target_cols, seq_len, cache_size=5
        )
        val_ds = ChunkedParquetDataset(
            val_files, feature_cols, target_cols, seq_len, cache_size=3
        )
        test_ds = ChunkedParquetDataset(
            test_files, feature_cols, target_cols, seq_len, cache_size=3
        )
    
    return train_ds, val_ds, test_ds
