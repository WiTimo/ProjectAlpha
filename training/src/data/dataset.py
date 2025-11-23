import torch
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import List, Tuple, Dict
from src.data.loader import StreamingFileEntry

class MultiResolutionSequenceDataset(Dataset):
    def __init__(
        self,
        entries: List[StreamingFileEntry],
        seq_len: int,
        resolution_order: List[str],
    ):
        self.entries = [e for e in entries if e.rows >= seq_len]
        self.seq_len = seq_len
        self.resolution_order = resolution_order
        self.cumulative_windows = []
        self._feature_handles = {}
        self._target_handles = {}
        self._mask_handles = {}
        self.valid_targets: List[np.ndarray] = []
        
        total = 0
        for idx, entry in enumerate(self.entries):
            mask = self._open_mask(idx)
            valid_idx = np.where(mask)[0]
            valid_idx = valid_idx[valid_idx >= seq_len - 1]
            self.valid_targets.append(valid_idx)
            windows = len(valid_idx)
            self.cumulative_windows.append(total)
            total += windows
        self.total_windows = total

    def __len__(self) -> int:
        return self.total_windows

    def _open_feature(self, entry_idx: int, res: str) -> np.ndarray:
        key = (entry_idx, res)
        if key not in self._feature_handles:
            path = self.entries[entry_idx].feature_paths[res]
            self._feature_handles[key] = np.load(path, mmap_mode="r")
        return self._feature_handles[key]

    def _open_target(self, entry_idx: int) -> np.ndarray:
        if entry_idx not in self._target_handles:
            path = self.entries[entry_idx].targets_path
            self._target_handles[entry_idx] = np.load(path, mmap_mode="r")
        return self._target_handles[entry_idx]

    def _locate(self, index: int) -> Tuple[int, int]:
        # Binary search for the file index
        import bisect
        idx = bisect.bisect_right(self.cumulative_windows, index) - 1
        return idx, index - self.cumulative_windows[idx]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, offset = self._locate(index)
        target_idx = int(self.valid_targets[file_idx][offset])
        end = target_idx + 1
        start = end - self.seq_len
        
        windows = []
        for res in self.resolution_order:
            data = self._open_feature(file_idx, res)
            windows.append(data[start:end])
            
        # Concatenate features from different resolutions
        combined = np.concatenate(windows, axis=1)
        
        # Get target for the last step in sequence
        targets = self._open_target(file_idx)
        target_vec = targets[target_idx]
        
        # Transpose to (Channels, Length) for Conv1d
        x = torch.from_numpy(combined.astype(np.float32)).transpose(0, 1)
        y = torch.from_numpy(target_vec.astype(np.int64))
        meta = torch.tensor([file_idx, target_idx], dtype=torch.int64)
        
        return x, y, meta

    def _open_mask(self, entry_idx: int) -> np.ndarray:
        if entry_idx not in self._mask_handles:
            path = self.entries[entry_idx].target_mask_path
            self._mask_handles[entry_idx] = np.load(path, mmap_mode="r")
        return self._mask_handles[entry_idx]


@dataclass
class LoadedChunkEntry:
    rows: int
    feature_arrays: Dict[str, np.ndarray]
    targets_array: np.ndarray
    valid_targets: np.ndarray


class InMemoryMultiResolutionDataset(Dataset):
    """Dataset variant that operates on fully materialized chunks in RAM."""

    def __init__(
        self,
        entries: List[LoadedChunkEntry],
        seq_len: int,
        resolution_order: List[str],
    ):
        self.entries = [e for e in entries if e.rows >= seq_len]
        self.seq_len = seq_len
        self.resolution_order = resolution_order
        self.cumulative_windows = []
        total = 0
        for entry in self.entries:
            valid_idx = entry.valid_targets
            valid_idx = valid_idx[valid_idx >= seq_len - 1]
            self.cumulative_windows.append(total)
            total += len(valid_idx)
        self.total_windows = total

    def __len__(self) -> int:
        return self.total_windows

    def _locate(self, index: int) -> Tuple[int, int]:
        import bisect

        idx = bisect.bisect_right(self.cumulative_windows, index) - 1
        return idx, index - self.cumulative_windows[idx]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx, offset = self._locate(index)

        entry = self.entries[file_idx]
        target_idx = int(entry.valid_targets[offset])
        end = target_idx + 1
        start = end - self.seq_len

        windows = []
        for res in self.resolution_order:
            data = entry.feature_arrays[res]
            windows.append(data[start:end])

        combined = np.concatenate(windows, axis=1)

        targets = entry.targets_array
        target_vec = targets[target_idx]

        x = torch.from_numpy(combined.astype(np.float32)).transpose(0, 1)
        y = torch.from_numpy(target_vec.astype(np.int64))

        meta = torch.tensor([file_idx, target_idx], dtype=torch.int64)

        return x, y, meta
