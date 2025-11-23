import logging
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, Iterator, List, Sequence

import numpy as np

from src.data.loader import StreamingFileEntry
from src.data.dataset import LoadedChunkEntry


@dataclass(frozen=True)
class ChunkSpec:
    """Represents a set of cached files that should fit in memory together."""

    chunk_id: int
    entries: List[StreamingFileEntry]
    est_bytes: int


@dataclass(frozen=True)
class ChunkPayload:
    """Materialized chunk that lives fully in RAM until released."""

    spec: ChunkSpec
    entries: List[LoadedChunkEntry]
    realized_bytes: int


def _path_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def estimate_entry_bytes(entry: StreamingFileEntry, resolution_order: Sequence[str]) -> int:
    total = 0
    for res in resolution_order:
        path = entry.feature_paths.get(res)
        if isinstance(path, Path):
            total += _path_size(path)
        else:
            total += int(getattr(path, "nbytes", 0))
    targets_path = entry.targets_path
    if isinstance(targets_path, Path):
        total += _path_size(targets_path)
    else:
        total += int(getattr(targets_path, "nbytes", 0))
    return total


def build_chunk_specs(
    entries: Sequence[StreamingFileEntry],
    resolution_order: Sequence[str],
    max_chunk_bytes: int,
    shuffle: bool,
    seed: int,
) -> Iterator[ChunkSpec]:
    if max_chunk_bytes <= 0:
        raise ValueError("max_chunk_bytes must be positive")

    order = list(entries)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)

    chunk: List[StreamingFileEntry] = []
    chunk_bytes = 0
    chunk_id = 0

    for entry in order:
        entry_bytes = max(estimate_entry_bytes(entry, resolution_order), 1)

        # If adding this entry would overflow the chunk, flush the chunk first.
        if chunk and chunk_bytes + entry_bytes > max_chunk_bytes:
            yield ChunkSpec(chunk_id=chunk_id, entries=list(chunk), est_bytes=chunk_bytes)
            chunk_id += 1
            chunk.clear()
            chunk_bytes = 0

        # Always include the entry, even if it alone exceeds the desired chunk size.
        chunk.append(entry)
        chunk_bytes += entry_bytes

        if chunk_bytes >= max_chunk_bytes:
            yield ChunkSpec(chunk_id=chunk_id, entries=list(chunk), est_bytes=chunk_bytes)
            chunk_id += 1
            chunk.clear()
            chunk_bytes = 0

    if chunk:
        yield ChunkSpec(chunk_id=chunk_id, entries=list(chunk), est_bytes=chunk_bytes)


def _materialize_entry(entry: StreamingFileEntry, resolution_order: Sequence[str]) -> LoadedChunkEntry:
    feature_arrays = {}
    for res in resolution_order:
        path = entry.feature_paths[res]
        # np.load without mmap_mode loads the full array into RAM.
        arr = np.load(path, allow_pickle=False)
        feature_arrays[res] = np.asarray(arr, dtype=np.float32)
    targets = np.load(entry.targets_path, allow_pickle=False)
    targets = np.asarray(targets, dtype=np.int64)
    mask = np.load(entry.target_mask_path, allow_pickle=False)
    valid_targets = np.where(mask)[0].astype(np.int64)
    return LoadedChunkEntry(
        rows=entry.rows,
        feature_arrays=feature_arrays,
        targets_array=targets,
        valid_targets=valid_targets,
    )


def materialize_chunk(spec: ChunkSpec, resolution_order: Sequence[str]) -> ChunkPayload:
    loaded_entries: List[LoadedChunkEntry] = []
    realized_bytes = 0

    for entry in spec.entries:
        loaded = _materialize_entry(entry, resolution_order)
        loaded_entries.append(loaded)
        realized_bytes += sum(arr.nbytes for arr in loaded.feature_arrays.values())
        realized_bytes += loaded.targets_array.nbytes

    logging.debug(
        "Chunk %s loaded (%0.2f MB est -> %0.2f MB) with %d entries",
        spec.chunk_id,
        spec.est_bytes / 1e6,
        realized_bytes / 1e6,
        len(loaded_entries),
    )

    return ChunkPayload(spec=spec, entries=loaded_entries, realized_bytes=realized_bytes)


class ChunkPrefetchIterator:
    """Iterates over materialized chunks with optional background prefetching."""

    def __init__(
        self,
        specs: Iterable[ChunkSpec],
        resolution_order: Sequence[str],
        prefetch: int,
    ) -> None:
        self._specs = iter(specs)
        self._resolution_order = resolution_order
        self._prefetch = max(prefetch, 0)

    def __iter__(self) -> Iterator[ChunkPayload]:
        if self._prefetch <= 0:
            for spec in self._specs:
                yield materialize_chunk(spec, self._resolution_order)
            return

        with ThreadPoolExecutor(max_workers=self._prefetch) as pool:
            queue: Deque[Future] = deque()

            def _submit() -> bool:
                try:
                    spec = next(self._specs)
                except StopIteration:
                    return False
                queue.append(pool.submit(materialize_chunk, spec, self._resolution_order))
                return True

            for _ in range(self._prefetch):
                if not _submit():
                    break

            while queue:
                future = queue.popleft()
                result = future.result()
                _submit()
                yield result