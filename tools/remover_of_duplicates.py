import os
from pathlib import Path

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
BASE_DIR = r"D:\\Ninjatrader_Market_Data\\csv"  # Change this to your root folder
LOG_DELETIONS = True
# ----------------------------------------------------------

def delete_duplicate_filenames(base_dir: str):
    """
    Walks through a directory recursively and deletes files that have
    a duplicate filename *anywhere* in the directory tree.
    Keeps the first occurrence found.
    """
    seen = {}  # filename -> full path (first occurrence)

    base_path = Path(base_dir)

    for file_path in base_path.rglob("*"):
        if not file_path.is_file():
            continue

        filename = file_path.name

        if filename not in seen:
            # First time we see this filename — keep it
            seen[filename] = file_path
        else:
            # Duplicate filename found — delete this file
            try:
                file_path.unlink()
                if LOG_DELETIONS:
                    print(f"[DELETED] Duplicate: {file_path} (kept: {seen[filename]})")
            except Exception as e:
                print(f"[ERROR] Could not delete {file_path}: {e}")

    print("\nDone. All duplicates removed based on filename.")


if __name__ == "__main__":
    delete_duplicate_filenames(BASE_DIR)
