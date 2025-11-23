import os
from pathlib import Path

# ----------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------
BASE_DIR = r"C:\\Users\\timowilde\\Documents\\coding\\ProjectAlpha\\data"   # Change to your directory
SIZE_LIMIT_BYTES = 50 * 1024           # 500 KB
LOG_DELETIONS = True
# ----------------------------------------------------------

def delete_small_files(base_dir: str):
    """
    Recursively deletes every file smaller than 500 KB.
    """
    base_path = Path(base_dir)

    for file_path in base_path.rglob("*"):
        if not file_path.is_file():
            continue

        file_size = file_path.stat().st_size
        
        if file_size < SIZE_LIMIT_BYTES:
            try:
                file_path.unlink()
                if LOG_DELETIONS:
                    print(f"[DELETED] {file_path} ({file_size} bytes)")
            except Exception as e:
                print(f"[ERROR] Could not delete {file_path}: {e}")

    print("\nDone. All files < 500KB have been deleted.")


if __name__ == "__main__":
    delete_small_files(BASE_DIR)
