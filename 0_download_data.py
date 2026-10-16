"""Download and cache the GAIR/LIMA dataset locally.

HF_ENDPOINT=https://hf-mirror.com python 0_download_data.py

"""

import shutil
import tempfile
from pathlib import Path

from datasets import load_dataset, load_from_disk


DATASET_ID = "pvduy/lima_vicuna_format"
LOCAL_DATA_DIR = Path(__file__).resolve().parent / "data" / "lima"


def download_data(output_dir: Path = LOCAL_DATA_DIR) -> Path:
    """Download LIMA once and persist it in the Hugging Face datasets format."""
    output_dir = Path(output_dir).expanduser().resolve()

    if output_dir.exists():
        try:
            load_from_disk(str(output_dir))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"Found an invalid dataset cache at {output_dir}. Remove it and retry."
            ) from exc
        print(f"Dataset already exists at {output_dir}; skipping download.")
        return output_dir

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=".lima-download-", dir=output_dir.parent))
    temp_dataset_dir = temp_root / "dataset"
    try:
        print(f"Downloading dataset {DATASET_ID} from Hugging Face ...")
        dataset = load_dataset(DATASET_ID)
        dataset.save_to_disk(str(temp_dataset_dir))
        temp_dataset_dir.rename(output_dir)
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    print(f"Dataset download complete, saved to: {output_dir}")
    return output_dir


if __name__ == "__main__":
    download_data()


