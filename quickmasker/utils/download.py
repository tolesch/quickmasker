"""
Download utility with progress bar.
"""
from pathlib import Path
import os
import urllib.request
from tqdm import tqdm
import logging
from typing import Optional

log = logging.getLogger(__name__)

class TqdmUpTo(tqdm):
    """Provides `update_to(block_num, block_size, total_size)` for urllib"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_with_progress(url: str, output_path: Path) -> Optional[Path]:
    """
    Downloads a file from a URL to a specified path, displaying a progress bar.
    Skips download if the file already exists. Ensures parent directory exists.

    Args:
        url (str): The URL to download from.
        output_path (Path): The directory or full file path to save to.

    Returns:
        Optional[Path]: The path to the downloaded file, or None if download failed.
    """
    log.info(f"Checking for file required by download: {output_path}")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error(f"Could not create directory {output_path.parent}: {e}")
        return None

    filename = os.path.basename(url)
    download_target = output_path / filename if output_path.is_dir() else output_path

    if download_target.exists():
        log.info(f"File already exists: {download_target}. Skipping download.")
        return download_target

    log.info(f"Downloading {url} to {download_target}...")
    try:
        with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1,
                      desc=filename) as t:
            urllib.request.urlretrieve(url, filename=str(download_target),
                                       reporthook=t.update_to, data=None)
            t.total = t.n # Adjust total to exact bytes downloaded
        log.info(f"Download complete: {download_target}")
        return download_target
    except Exception as e:
        log.error(f"Error downloading {url}: {e}", exc_info=True)
        # Clean up potentially incomplete file
        if download_target.exists():
            try:
                download_target.unlink()
                log.info(f"Removed incomplete file: {download_target}")
            except OSError as rm_err:
                log.error(f"Error removing incomplete file {download_target}: {rm_err}")
        return None
