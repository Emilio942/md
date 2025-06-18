import os
import requests
from pathlib import Path
from typing import Optional

REMOTE_CACHE_DIR = Path(os.environ.get("PROTEINMD_REMOTE_CACHE", "~/.proteinmd_cache")).expanduser()
REMOTE_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def download_pdb(pdb_id: str, overwrite: bool = False) -> Path:
    """Lade eine PDB-Datei von der RCSB PDB und speichere sie im lokalen Cache."""
    pdb_id = pdb_id.lower()
    out_path = REMOTE_CACHE_DIR / f"{pdb_id}.pdb"
    if out_path.exists() and not overwrite:
        return out_path
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url)
    r.raise_for_status()
    out_path.write_text(r.text)
    return out_path


def fetch_rcsb_metadata(pdb_id: str) -> dict:
    """Hole Metadaten zu einer PDB-ID von der RCSB REST API."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def download_remote_file(url: str, overwrite: bool = False) -> Path:
    """Lade eine Datei per HTTP/FTP und speichere sie im lokalen Cache."""
    filename = url.split("/")[-1]
    out_path = REMOTE_CACHE_DIR / filename
    if out_path.exists() and not overwrite:
        return out_path
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return out_path


def clear_cache():
    """Leere den Remote Data Cache."""
    for f in REMOTE_CACHE_DIR.glob("*"):
        f.unlink()
