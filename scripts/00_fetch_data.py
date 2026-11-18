"""Fetch the exact public raw file specified in config.yaml, with SHA-256 verification."""
from pathlib import Path
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hxai.common import ROOT, load_config

MIRRORS = [
    "https://raw.githubusercontent.com/yaronshap/GloballyConsistentRules/master/heloc_dataset_v1.csv",
    "https://raw.githubusercontent.com/benoitparis/explainable-challenge/master/heloc_dataset_v1.csv",
    "https://raw.githubusercontent.com/ckormylo/AI-Audit/master/heloc_dataset_v1.csv",
]

if __name__ == "__main__":
    cfg = load_config()
    dest = ROOT / cfg["data"]["raw_path"]
    expected = cfg["data"]["sha256"]
    if dest.exists():
        if hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
            raise SystemExit("Existing raw file has an unexpected hash; refusing to replace it.")
        print("Raw file already present and verified.")
        raise SystemExit(0)
    errors = []
    for url in MIRRORS:
        try:
            with urllib.request.urlopen(url, timeout=45) as response:
                data = response.read()
            digest = hashlib.sha256(data).hexdigest()
            if digest != expected:
                raise ValueError(f"unexpected SHA-256: {digest}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            (ROOT / "data" / "source.json").write_text(json.dumps({
                "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                "url": url, "sha256": digest, "bytes": len(data),
                "note": "Public secondary copy, matching the recorded raw-file SHA-256. Raw data excluded from distribution archive.",
            }, indent=2))
            print(f"Downloaded and verified {dest.name}: {digest}")
            break
        except Exception as exc:
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    else:
        raise SystemExit("No mirror yielded the expected data:\n" + "\n".join(errors))
