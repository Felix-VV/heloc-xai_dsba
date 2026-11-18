"""Install an official portable Tectonic release inside this project, without Homebrew changes."""
from pathlib import Path
import hashlib
import io
import json
import os
import platform
import tarfile
import urllib.request
import urllib.parse

root = Path(__file__).resolve().parents[1]
dest = root / "tools" / "tectonic"
if dest.exists():
    print("Project-local Tectonic already exists.")
    raise SystemExit(0)
headers = {"User-Agent": "heloc-xai-reproducible-research"}
def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
        return response.read()
release_tag = "tectonic@0.17.0"
release = json.loads(fetch("https://api.github.com/repos/tectonic-typesetting/tectonic/releases/tags/" + urllib.parse.quote(release_tag, safe="")))
if platform.system() != "Darwin":
    raise SystemExit("This helper targets macOS. Install Tectonic or latexmk for your platform; see README.")
arch = "aarch64" if platform.machine() == "arm64" else "x86_64"
assets = [a for a in release["assets"] if f"{arch}-apple-darwin" in a["name"] and a["name"].endswith(".tar.gz")]
if len(assets) != 1:
    raise SystemExit(f"Could not uniquely identify official macOS archive: {[a['name'] for a in release['assets']]}")
asset = assets[0]
data = fetch(asset["browser_download_url"])
digest = hashlib.sha256(data).hexdigest()
if asset.get("digest") and asset["digest"] != f"sha256:{digest}":
    raise SystemExit("Official release digest mismatch.")
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
    members = [m for m in archive.getmembers() if Path(m.name).name == "tectonic" and m.isfile()]
    if len(members) != 1:
        raise SystemExit("Expected exactly one regular Tectonic executable.")
    dest.write_bytes(archive.extractfile(members[0]).read())
dest.chmod(0o755)
(root / "tools/tectonic_source.json").write_text(json.dumps({
    "release": release["tag_name"], "url": asset["browser_download_url"], "sha256": digest,
    "published_digest": asset.get("digest"),
}, indent=2))
print(f"Installed official project-local {release['tag_name']} ({digest}).")
