"""Run the project venv with local caches, limited threads and the bundled macOS OpenMP runtime."""
from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parents[1]
python = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if not python.exists():
    raise SystemExit("Create .venv and install requirements.txt first; see README.md.")
env = os.environ.copy()
cache = root / ".cache" / "matplotlib"
cache.mkdir(parents=True, exist_ok=True)
env["MPLCONFIGDIR"] = str(cache)
env["PYTHONUNBUFFERED"] = "1"
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    env[key] = "1"
if sys.platform == "darwin":
    candidates = list((root / ".venv" / "lib").glob("python*/site-packages/sklearn/.dylibs/libomp.dylib"))
    if candidates:
        library_dir = str(candidates[0].parent)
        existing = env.get("DYLD_LIBRARY_PATH", "")
        env["DYLD_LIBRARY_PATH"] = library_dir + (os.pathsep + existing if existing else "")
os.chdir(root)
os.execve(str(python), [str(python), *sys.argv[1:]], env)
