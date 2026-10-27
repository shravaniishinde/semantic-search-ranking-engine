"""Start the API (port 8000) and the dashboard dev server (port 5173) together.

    python scripts/dev.py            # Ctrl+C stops both

Uses the project virtualenv if present. Install frontend deps once with `npm --prefix frontend install`.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def venv_python() -> str:
    for candidate in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def main() -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise SystemExit("npm not found on PATH; install Node.js 20+ to run the dashboard")
    if not (ROOT / "frontend" / "node_modules").exists():
        subprocess.run([npm, "--prefix", "frontend", "install"], cwd=ROOT, check=True)

    env = {**os.environ, "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"}
    procs = [
        subprocess.Popen([venv_python(), "-m", "uvicorn", "app.api.main:app", "--port", "8000"], cwd=ROOT, env=env),
        subprocess.Popen([npm, "--prefix", "frontend", "run", "dev"], cwd=ROOT, env=env),
    ]
    print("API: http://localhost:8000/docs   Dashboard: http://localhost:5173   (Ctrl+C to stop)")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        pass
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()
