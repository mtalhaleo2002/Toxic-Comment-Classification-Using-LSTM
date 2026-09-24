set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"

unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PIP_REQUIRE_VIRTUALENV=true
export KERAS_HOME="$project_root/.cache/keras"
export MPLCONFIGDIR="$project_root/.cache/matplotlib"
export XDG_CACHE_HOME="$project_root/.cache"
export XDG_CONFIG_HOME="$project_root/.cache/config"
export TMPDIR="$project_root/.cache/tmp"
mkdir -p "$KERAS_HOME" "$MPLCONFIGDIR" "$XDG_CONFIG_HOME" "$TMPDIR"

venv_dir="$project_root/.venv"
venv_python="$venv_dir/bin/python"
if [[ -L "$venv_dir" ]]; then
    printf '%s\n' 'The project .venv must be a local directory, not a symbolic link.' >&2
    exit 1
fi
if [[ ! -e "$venv_dir" ]]; then
    python3 -I - <<'PY'
import sys

if not (3, 10) <= sys.version_info[:2] < (3, 13):
    raise SystemExit("Python 3.10, 3.11, or 3.12 is required.")
PY
    python3 -I -m venv --without-pip "$venv_dir"
fi
if [[ ! -f "$venv_dir/pyvenv.cfg" || ! -x "$venv_python" ]]; then
    printf '%s\n' 'The existing .venv is incomplete. Recreate it with Python 3.10, 3.11, or 3.12.' >&2
    exit 1
fi

source "$venv_dir/bin/activate"
"$venv_python" -I - "$venv_dir" <<'PY'
from pathlib import Path
import sys

expected = Path(sys.argv[1]).resolve()
if not (3, 10) <= sys.version_info[:2] < (3, 13):
    raise SystemExit("The existing .venv requires Python 3.10, 3.11, or 3.12.")
if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() != expected:
    raise SystemExit("The Python interpreter is not running in the project .venv.")
configuration = {}
for line in (expected / "pyvenv.cfg").read_text().splitlines():
    key, separator, value = line.partition("=")
    if separator:
        configuration[key.strip().lower()] = value.strip().lower()
if configuration.get("include-system-site-packages") != "false":
    raise SystemExit("The .venv must have include-system-site-packages = false.")
PY

if ! "$venv_python" -m pip --version >/dev/null 2>&1; then
    if ! "$venv_python" -m ensurepip --upgrade; then
        "$venv_python" - <<'PY'
from pathlib import Path
import subprocess
import sys
import urllib.request

bootstrap = Path.cwd() / ".cache" / "get-pip.py"
with urllib.request.urlopen("https://bootstrap.pypa.io/get-pip.py", timeout=60) as response:
    bootstrap.write_bytes(response.read())
subprocess.run([sys.executable, str(bootstrap), "--no-cache-dir"], check=True)
PY
    fi
fi

"$venv_python" -m pip --isolated --no-cache-dir install -r "$project_root/requirements.txt"
"$venv_python" -m pip --isolated check

if [[ "${1:-}" == "--setup-only" ]]; then
    printf '%s\n' 'The project virtual environment is ready.'
    exit 0
fi

exec "$venv_python" "$project_root/main.py" "$@"
