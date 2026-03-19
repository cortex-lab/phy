#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/release_smoke_test.sh local
  scripts/release_smoke_test.sh pypi
  scripts/release_smoke_test.sh open

Environment variables:
  PYTHON                    Python executable to use for venv creation (default: python3)
  UV                        uv executable to use for venv/package management (default: uv)
  RELEASE_SMOKE_DATASET     Dataset directory containing params.py
                           (default: ../phy-data/template)
  RELEASE_SMOKE_ENV         Virtualenv directory to create/use
  RELEASE_SMOKE_VERSION     Version to install in pypi mode
  RELEASE_SMOKE_INDEX_URL   Optional --index-url value for pypi mode
  RELEASE_SMOKE_EXTRA_INDEX_URL
                           Optional --extra-index-url value for pypi mode
EOF
}

if [[ $# -ne 1 ]]; then
    usage
    exit 1
fi

MODE="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
UV_BIN="${UV:-uv}"
DATASET_DIR="${RELEASE_SMOKE_DATASET:-$REPO_ROOT/../phy-data/template}"

case "$MODE" in
    local)
        ENV_DIR="${RELEASE_SMOKE_ENV:-$REPO_ROOT/.release-smoke/local}"
        ;;
    pypi)
        VERSION="${RELEASE_SMOKE_VERSION:-}"
        if [[ -z "$VERSION" ]]; then
            echo "RELEASE_SMOKE_VERSION must be set in pypi mode." >&2
            exit 1
        fi
        ENV_DIR="${RELEASE_SMOKE_ENV:-$REPO_ROOT/.release-smoke/pypi-$VERSION}"
        ;;
    open)
        ENV_DIR="${RELEASE_SMOKE_ENV:-$REPO_ROOT/.release-smoke/local}"
        ;;
    *)
        usage
        exit 1
        ;;
esac

PARAMS_PATH="$DATASET_DIR/params.py"

if [[ ! -f "$PARAMS_PATH" ]]; then
    echo "Dataset params file not found: $PARAMS_PATH" >&2
    exit 1
fi

if [[ -x "$ENV_DIR/bin/python" ]]; then
    ENV_PYTHON="$ENV_DIR/bin/python"
    ENV_PHY="$ENV_DIR/bin/phy"
else
    ENV_PYTHON="$ENV_DIR/Scripts/python.exe"
    ENV_PHY="$ENV_DIR/Scripts/phy.exe"
fi

make_venv() {
    rm -rf "$ENV_DIR"
    mkdir -p "$(dirname "$ENV_DIR")"
    "$UV_BIN" venv --python "$PYTHON_BIN" "$ENV_DIR"
}

require_env() {
    if [[ ! -x "$ENV_PYTHON" || ! -x "$ENV_PHY" ]]; then
        echo "Virtualenv not found or incomplete: $ENV_DIR" >&2
        echo "Run the matching smoke target first." >&2
        exit 1
    fi
}

install_local() {
    local wheel

    wheel="$(ls -t "$REPO_ROOT"/dist/phy-*.whl 2>/dev/null | head -n 1 || true)"
    if [[ -z "$wheel" ]]; then
        echo "No wheel found under dist/. Run 'make build' first." >&2
        exit 1
    fi

    "$UV_BIN" pip install --python "$ENV_PYTHON" "$wheel"
}

install_pypi() {
    local -a pip_args

    pip_args=()
    if [[ -n "${RELEASE_SMOKE_INDEX_URL:-}" ]]; then
        pip_args+=(--index-url "$RELEASE_SMOKE_INDEX_URL")
    fi
    if [[ -n "${RELEASE_SMOKE_EXTRA_INDEX_URL:-}" ]]; then
        pip_args+=(--extra-index-url "$RELEASE_SMOKE_EXTRA_INDEX_URL")
    fi

    "$UV_BIN" pip install --python "$ENV_PYTHON" "${pip_args[@]}" "phy==$VERSION"
}

verify_install() {
    "$ENV_PYTHON" -c "import phy, phylib; print('phy', phy.__version__); print('phylib', phylib.__version__)"
    "$ENV_PYTHON" -c "import PyQt5; print('PyQt5', PyQt5.__file__)"
    "$ENV_PHY" --version
    "$ENV_PHY" template-describe "$PARAMS_PATH"
}

open_gui() {
    require_env
    echo "Launching phy on $PARAMS_PATH"
    exec "$ENV_PHY" template-gui "$PARAMS_PATH"
}

print_next_steps() {
    cat <<EOF

Smoke checks passed.

Environment: $ENV_DIR
Dataset:     $DATASET_DIR
Params:      $PARAMS_PATH

Manual GUI check:
  $ENV_PHY template-gui "$PARAMS_PATH"
EOF
}

case "$MODE" in
    local)
        make_venv
        install_local
        verify_install
        print_next_steps
        ;;
    pypi)
        make_venv
        install_pypi
        verify_install
        print_next_steps
        ;;
    open)
        open_gui
        ;;
esac
