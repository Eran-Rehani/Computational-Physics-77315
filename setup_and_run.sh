#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

DEFAULT_VENV_DIR=".venv"
FALLBACK_VENV_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/computational_physics_77315/venv"
VENV_DIR="${VENV_DIR:-$DEFAULT_VENV_DIR}"

echo "=========================================="
echo " Jupyter Notebook Setup & Run Script "
echo "=========================================="

ensure_venv() {
    local target_dir="$1"

    if [ -d "$target_dir" ] && [ ! -f "$target_dir/bin/activate" ]; then
        echo "[!] Existing virtual environment looks broken (missing bin/activate). Rebuilding..."
        rm -rf "$target_dir"
    fi

    if [ ! -d "$target_dir" ]; then
        echo "[*] Creating virtual environment in $target_dir..."
        python3 -m venv --copies "$target_dir"
    else
        echo "[*] Virtual environment already exists at $target_dir."
    fi
}

# 1. Create or repair virtual environment
if ! ensure_venv "$VENV_DIR"; then
    if [ "$VENV_DIR" = "$DEFAULT_VENV_DIR" ]; then
        echo "[!] Failed to create $DEFAULT_VENV_DIR on this filesystem. Falling back to $FALLBACK_VENV_DIR..."
        VENV_DIR="$FALLBACK_VENV_DIR"
        ensure_venv "$VENV_DIR"
    else
        exit 1
    fi
fi

# 2. Activate virtual environment
echo "[*] Activating virtual environment..."
                        source "$VENV_DIR/bin/activate"
                                        
# 3. Install packages
if [ -f "requirements.txt" ]; then
    echo "[*] Installing packages from requirements.txt..."
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "[*] Installing default Jupyter packages..."
    pip install --upgrade pip
    pip install jupyter ipykernel
fi

# 4. Register kernel
echo "[*] Registering Jupyter kernel..."
python -m ipykernel install --user --name=computational_physics_77315 --display-name="Python (Computational Physics)"

echo "[*] Setup complete! Starting Jupyter Notebook..."
# 5. Start jupyter notebook
jupyter notebook
