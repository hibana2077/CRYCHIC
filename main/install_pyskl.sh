#!/bin/bash

# Clone the pyskl repository
git clone https://github.com/kennymckormick/pyskl.git

# Move necessary files to the correct directories
mv pyskl_patch/models/cnns/* pyskl/pyskl/models/cnns/
mv configs/* pyskl/configs/

# Navigate into the pyskl directory
cd pyskl || { echo "Failed to change directory to pyskl. Exiting."; exit 1; }

# Check and update conda version (if necessary)
conda_version=$(conda --version | awk '{print $2}')
required_version="22.9.0"
if [ "$(printf '%s\n' "$required_version" "$conda_version" | sort -V | head -n1)" != "$required_version" ]; then
  echo "Updating conda to the latest version..."
  conda update -n base -c defaults conda -y
fi

# Create and activate the conda environment
conda env create -f pyskl.yaml
if [ $? -ne 0 ]; then
  echo "Error creating conda environment. Exiting."
  exit 1
fi
conda activate pyskl || { echo "Failed to activate conda environment. Exiting."; exit 1; }

# Install pyskl in editable mode
pip install -e . || { echo "Failed to install pyskl. Exiting."; exit 1; }

# Install Rust using rustup
if ! command -v rustc &> /dev/null; then
  echo "Rust is not installed. Installing Rust..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y || { echo "Failed to install Rust. Exiting."; exit 1; }
  # Source Rust environment (may require manual execution if script is run interactively)
  echo "Run the following command to activate Rust:"
  echo ". \"$HOME/.cargo/env\""
else
  echo "Rust is already installed."
fi

echo "Setup completed successfully."
