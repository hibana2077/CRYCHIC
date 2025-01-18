git clone https://github.com/kennymckormick/pyskl.git
mv pyskl_patch/models/cnns/* pyskl/pyskl/models/cnns/
mv configs/* pyskl/configs/
cd pyskl
# This command runs well with conda 22.9.0, if you are running an early conda version and got some errors, try to update your conda first
conda env create -f pyskl.yaml
conda activate pyskl
pip install -e .
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# . "$HOME/.cargo/env" # still manually run this command
# pip install timm #  OMG it can work with python 3.7 -> only support to 0.9.12