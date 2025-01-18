# CRYCHIC

CRYCHIC (Cross-modal Representation Yielding Complementary Hierarchical Integration and Consistency)

## Usage

```bash
git clone https://github.com/hibana2077/CRYCHIC.git
cd CRYCHIC/main/
bash ./conda.sh
bash ./Anaxxx.sh
bash ./install_pyskl.sh
bash ./download_skl.sh
cd pyskl
. "$HOME/.cargo/env"
conda activate pyskl_py38
pip install timm
```

run experiments

```bash
bash tools/dist_train.sh configs/crychic/ghostnet3d_gym/joint.py 6 --validate --test-last --test-best
```