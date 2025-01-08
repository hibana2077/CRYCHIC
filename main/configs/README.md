# CRYCHIC

## Usage

### 1. Install the package

```bash
pip3 install -e .
```

### 2. Move configs to pyskl directory

```bash
mv -d configs/* pyskl/configs/
```

### 3. Run the Training or Inference

```bash
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train STGCN on NTURGB+D XSub (3D skeleton, Joint Modality) with 8 GPUs, with validation, with PYSKL practice, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/crychic/crychic_gym/joint.py 8 --validate --test-last --test-best
```