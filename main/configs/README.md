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

You can use the following command to train a model.

```bash
bash tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS} [optional arguments]
# For example: train CRYCHIC on Gym (2D skeleton, Joint Modality) with 8 GPUs, with validation, with PYSKL practice, and test the last and the best (with best validation metric) checkpoint.
bash tools/dist_train.sh configs/crychic/crychic_gym/joint.py 8 --validate --test-last --test-best
```

You can use the following command to test a model.

```bash
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${NUM_GPUS} [optional arguments]
# For example: test CRYCHIC on Gym (2D skeleton, Joint Modality) with 8 GPUs, with top-k accuracy metric, and save the result to result.pkl.
bash tools/dist_test.sh configs/crychic/crychic_gym/joint.py checkpoints/SOME_CHECKPOINT.pth 8 --eval top_k_accuracy --out result.pkl
```