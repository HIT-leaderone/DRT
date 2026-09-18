# DRT Training Data

This directory records the training data used by the DRT SFT and RL runs.

## SFT Data

- Hugging Face dataset: https://huggingface.co/datasets/leaderonehit/DRT-SFT-8B-training-data
- Files: 20 parquet shards
- Rows: 194,719
- Columns: `problem_id`, `content`, `role`, `image`
- Size: about 30.4 GiB

## RL Data

- Hugging Face dataset: https://huggingface.co/datasets/leaderonehit/DRT-RL-8B-training-data
- Manifest: [`rl_training_data_manifest.json`](rl_training_data_manifest.json)
- Files: 4
- Rows:
  - `train.parquet`: 14,724
  - `test.parquet`: 618
  - `mmfinereason_train.parquet`: 32,362
- Size: about 1.6 GiB

The original RL parquet files are hosted on Hugging Face without modifying their contents. Machine-specific paths have been removed from accompanying metadata. Git stores only documentation and manifests. The default `visionr1_dapo_train` configuration contains the original training split; `visionr1_dapo_test` contains the test split; `mmfinereason` contains the additional experimental training dataset, not the paper's primary RL dataset. Train and test are separate configurations because their nested `extra_info` fields differ. Do not automatically mix the two training files when reproducing the original VisionR1 + DAPO run.

## Reproduction

Public download (no internal HDFS access required):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="leaderonehit/DRT-RL-8B-training-data",
    repo_type="dataset",
    local_dir="training_data/VisionR1_DAPO",
)
```

Load an individual dataset configuration:

```python
from datasets import load_dataset

train = load_dataset(
    "leaderonehit/DRT-RL-8B-training-data", "visionr1_dapo_train", split="train"
)
```
