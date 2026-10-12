# Supervised Fine-Tuning

TrajCond-VLA trains the same model in two SFT stages:

- **Stage 1** — Learn driving reasoning and discrete trajectory actions (`CoT → AT`)
- **Stage 2** — Initialize from Stage 1 and learn conditioned planning (`AT → CoT → PT`)

## Environment

Install the SFT and vocabulary tool dependencies, including [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), then install DeepSpeed separately for training:

```bash
pip install -r sft/requirements.txt
```

The requirements pin PyTorch `2.8.0`, torchvision `0.23.0`, Transformers `4.57.1`, and LLaMA-Factory `0.9.4`.

The YAMLs use official LLaMA-Factory fields and the `qwen3_vl` template. Driving CoT remains in the JSON `explanation` field as `<thinking>...</thinking>`.

## Expand the action vocabulary

Run the standalone tool once before Stage 1:

```bash
python sft/extend_action_vocab.py \
  --base-model /path/to/Qwen3-VL-4B-Instruct \
  --output-dir models/Qwen3-VL-4B-Instruct-768action \
  --num-action-tokens 768
```

Set `--num-action-tokens` to the actual total codebook size. The tool supports any positive count, including 512. For Brohan, use the sum of the three axis sizes; for KMeans/KDisks, use the number of codebook rows. The current codebooks all use 768 action tokens and can share the same expanded base model. Four structural special tokens are added separately and do not count toward this argument.

The tool adds `<action_0>` through `<action_(N-1)>` as ordinary `AddedToken` entries and registers `<thinking>`, `</thinking>`, `<answer>`, and `</answer>` as additional special tokens, preserving the model's existing special tokens. It checks atomic encoding and resizes both input and output embeddings. Newly added thinking tags are initialized from the mean embedding of the tokenized `explanation`; answer tags use the tokenized `PT`. The same initialization is applied to an untied output head using its own reference rows. Existing tag embeddings are preserved.

It saves model weights, tokenizer, and multimodal processor together, then copies every remaining top-level source file only if the destination file does not exist. Newly saved files are never overwritten, and source subdirectories are not copied. It reloads the tokenizer to check token IDs and special-token registration. The output directory must be new or empty. Loading defaults to BF16 on CPU; `--dtype float16` or `--dtype float32` changes the saved weight precision. Decode with `skip_special_tokens=False` to retain the four tags.

## Train

Run from the repository root after constructing the datasets. Update model, dataset, and output paths in the selected YAML:

| Representation | Stage 1 | Stage 2 |
|----------------|---------|---------|
| KMeans | [navsim_kmeans_stage1.yaml](configs/navsim_kmeans_stage1.yaml) | [navsim_kmeans_stage2.yaml](configs/navsim_kmeans_stage2.yaml) |
| KDisks | [navsim_kdisks_stage1.yaml](configs/navsim_kdisks_stage1.yaml) | [navsim_kdisks_stage2.yaml](configs/navsim_kdisks_stage2.yaml) |
| Brohan | [navsim_brohan_stage1.yaml](configs/navsim_brohan_stage1.yaml) | [navsim_brohan_stage2.yaml](configs/navsim_brohan_stage2.yaml) |

```bash
FORCE_TORCHRUN=1 NPROC_PER_NODE=16 llamafactory-cli train sft/configs/navsim_brohan_stage1.yaml
FORCE_TORCHRUN=1 NPROC_PER_NODE=16 llamafactory-cli train sft/configs/navsim_brohan_stage2.yaml
```

Stage 1 uses `models/Qwen3-VL-4B-Instruct-768action`; Stage 2 uses the corresponding `models/<method>_stage1` checkpoint. Vocabulary expansion is completed before training, so the YAMLs do not contain an action token list or resize the vocabulary again.

The flat configs follow the supplied training parameters: full SFT, three epochs, learning rate `4e-5`, BF16, cosine scheduling, training and evaluation batch size 2 per GPU, accumulation 1, `image_max_pixels: 262144`, and `cutoff_len: 65536`. Sixteen GPUs give a global training batch size of 32. Vision, multimodal projector, and language parameters remain trainable in both stages. Validation uses 1% of the dataset every 100 steps; checkpoints are saved every 1000 steps, keeping one.

The [dataset registry](datasets/dataset_info.json) points to the six QA files under `data_engine/output/`. Both absolute and relative image paths can be used in the QA files for your training environment. LLaMA-Factory resolves relative image paths from `dataset_dir` by default. Preserve the expanded tokenizer and processor files in both stage checkpoints.

## DeepSpeed

All six configs use the existing [ZeRO-2 configuration](../deepspeed/ds_z2_config.json). Batch size and mixed precision are resolved from the training arguments.

## GRPO

Reinforcement fine-tuning follows the [Curious-VLA GRPO workflow](https://github.com/Mashiroln/curious_vla/blob/main/docs/train_grpo.md). Start from the Stage-2 checkpoint and preserve its action vocabulary. Adapt rollout prompts and the reward trajectory parser to the `AT → CoT → PT` JSON contract and the same fixed pose statistics.
