# AgentPRE: core implementation

This repository contains the main implementation of supervised field mapping and
byte-level relation checking for industrial protocol format recovery. It includes
the field mapper, frozen length-convention calibration, raw-byte length and CRC
checks, a SQLite evidence store and an observed-opcode transition component.

## Quick start

Python 3.10+ is required. The checker example uses synthetic bytes and runs
without model weights, network access or external packages:

```bash
python examples/checker_demo.py
```

The example calibrates a length convention on three synthetic frames and checks
a valid and an altered frame. It demonstrates the checker; it is not a replay
of an experiment in the paper.

Install the lightweight dependencies for the field-mapping client and evaluation
modules with `python -m pip install -r requirements.txt`. The mapper connects to
a user-configured OpenAI-compatible model endpoint; no server address or API
credential is supplied. The mapper's optional gap filling is separate from the
paper's raw-output scoring, and should be disabled for raw predictions.

## Optional model inference and training

`experiments/inference/infer_measured.py` provides the recorded greedy inference
entry point. It requires a compatible GPU environment with PyTorch, Transformers,
PEFT and bitsandbytes, and user-supplied base-model, adapter and labeled JSONL
paths. Run `python experiments/inference/infer_measured.py --help` for arguments.
Set batch size and output cap to the experiment being reproduced. The repository includes packet captures but does not contain model weights,
reference labels or saved predictions; it does not by itself reproduce the paper's numerical results.

`training/qlora_example.yaml` records the main supervised fine-tuning settings
with local path placeholders for use with a compatible LLaMA-Factory installation.
Provide the training dataset and its registration separately. It is an example
configuration, not the historical training environment or a guarantee of
identical model weights. Model, dataset and dependency terms apply separately.

## Layout

- `src/agents/`: field mapping, byte-level relation checks, calibration and
  auxiliary correlation/transition logic.
- `src/storage/`: SQLite evidence store.
- `src/eval/`: prediction parsing, scoring and optional gap filling.
- `experiments/inference/`: optional measured inference entry point.
- `examples/`: synthetic checker demonstration.
- `training/`: example QLoRA settings with local placeholders.
- `data/pcaps/`: original packet captures and source attribution.

The repository contains no reference labels, saved model predictions, model
weights, frozen server metadata or fuzzing implementation. See
[`data/pcaps/README.md`](data/pcaps/README.md) for capture provenance and scope.
No project distribution license has been selected.
