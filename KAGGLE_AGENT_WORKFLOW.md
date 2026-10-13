# Local AI agent ↔ Kaggle GPU workflow (Day 10)

The GitHub repo is the source of truth. The local agent (Claude/Codex in VS Code)
edits and tests code, Kaggle executes GPU jobs, and the agent inspects retrieved
outputs before making the next change. Kaggle CLI batch jobs are **not** an
interactive connection to an already running Kaggle notebook.

## One-time local setup (PowerShell, Python 3.11+)

```powershell
cd C:\AI\vinai20k\K4-Track-4-Day10-VisionLanguageModel-02873
python -m pip install --upgrade kaggle nbformat pytest
kaggle auth login
kaggle kernels list --user YOUR_KAGGLE_USERNAME
```

Do not give agent chat logs/API tokens; let Kaggle CLI read your local authenticated
session. Never commit `kaggle.json`, `access_token`, or raw credentials.

## Execution loop

1. Make the change on a Git branch; run CPU tests and validate notebook JSON.
2. Commit and push. Record the **exact Git SHA** in the runner.
3. Create a **small separate Kaggle runner** under `kaggle_run/` for a short
   smoke test. Do not publish or automatically execute the full lab notebook,
   because it also runs expensive E0, E1–E4 cells and contains Colab-only imports.
4. The runner installs `requirements-colab.txt`, clones the repository at the
   pinned commit, prepares the dataset (from Kaggle Input or Kaggle CLI if enabled),
   then runs only the selected smoke-test stage. It should write a machine-readable
   `status.json` and on failure `traceback.log` into `/kaggle/working/`.
   Use `try/except` around the **runner invocation** to write the traceback
   and re-raise, so Kaggle still marks the job failed.
5. Start with `max_steps=1`, `batch_size=1`. Only advance to 300 steps after
   model loading, forward/backward, checkpoint round-trip, and peak VRAM are verified.
6. Review metrics, versions, logs, and code differences before submitting new runs.

## Kaggle CLI commands (local terminal)

```powershell
kaggle kernels init -p kaggle_run
# Edit kaggle_run/kernel-metadata.json:
#   id: YOUR_USERNAME/day10-vlm-smoke
#   code_file: runner.ipynb (relative to metadata)
#   language: python; kernel_type: notebook
#   is_private: true; enable_gpu: true; enable_internet: true
#   machine_shape: NvidiaTeslaT4
# Save a minimal runner.ipynb next to kernel-metadata.json
kaggle kernels push -p kaggle_run --no-run   # upload only, no GPU used
kaggle kernels push -p kaggle_run            # upload + run (uses GPU quota)
kaggle kernels status YOUR_USERNAME/day10-vlm-smoke
kaggle kernels files YOUR_USERNAME/day10-vlm-smoke
kaggle kernels output YOUR_USERNAME/day10-vlm-smoke -p kaggle_results
```

The exact Kaggle username is private account information: use the real one, never
guess. Update `kernel-metadata.json` from the initialized template.
`enable_internet` is needed when the runner clones GitHub/downloads Hugging Face;
Kaggle Internet access must actually be enabled in the notebook settings.

Do not assume `kaggle kernels output` provides full traceback for every failed
run: explicitly write `traceback.log` to the working output and/or inspect
the executed notebook's Kaggle Logs UI. Do not assume `kaggle kernels pull`
provides a live interactive terminal.

## Specific pitfalls of this lab

- Keep the official `scripts/` and `tests/` unchanged.
- E2/E3 (and E4 when based on E2) use **trainable, unquantized fp32** mergers.
  `param.data.to(float32)` is not a valid way to dequantize a bitsandbytes
  `Params4bit` layer. The notebook converts projector `Linear4bit` modules
  to normal `nn.Linear` before PEFT injection.
- Save projector checkpoints in `work/*_projector.safetensors` as CPU tensors,
  and restore them **after** preparing the inference model's projectors.
  Do not load legacy quantized `*.pt` with `weights_only=False`.
- Verify `mm_token_type_ids`, matched LoRA targets, finite loss, gradients,
  projector checkpoint roundtrip, and VRAM before training long jobs.
- The Kaggle output filesystem is ephemeral. Export `report/evidence/`, logs,
  and small results. Model checkpoints can be large; manage their persistence
  deliberately and never push them to GitHub.
- Run E0–E4 before optional GRPO. Keep test split held out.

Useful CLI docs:
https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels_metadata.md

## Kaggle T4 x2 and the OOM gate

Kaggle's `NvidiaTeslaT4` accelerator can provide **two** T4 GPUs in a
single Kaggle kernel. Confirm availability in each session:

```python
import torch
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
```

The training notebook now loads the base model through `load_base_model()`.
If it sees two CUDA devices, it uses Hugging Face's
`device_map="balanced"` in one Python process with an 11 GiB placement budget
per GPU. This is model sharding, **not DDP**: do not launch `torchrun` or
`accelerate launch` with two processes, which would duplicate the model.
The placement is visible through `model.hf_device_map` and MUST be verified
on the running GPUs.

The original OOM occurred when Qwen3-VL generated logits for every image,
prompt and answer token. `answer_only_loss` now restricts the LM head to the
answer-supervised tail, preserving the masked next-token cross-entropy. The
training micro-batch is 1 with accumulation 8, retaining effective batch 8.
Smoke-test first; automatic sharding is a heuristic and may still OOM.

After an OOM restart the Kaggle kernel to release retained model tensors before
retrying. Model-sharding support for training is less mature than for inference,
so only mark the GPU gate passed after real forward, backward, gradient check,
and projector safetensors round-trip on Kaggle. Log per-device peak VRAM.
