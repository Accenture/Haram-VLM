---
license: apache-2.0
base_model: microsoft/Phi-3-vision-128k-instruct
library_name: peft
tags:
  - vision-language
  - hallucination
  - lora
  - phi-3-vision
  - pope
datasets:
  - detection-datasets/coco
pipeline_tag: image-text-to-text
---

# HARAM-VLM — Phi-3-Vision LoRA adapters

Hallucination-aware adaptive-resolution adapters for `microsoft/Phi-3-vision-128k-instruct`.
Code and evaluation harness: **https://github.com/samaonline/HARAM-VLM**

These are **PEFT LoRA adapters**, not full models. You need the Phi-3-Vision base weights,
which are downloaded separately from Microsoft's repository.

## Results

Held-out POPE, image-disjoint, 6k probes per split (`haram_full_4gpu_20260610_072155`):

| Split | Accuracy | F1 | Hallucination |
|---|---|---|---|
| Random | 0.939 | 0.939 | 5.7% |
| Popular | 0.949 | 0.948 | 3.4% |
| Adversarial | 0.908 | 0.910 | 11.5% |

Evaluation uses **regenerated, image-disjoint** POPE-style probes rather than the upstream
POPE files. The original training set overlapped standard POPE test images by 93%; the
`_oldbaseline` adapter below is that contaminated model, retained so the before/after
comparison is reproducible.

## Contents

| Directory | Role |
|---|---|
| `haram_full_4gpu_20260610_072155/` | **Main clean model** — the numbers above. Start here. |
| `haram_full_4gpu_20260610_003136/` | Contaminated first run; controlled before/after baseline. |
| `haram_visionlora_4gpu_20260610_204531/` | Ablation adding vision-tower LoRA. **Negative result.** |
| `haram_full_4gpu_20260611_044833/` | Larger-data run. |
| `haram_plainlora_6k_4gpu_20260615_030648/` | Plain LoRA without the HARAM heads. |

## Configuration

Rank-64 LoRA (`lora_alpha=64`, `lora_dropout=0.05`), targeting **LLM layers only** —
`qkv_proj`, `o_proj`, `gate_up_proj`, `down_proj` across all 32 blocks. The vision tower
is frozen: the vision-LoRA ablation was negative (mean F1 0.932 → 0.929, mean
hallucination 6.9% → 7.2%).

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoProcessor

base = "microsoft/Phi-3-vision-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    base, trust_remote_code=True, torch_dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(
    model, "samaonline/haram-vlm-phi3v-lora",
    subfolder="haram_full_4gpu_20260610_072155")
processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
```

## Limitations

Please read these before building on the results.

- **Resolution routing is training-time only.** The resolution router and adaptive token
  manager act as auxiliary training signals. At inference the model runs at a fixed
  16-crop resolution — inference-time adaptive routing is *not* implemented here.
- **The HARAM heads contribute little.** They add roughly +0.013 mean F1 over plain
  LoRA (0.919 -> 0.932); the LoRA fine-tuning itself does the work.
- **Vision-LoRA does not help** (see above); the default keeps the vision tower frozen.
- **These adapters are not the paper's headline contribution.** That is a *training-free*
  scout-and-escalate controller that wraps any frozen VLM and needs none of these
  weights. This checkpoint exists to study how calibration affects the controller.
- Evaluated on COCO-derived POPE probes only. Behaviour on other domains is uncharacterised.
- Inherits all limitations and biases of the Phi-3-Vision base model.

## License

Apache-2.0. Derived from `microsoft/Phi-3-vision-128k-instruct` (MIT) and built on
training code from [Phi3-Vision-Finetune](https://github.com/2U1/Phi3-Vision-Finetune)
(Apache-2.0). See the `NOTICE` file in the GitHub repository for full attribution.
COCO and POPE retain their own licenses and are not redistributed.
