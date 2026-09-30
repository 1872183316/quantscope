# quantscope

English | [中文](README.zh-CN.md)

Find the **real** precision of a quantized LLM without downloading it.

Names such as `IQ4_XS` or `Q4_K_XL` are only the maker's recipe label. The storage format and byte
size of every tensor are written in each file's header. quantscope reads just the first few MB of
each file with HTTP range requests and reports:

- for every weight class (routed experts, shared experts, attention, linear attention/SSM,
  embeddings, output head, …): parameter count, size and the **actual bits per weight**;
- the share of each storage type within the class (e.g. `Q4_K 58%, Q5_1 35%`);
- which Q1–Q8 tier the routed experts really fall into.

Tensor sizes are computed from the offsets in the header, so quantization types added by forks
such as ik_llama.cpp are measured correctly as well (shown as `type#N`).

## Usage

Only the Python 3.10+ standard library is needed.

```bash
python quantscope.py ms:unsloth/Qwen3.8-Flash-Next-GGUF              # list the variants in a repository
python quantscope.py ms:unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL   # inspect one variant
python quantscope.py ms:unsloth/Qwen3.8-Flash-Next-GGUF --all        # compare every variant
python quantscope.py hf:Qwen/Qwen3-30B-A3B-GPTQ-Int4                 # Hugging Face
python quantscope.py https://example.com/model-00001-of-00004.gguf   # a direct URL
python quantscope.py models/xxx                                      # a local file or directory
```

`ms:` is ModelScope and `hf:` is Hugging Face. If huggingface.co is unreachable, set
`HF_ENDPOINT=https://hf-mirror.com`.

## Supported formats

- **GGUF** (llama.cpp / ik_llama.cpp, including multi-part files);
- **NInfer v3 `.ninfer`**: reads the directory JSON at the start of the file and reports storage
  formats and actual bytes per logical parameter (e.g. each expert's gate/up/down); when several
  parameters share one storage object, bytes are split by their element share;
- **safetensors**: BF16/FP16/FP8 are measured from the dtype. For packed integer weights such as
  GPTQ/AWQ, the bit width in `config.json`'s `quantization_config` recovers the real parameter
  count, and scales, zero points and other side data count toward the bytes, so the result is the
  actual bits per weight including overhead.

## Example: unsloth GGUFs of Qwen3.8-Flash-Next (2026-09-30)

| Variant | Size | Routed-expert bits/weight |
|---|---|---|
| UD-IQ1_S | 72.5 GB | 2.64 (Q2 tier) |
| UD-IQ1_M | 74.5 GB | 2.77 (Q2 tier) |
| UD-Q2_K_XL | 78.9 GB | 3.05 (Q3 tier) |
| UD-IQ3_XXS | 82.0 GB | 3.22 (Q3 tier) |
| UD-Q3_K_XL | 90.0 GB | 3.70 (Q3 tier) |
| UD-IQ4_XS | 93.7 GB | 3.94 (Q4 tier) |
| UD-Q4_K_XL | 111.3 GB | 5.10 (Q5 tier) |
| UD-Q5_K_XL | 158.3 GB | 6.51 (Q6 tier) |
| UD-Q6_K_XL | 169.2 GB | 7.24 (Q6 tier) |

`qwen3_8_flash_next.ninfer` (108.02 GB) converted by
[NInfer-Offload](https://github.com/1872183316/ninfer-offload): routed experts 4.58 bits/weight
(Q4_G64 62%, Q5_G64 38%, Q4 tier), n-gram table 5.25 bits, other projections about 8.5 bits,
output head 6.25 bits.

## Limitations

- Weights are classified by tensor name. In GGUF the GatedDeltaNet input projections are also
  called `attn_qkv`/`attn_gate`, so they are grouped under "attention (incl. GDN qkv/gate)".
- For NVFP4/MXFP4 and similar formats packed as U8 in safetensors, 4 bits are assumed when
  `config.json` does not state the bit width.

## License

Apache-2.0, see [LICENSE](LICENSE).
