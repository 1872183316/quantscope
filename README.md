# quantscope

一键查清大模型量化版本的**真实**精度，不用下载模型。

模型文件名里的 `IQ4_XS`、`Q4_K_XL` 只是制作者起的配方名。每个张量实际用什么格式存储、占多少字节，
都写在文件开头的头部里。quantscope 只用 HTTP 范围请求读取每个文件开头的几 MB，就能算出：

- 每一类权重（路由专家、共享专家、注意力、线性注意力/SSM、嵌入表、输出头……）的参数量、
  大小和**实际每个权重多少 bit**；
- 各类存储格式在其中的占比（例如 `Q4_K 58%, Q5_1 35%`）；
- 路由专家实际属于 Q1～Q8 的哪一档。

张量大小是用头部里的偏移量直接算出来的，所以 ik_llama.cpp 等分支新增的量化类型也能算准
（类型名会显示为 `type#N`）。

## 用法

只依赖 Python 3.10+ 标准库。

```bash
python quantscope.py ms:unsloth/Qwen3.8-Flash-Next-GGUF              # 列出仓库里的所有版本
python quantscope.py ms:unsloth/Qwen3.8-Flash-Next-GGUF UD-Q4_K_XL   # 查一个版本
python quantscope.py ms:unsloth/Qwen3.8-Flash-Next-GGUF --all        # 对比所有版本
python quantscope.py hf:Qwen/Qwen3-30B-A3B-GPTQ-Int4                 # Hugging Face
python quantscope.py https://example.com/model-00001-of-00004.gguf   # 直接给 URL
python quantscope.py /data/models/xxx                                # 本地文件或目录
```

`ms:` 是 ModelScope，`hf:` 是 Hugging Face。连不上 huggingface.co 时，可以设置
`HF_ENDPOINT=https://hf-mirror.com`。

## 支持的格式

- **GGUF**（llama.cpp / ik_llama.cpp，包括多分片文件）；
- **safetensors**：BF16/FP16/FP8 直接按数据类型计算。GPTQ/AWQ 等打包的整数权重，按
  `config.json` 里 `quantization_config` 的 bit 数换算出真实参数量，缩放因子、零点等附加数据
  计入字节数，所以得出的是"含开销"的实际 bit/权重。

## 示例：Qwen3.8-Flash-Next 的 unsloth GGUF（2026-09-30）

| 版本名 | 大小 | 路由专家实际 bit/权重 |
|---|---|---|
| UD-IQ1_S | 72.5 GB | 2.64（Q2 档） |
| UD-IQ1_M | 74.5 GB | 2.77（Q2 档） |
| UD-Q2_K_XL | 78.9 GB | 3.05（Q3 档） |
| UD-IQ3_XXS | 82.0 GB | 3.22（Q3 档） |
| UD-Q3_K_XL | 90.0 GB | 3.70（Q3 档） |
| UD-IQ4_XS | 93.7 GB | 3.94（Q4 档） |
| UD-Q4_K_XL | 111.3 GB | 5.10（Q5 档） |
| UD-Q5_K_XL | 158.3 GB | 6.51（Q6 档） |
| UD-Q6_K_XL | 169.2 GB | 7.24（Q6 档） |

## 局限

- 权重分类靠张量名匹配。GGUF 里 GatedDeltaNet 的输入投影也叫 `attn_qkv`/`attn_gate`，
  因此归在"注意力（含 GDN qkv/gate）"一组。
- NVFP4/MXFP4 等打包成 U8 的 safetensors，如果 `config.json` 里没有写 bit 数，
  默认按 4 bit 换算。
