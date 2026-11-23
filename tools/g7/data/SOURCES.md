# G7 task items

Three frozen sets of 500 test items, the same files the InkSpark (Inkling 2 Small) recipe used, so the two models
answer identical questions. `id` is the row index in the split (Global-MMLU-Lite: the index in the pooled list of all
languages); every set is sorted by `id`.

| File | Items | Built by | sha256 |
|---|---|---|---|
| `items-gsm8k.json` | 500 of 1,319 | `make_items.py`, `random.Random(1234).sample` | `af432e2fc44763da522cd2e6229e1403a80531661d05a6b60b247a8e0ebf9994` |
| `items-mmlu_pro.json` | 500 of 12,032 | `make_items.py`, `random.Random(1234).sample` | `606a0de621992400dbd71ef609e9bc115007ac5f88dff759b84c5de6af3540b1` |
| `items-gmmlu.json` | 500 of 9,200 (23 languages x 400) | `make_gmmlu.py`, `random.Random(1234).sample` over (language, row) | `756852b9e158fdfca559b5b06c5816c639383dec3946c9d5c264a87fdddfdba6` |

GSM8K and MMLU-Pro are the standard items shared with the MiMo, Ling and InkSpark recipes (byte-identical to the
`datasets.load_dataset` sample of the quality-gate skill). MMLU-Pro categories in the sample: math 68, physics 56,
law 46, other 43, engineering 41, chemistry 40, business 33, health 32, psychology 31, biology 30, economics 29,
philosophy 23, history 16, computer science 12.

## Sources

| Dataset | Hugging Face repo @ revision | Files | Licence |
|---|---|---|---|
| GSM8K (Cobbe et al., 2021) | `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866` | `main/test-00000-of-00001.parquet`, sha256 `ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59` | MIT |
| MMLU-Pro (Wang et al., 2024) | `TIGER-Lab/MMLU-Pro` @ `b189ec765aa7ed75c8acfea42df31fdae71f97be` | `data/test-00000-of-00001.parquet`, sha256 `0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8` | MIT |
| Global-MMLU-Lite (Singh et al., 2024) | `CohereLabs/Global-MMLU-Lite` @ `36c2fd7` | `<lang>/test-*.parquet` for all 23 languages | Apache-2.0 |

Rebuild (no GPU, any image with pyarrow, `--network none`):

```bash
python3 tools/g7/make_items.py --gsm8k gsm8k-main-test.parquet --mmlu-pro mmlu_pro-test.parquet --out items-check
python3 tools/g7/make_gmmlu.py <dir of <lang>-test.parquet> items-check/items-gmmlu.json
cmp items-check/items-gsm8k.json tools/g7/data/items-gsm8k.json   # and the other two
```
