# Sparse Emb Util

Efficient Sparse Embedding Utilities for IR Research

**Rust + PyO3 + Maturin | Multi-threaded | NumPy Compatible**

> 🔬 This library is part of the official implementation of  
> [**LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000× Faster Query Inference**](https://arxiv.org/abs/2505.12260).

---

## Introduction

`sparse_emb_util` is a high-performance Python extension written in Rust, designed to efficiently process sparse embeddings in sparse retrieval systems. It enables:

- Multi-threaded quantization of float16/float32 sparse vectors
- Conversion to JSON or pseudo-text format
- Regex-based and Unicode-based multilingual tokenization
- Lightweight answer annotation via substring token match

---

## Installation

### Install from PyPI

```bash
pip install sparse_emb_util
```

Or build from source:

```bash
pip install maturin
git clone https://github.com/ma787639046/sparse_emb_util.git
cd sparse_emb_util
maturin develop --release
```

## API Overview

### `Converter`: Multi-threaded Quantization of Float16/Float32 Sparse Vectors

```python
from sparse_emb_util import Converter
import numpy as np

converter = Converter(vocab_dict={0: "the", 1: "world"})
reps = np.array([[0.2, 0.8]], dtype=np.float32)

# Convert to quantized JSON (`json_reps == {'the': 20, 'world': 80}`)
json_reps = converter.convert_sparse_reps_to_json(reps, convert_id_to_token=True)

# Convert to pseudo text (`text_reps == the the ... (Repeat x20) world world world ... (Repeat x80)`)
text_reps = converter.convert_sparse_reps_to_pseudo_text(reps, convert_id_to_token=True)
```

#### Converter Methods

| Method | Input Type | Output | Description |
|--------|------------|--------|-------------|
| `convert_sparse_reps_to_json` | `np.ndarray[np.float32]` | `List[Dict[str, int]]` | Convert float32 sparse vectors to quantized JSON format |
| `convert_sparse_reps_to_json_f32` | `np.ndarray[np.float32]` | `List[Dict[str, int]]` | Same as above, explicitly for float32 |
| `convert_sparse_reps_to_json_f16` | `np.ndarray[np.float16]` | `List[Dict[str, int]]` | Convert float16 sparse vectors to quantized JSON format |
| `convert_sparse_reps_to_pseudo_text` | `np.ndarray[np.float32]` | `List[str]` | Convert float32 sparse vectors to quantized pseudo text |
| `convert_sparse_reps_to_pseudo_text_f32` | `np.ndarray[np.float32]` | `List[str]` | Same as above, explicitly for float32 |
| `convert_sparse_reps_to_pseudo_text_f16` | `np.ndarray[np.float16]` | `List[str]` | Convert float16 sparse vectors to quantized pseudo text |
| `convert_json_reps_to_pseudo_text` | `List[Dict[str, int]]` | `List[str]` | Convert JSON representations back into pseudo-text format |

Optional kwargs (supported by all functions):

- `quantization_factor`: Quantization Factor for upscale before flooring (e.g., `100`)

- `convert_id_to_token`: Whether to use `vocab_dict` to convert `IDs` to `strings`

- `allow_negative_values`: Allow `neg_` prefixed keys

- `negative_prefix`: Customize the prefix for negative values



### `RegexTokenizer`: Regex-Based Tokenization

Mimics Facebook DPR / DrQA regex logic.

```python
from sparse_emb_util import RegexTokenizer

tokenizer = RegexTokenizer(pattern=None, lowercase=True, normalize=True, normalization_from="NFD")
tokens = tokenizer.tokenize("Hello, World!")
```

- Use `batch_tokenize()` for multiple strings

- Callable: `tokenizer(["string1", "string2"])`




### `ICUWordPreTokenizer`: Unicode-Aware Tokenizer

Uses [ICU4X](https://github.com/unicode-org/icu4x) for multilingual word boundary detection.

```python
from sparse_emb_util import ICUWordPreTokenizer

tokenizer = ICUWordPreTokenizer(stopword_sets={"the", "is"})
tokens = tokenizer.tokenize("これは日本語とEnglishの混合文です。")
```

- Supports control-sequence removal, stopword filtering, and lowercasing

- Use `batch_tokenize()` or `__call__()` for batched input



### `QAAnnotator`: Question-Answer Relevance Judging

Match answers against pre-tokenized corpus via multi-thread sub-list matching for simple QA supervision.

```python
from sparse_emb_util import QAAnnotator

annotator = QAAnnotator(
    docid_to_tokenized_corpus={"docid1": ["hello", "world", "my", "friend", "!"]},
    pattern=None,
    lowercase=True,
    normalize=True,
    normalization_from="NFD"
)

# ["hello", "world"] is a sub-list of ["hello", "world", "my", "friend", "!"]
# Return `{"qid1": {"docid1": 1}}`
qrels = annotator.annotate(
    qid_to_docids={"qid1": ["docid1"]},
    qid_to_answers={"qid1": ["hello", "world"]}
)

# ["hi", "friend"] is not a sub-list of ["hello", "world", "my", "friend", "!"]
# Return `{"qid1": {"docid1": 0}}`
qrels = annotator.annotate(
    qid_to_docids={"qid1": ["docid1"]},
    qid_to_answers={"qid1": ["hi", "friend"]}
)
```


## Citation

If you use this library, please cite the following paper:

```bibtex
@misc{Ma2025LightRetriever,
    title={LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000x Faster Query Inference}, 
    author={Guangyuan Ma and Yongliang Ma and Xuanrui Gou and Zhenpeng Su and Ming Zhou and Songlin Hu},
    year={2025},
    eprint={2505.12260},
    archivePrefix={arXiv},
    primaryClass={cs.IR},
    url={https://arxiv.org/abs/2505.12260}, 
}
```

