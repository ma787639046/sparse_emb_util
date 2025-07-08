import time
import torch

from sparse_emb_util import Converter
from pyimpl_converter import Converter as ConverterPy

def compare_torch_float32_to_sparse_json():
    print("== compare_torch_float32_to_sparse_json ==")

    bs = 400
    vocab_size = 200000

    # Generate a random vector
    emb = torch.rand((bs, vocab_size), dtype=torch.float32)

    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = emb < torch.topk(emb, 1024)[0][..., -1, None]
    emb = emb.masked_fill(indices_to_remove, 0.)

    converter = Converter()

    start_time = time.time()
    for _ in range(10):
        spr_emb_rust = converter.convert_sparse_reps_to_json(emb.numpy())
    print(f"[Rust] {time.time()-start_time} s")

    converter_py = ConverterPy()

    start_time = time.time()
    for _ in range(10):
        spr_emb_py = converter_py.convert_sparse_reps_to_json(emb)
    print(f"[Python] {time.time()-start_time} s")

    print()


def compare_torch_float32_to_sparse_string():
    print("== compare_torch_float32_to_sparse_string ==")

    bs = 400
    vocab_size = 200000

    # Generate a random vector
    emb = torch.rand((bs, vocab_size), dtype=torch.float32)

    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = emb < torch.topk(emb, 1024)[0][..., -1, None]
    emb = emb.masked_fill(indices_to_remove, 0.)

    converter = Converter()

    start_time = time.time()
    for _ in range(1):
        spr_emb_rust = converter.convert_sparse_reps_to_pseudo_text(emb.numpy(), quantization_factor=100)
    print(f"[Rust] {time.time()-start_time} s")

    converter_py = ConverterPy()

    start_time = time.time()
    for _ in range(1):
        spr_emb_py = converter_py.convert_sparse_reps_to_pseudo_text(emb, quantization_factor=100)
    print(f"[Python] {time.time()-start_time} s")

    print()

if __name__ == '__main__':
    """
    == compare_torch_float32_to_sparse_json ==
    [Rust] 0.6627538204193115 s
    [Python] 2.354196786880493 s

    == compare_torch_float32_to_sparse_string ==
    [Rust] 0.1326291561126709 s
    [Python] 58.02565312385559 s
    """
    compare_torch_float32_to_sparse_json()
    compare_torch_float32_to_sparse_string()

