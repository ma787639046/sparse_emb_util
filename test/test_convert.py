import numpy as np
from sparse_emb_util import Converter

def test_convert_sparse_reps_to_json():
    vocab_dict = {
        0: "token0",
        1: "token1"
    }
    converter = Converter(vocab_dict)

    reps = np.array([[0.5, 0.0], [0.3, 0.7]], dtype=np.float32)  # shape: (2, 2)

    result = converter.convert_sparse_reps_to_json_f32(reps, quantization_factor=1000, convert_id_to_token=True)

    assert result[0]["token0"] == 500
    assert result[1]["token0"] == 300
    assert result[1]["token1"] == 700

    result = converter.convert_sparse_reps_to_json_f32(reps, quantization_factor=1000, convert_id_to_token=False)

    assert result[0]["0"] == 500
    assert result[1]["0"] == 300
    assert result[1]["1"] == 700

def test_convert_sparse_reps_to_json_f16():
    vocab_dict = {
        0: "token0",
        1: "token1"
    }
    converter = Converter(vocab_dict)

    reps = np.array([[0.5, 0.0], [0.3, 0.7]], dtype=np.float16)  # shape: (2, 2)

    result = converter.convert_sparse_reps_to_json_f16(reps, quantization_factor=1000, convert_id_to_token=True)

    assert result[0]["token0"] == 500
    assert result[1]["token0"] == 300
    assert result[1]["token1"] == 700

    result = converter.convert_sparse_reps_to_json_f16(reps, quantization_factor=1000, convert_id_to_token=False)

    assert result[0]["0"] == 500
    assert result[1]["0"] == 300
    assert result[1]["1"] == 700

def test_convert_sparse_reps_to_json_w_negs():
    vocab_dict = {
        0: "token0",
        1: "token1"
    }
    converter = Converter(vocab_dict)

    reps = np.array([[0.5, 0.0], [0.3, -0.7]], dtype=np.float32)  # shape: (2, 2)

    result = converter.convert_sparse_reps_to_json_f32(reps, quantization_factor=1000, convert_id_to_token=True, allow_negative_values=True)

    assert result[0]["token0"] == 500
    assert result[1]["token0"] == 300
    assert result[1]["neg_token1"] == 700

    result = converter.convert_sparse_reps_to_json_f32(reps, quantization_factor=1000, convert_id_to_token=False, allow_negative_values=True)

    assert result[0]["0"] == 500
    assert result[1]["0"] == 300
    assert result[1]["neg_1"] == 700


def test_convert_sparse_reps_to_pseudo_text():
    vocab_dict = {
        0: "token0",
        1: "token1"
    }
    converter = Converter(vocab_dict)

    reps = np.array([[0.5, 0.0], [0.3, 0.7]], dtype=np.float32)  # shape: (2, 2)

    result = converter.convert_sparse_reps_to_pseudo_text_f32(reps, quantization_factor=10, convert_id_to_token=True)

    assert result[0].count("token0") == 5
    assert result[1].count("token0") == 3
    assert result[1].count("token1") == 7

    result = converter.convert_sparse_reps_to_pseudo_text_f32(reps, quantization_factor=10, convert_id_to_token=False)

    assert result[0].count("0") == 5
    assert result[1].count("0") == 3
    assert result[1].count("1") == 7

def test_convert_sparse_reps_to_pseudo_text_f16():
    vocab_dict = {
        0: "token0",
        1: "token1"
    }
    converter = Converter(vocab_dict)

    reps = np.array([[0.5, 0.0], [0.3, 0.7]], dtype=np.float16)  # shape: (2, 2)

    result = converter.convert_sparse_reps_to_pseudo_text_f16(reps, quantization_factor=10, convert_id_to_token=True)

    assert result[0].count("token0") == 5
    assert result[1].count("token0") == 3
    assert result[1].count("token1") == 7

    result = converter.convert_sparse_reps_to_pseudo_text_f16(reps, quantization_factor=10, convert_id_to_token=False)

    assert result[0].count("0") == 5
    assert result[1].count("0") == 3
    assert result[1].count("1") == 7

def test_convert_sparse_reps_to_pseudo_text_w_negs():
    vocab_dict = {
        0: "token0",
        1: "token1"
    }
    converter = Converter(vocab_dict)

    reps = np.array([[0.5, 0.0], [0.3, -0.7]], dtype=np.float32)  # shape: (2, 2)

    result = converter.convert_sparse_reps_to_pseudo_text_f32(reps, quantization_factor=10, convert_id_to_token=True, allow_negative_values=True)

    assert result[0].count("token0") == 5
    assert result[1].count("token0") == 3
    assert result[1].count("neg_token1") == 7

    result = converter.convert_sparse_reps_to_pseudo_text_f32(reps, quantization_factor=10, convert_id_to_token=False, allow_negative_values=True)

    assert result[0].count("0") == 5
    assert result[1].count("0") == 3
    assert result[1].count("neg_1") == 7

# Run tests
if __name__ == "__main__":
    test_convert_sparse_reps_to_json()
    test_convert_sparse_reps_to_json_f16()
    test_convert_sparse_reps_to_json_w_negs()
    test_convert_sparse_reps_to_pseudo_text()
    test_convert_sparse_reps_to_pseudo_text_f16()
    test_convert_sparse_reps_to_pseudo_text_w_negs()
    print("All tests passed!")
