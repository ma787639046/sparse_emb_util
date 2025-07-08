from sparse_emb_util import ICUWordPreTokenizer as ICUWordPreTokenizerRs
from pyimpl_icu_word_tokenizer import ICUWordPreTokenizer as ICUWordPreTokenizerPy

# [rs] ICU Word tokenize text use: 0.5409548282623291
# [py] ICU Word tokenize text use: 1.611401081085205
def test_word_tokenze_speed():
    import time

    text = "这是一句中文！。This is an English sentence. 여기에 한국어 문장이 있습니다.!" * 100
    text2 = "This is a test sentence. 这是一个测试句子。" * 100
    text3 = "Hello World!!. 这是一个测试句子。" * 100
    text4 = "你好👋。你好👋你好👋你好👋你好👋你好👋" * 100
    texts = [text, text2, text3, text4] * 1_000

    icu_tokenizer_rs = ICUWordPreTokenizerRs()
    start_time = time.time()
    token_lists_rs = icu_tokenizer_rs(texts)
    print(f"[rs] ICU Word tokenize text use: {time.time()-start_time}")

    icu_tokenizer_py = ICUWordPreTokenizerPy()
    start_time = time.time()
    for idx, _text in enumerate(texts):
        token_list = icu_tokenizer_py(_text)

        # ICU4X cutting is right: ['你好', '👋', '。', '你好', '👋', '你好', '👋', '你好', '👋', ...]
        # ICU4C cutting is wrong: ['你好', '👋。', '你', '好👋', '你好', '👋你', '好👋', '你好', '👋你', '好👋', '你好', ...]
        # assert token_lists_rs[idx] == token_list
    print(f"[py] ICU Word tokenize text use: {time.time()-start_time}")

    print()

if __name__ == "__main__":
    test_word_tokenze_speed()
