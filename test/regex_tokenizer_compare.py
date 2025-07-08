import time
import orjson
from pyimpl_regex_tokenizer import SimpleTokenizer, _normalize
from sparse_emb_util import RegexTokenizer
from tqdm import tqdm

# 100000it [00:07, 13715.81it/s]
# ========= compare_regex_tokenizer done: All regex splited results are equal to each other. =========
def compare_regex_tokenizer(corpus_path: str):
    corpus: list[str] = []
    with open(corpus_path, "r") as f:
        for line in f:
            item = orjson.loads(line)
            content: str = item["title"] + " " + item["text"]
            corpus.append(content)
    
    tokenizer_py = SimpleTokenizer()
    tokenizer_rs = RegexTokenizer()

    tokenized_rs_batched: list[list[str]] = tokenizer_rs.batch_tokenize(corpus)

    mismatch_cnt = 0
    for text, tokenized_rs in tqdm(zip(corpus, tokenized_rs_batched)):
        tokenized_py = tokenizer_py.tokenize(_normalize(text)).words(uncased=True)

        if tokenized_py != tokenized_rs:
            print(f"[{mismatch_cnt}] Mismatch: ")
            print(f"tokenized_py: {tokenized_py}")
            print(f"tokenized_rs: {tokenized_rs}")
            mismatch_cnt += 1
    
    if mismatch_cnt == 0:
        print("========= compare_regex_tokenizer done: All regex splited results are equal to each other. =========")
    else:
        print(f"========= compare_regex_tokenizer done: mismatch_cnt {mismatch_cnt}. =========")


# 1. Regex Tokenizer (Batched) uses: 1.6079902648925781
# 2. Regex Tokenizer (No Batch) uses: 15.535161972045898
# 3. Python Tokenizer uses: 7.121313810348511
def compare_speeds(corpus_path: str):
    corpus: list[str] = []
    with open(corpus_path, "r") as f:
        for line in f:
            item = orjson.loads(line)
            content: str = item["title"] + " " + item["text"]
            corpus.append(content)
    
    tokenizer_py = SimpleTokenizer()
    tokenizer_rs = RegexTokenizer()

    # 1. Regex Tokenizer (Batched)
    start_time = time.time()
    tokenized_rs_batched = tokenizer_rs.batch_tokenize(corpus)
    print(f"1. Regex Tokenizer (Batched) uses: {time.time() - start_time}")

    # 2. Regex Tokenizer (No Batch)
    start_time = time.time()
    for line in corpus:
        tokenized_rs = tokenizer_rs.tokenize(line)
    print(f"2. Regex Tokenizer (No Batch) uses: {time.time() - start_time}")

    # 3. Python Tokenizer
    start_time = time.time()
    for line in corpus:
        tokenized_py = tokenizer_py.tokenize(_normalize(line)).words(uncased=True)
    print(f"3. Python Tokenizer uses: {time.time() - start_time}")


if __name__ == "__main__":
    test_corpus_path = "test/nq.100k.debug.jsonl"
    compare_regex_tokenizer(test_corpus_path)
    compare_speeds(test_corpus_path)
