import numpy as np
from typing import Optional, List, Dict

class Converter:
    """ Converter can convert sparse arrays to JSON / Pseudo String format efficiently """
    def __init__(self, vocab_dict: Optional[Dict[int, str]] = None):
        """ Args:
                vocab_dict (Option<HashMap<i32, String>>): A map of `token_id -> token_str`
        """
        pass

    def convert_sparse_reps_to_json(
        self, 
        reps: np.ndarray[np.float32], 
        quantization_factor: Optional[int] = 100, 
        convert_id_to_token: Optional[bool] = False,
        allow_negative_values: Optional[bool] = False,
        negative_prefix: Optional[str] = "neg_"
    ) -> List[Dict[str, int]]: 
        """ Same as `self.convert_sparse_reps_to_json_f32`. 
        A float32 multi-threaded version of Convert sparse representations to quantized JSON format.
        Format: `{token_id / token: int frequency}`, all keys are `str`.
        
        Args:
            reps (PyReadonlyArray2<f32>): Numpy f32 array, shape [batch_size, vocab_dim]
            quantization_factor (i32): Upscaling factor. Quantized reps = (reps * quantization_factor).floor()
            convert_id_to_token (bool): True - Return token str; False - Return token_id str
            allow_negative_values (bool): Whether to preserve negative values.
            negative_prefix (&str): Sparse reps doesn't allow negative values. Thus the entry of negative value
                                    will be `{negative_prefix}_{token}`.
        """
        pass

    def convert_sparse_reps_to_json_f32(
        self, 
        reps: np.ndarray[np.float32], 
        quantization_factor: Optional[int] = 100, 
        convert_id_to_token: Optional[bool] = False,
        allow_negative_values: Optional[bool] = False,
        negative_prefix: Optional[str] = "neg_"
    ) -> List[Dict[str, int]]: 
        """ A float32 multi-threaded version of Convert sparse representations to quantized JSON format.
        Format: `{token_id / token: int frequency}`, all keys are `str`.
        
        Args:
            reps (PyReadonlyArray2<f32>): Numpy f32 array, shape [batch_size, vocab_dim]
            quantization_factor (i32): Upscaling factor. Quantized reps = (reps * quantization_factor).floor()
            convert_id_to_token (bool): True - Return token str; False - Return token_id str
            allow_negative_values (bool): Whether to preserve negative values.
            negative_prefix (&str): Sparse reps doesn't allow negative values. Thus the entry of negative value
                                    will be `{negative_prefix}_{token}`.
        """
        pass

    def convert_sparse_reps_to_json_f16(
        self, 
        reps: np.ndarray[np.float16], 
        quantization_factor: Optional[int] = 100, 
        convert_id_to_token: Optional[bool] = False,
        allow_negative_values: Optional[bool] = False,
        negative_prefix: Optional[str] = "neg_"
    ) -> List[Dict[str, int]]: 
        """ A float16 multi-threaded version of Convert sparse representations to quantized JSON format.
        Format: `{token_id / token: int frequency}`, all keys are `str`.
        
        Args:
            reps (PyReadonlyArray2<f16>): Numpy f16 array, shape [batch_size, vocab_dim]
            quantization_factor (i32): Upscaling factor. Quantized reps = (reps * quantization_factor).floor()
            convert_id_to_token (bool): True - Return token str; False - Return token_id str
            allow_negative_values (bool): Whether to preserve negative values.
            negative_prefix (&str): Sparse reps doesn't allow negative values. Thus the entry of negative value
                                    will be `{negative_prefix}_{token}`.
        """
        pass



    def convert_sparse_reps_to_pseudo_text(
        self, 
        reps: np.ndarray[np.float32], 
        quantization_factor: Optional[int] = 100, 
        convert_id_to_token: Optional[bool] = False,
        allow_negative_values: Optional[bool] = False,
        negative_prefix: Optional[str] = "neg_"
    ) -> List[str]:
        """ Same as `self.convert_sparse_reps_to_pseudo_text_f32`. 
        A float32 multi-threaded version of Convert sparse representations to quantized pseudo text.
        Format: `token1 token1 ... token1 token2 token2 ... token2 ...`, each `tokenx` will be repeated `frequency times`.
        
        Args:
            reps (PyReadonlyArray2<f32>): Numpy f32 array, shape [batch_size, vocab_dim]
            quantization_factor (i32): Upscaling factor. Quantized reps = (reps * quantization_factor).floor()
            convert_id_to_token (bool): True - Return token str; False - Return token_id str
            allow_negative_values (bool): Whether to preserve negative values.
            negative_prefix (&str): Sparse reps doesn't allow negative values. Thus the entry of negative value
                                    will be `{negative_prefix}_{token}`.
        """
        pass

    def convert_sparse_reps_to_pseudo_text_f32(
        self, 
        reps: np.ndarray[np.float32], 
        quantization_factor: Optional[int] = 100, 
        convert_id_to_token: Optional[bool] = False,
        allow_negative_values: Optional[bool] = False,
        negative_prefix: Optional[str] = "neg_"
    ) -> List[str]:
        """ A float32 multi-threaded version of Convert sparse representations to quantized pseudo text.
        Format: `token1 token1 ... token1 token2 token2 ... token2 ...`, each `tokenx` will be repeated `frequency times`.
        
        Args:
            reps (PyReadonlyArray2<f32>): Numpy f32 array, shape [batch_size, vocab_dim]
            quantization_factor (i32): Upscaling factor. Quantized reps = (reps * quantization_factor).floor()
            convert_id_to_token (bool): True - Return token str; False - Return token_id str
            allow_negative_values (bool): Whether to preserve negative values.
            negative_prefix (&str): Sparse reps doesn't allow negative values. Thus the entry of negative value
                                    will be `{negative_prefix}_{token}`.
        """
        pass

    def convert_sparse_reps_to_pseudo_text_f16(
        self, 
        reps: np.ndarray[np.float16], 
        quantization_factor: Optional[int] = 100, 
        convert_id_to_token: Optional[bool] = False,
        allow_negative_values: Optional[bool] = False,
        negative_prefix: Optional[str] = "neg_"
    ) -> List[str]:
        """ A float16 multi-threaded version of Convert sparse representations to quantized pseudo text.
        Format: `token1 token1 ... token1 token2 token2 ... token2 ...`, each `tokenx` will be repeated `frequency times`.
        
        Args:
            reps (PyReadonlyArray2<f16>): Numpy f16 array, shape [batch_size, vocab_dim]
            quantization_factor (i32): Upscaling factor. Quantized reps = (reps * quantization_factor).floor()
            convert_id_to_token (bool): True - Return token str; False - Return token_id str
            allow_negative_values (bool): Whether to preserve negative values.
            negative_prefix (&str): Sparse reps doesn't allow negative values. Thus the entry of negative value
                                    will be `{negative_prefix}_{token}`.
        """
        pass

    def convert_json_reps_to_pseudo_text(
        self,
        json_reps: list[dict[str, int]]
    ):
        """ Convert json reps to pseudo text
        
        Args:
            json_reps (Vec<HashMap<String, i32>>): Format: `{token_id / token: int frequency}`, all keys are `str`.
        """
        pass


class RegexTokenizer:
    """ RegexTokenizer minic the tokenization code from Facebook/DPR & DrQA codebase,
        performing a regex-based tokenization on the english string input. """
    def __init__(
        pattern: Optional[str], 
        lowercase: Optional[bool],
        normalize: Optional[bool],
        normalization_from: Optional[str],
    ):
        """ Create a regex tokenizer
        Args:
            pattern (Option<&str>): Default to cut Word Boundary.
            lowercase (Option<bool>): Default true, lowercase inputs.
            normalize: (Option<bool>): Default true, using normalize.
            normalization_from (Option<String>): Default to use "NFD"
        
        Default Pattern: 
            r"(?im)([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])"

        Explain:
            (?i): IGNORECASE.   
            (?m): MULTILINE.   
            r'[\p{L}\p{N}\p{M}]+': L - Letter; N - Number; M - Mark.   
            r'[^\p{Z}\p{C}]': Z - White Separator; C - Control.   
        """
        pass

    def tokenize(text: str) -> list[str]:
        """ Perform regex-based tokenization on `text` """
        pass

    def batch_tokenize(texts: list[str]) -> list[list[str]]:
        """ A batched version of performing regex-based tokenization on `texts` """
        pass

    def __call__(texts: list[str]) -> list[list[str]]:
        """ Same as `self.batch_tokenize`. 
        A batched version of performing regex-based tokenization on `texts` """
        pass


class QAAnnotator:
    """ QAAnnotator uses a Regex-based tokenizer to cut the english texts, and judge whether the certain
        documents contains the answers by sub-string list matching.
    """
    def __init__(
        self,
        docid_to_tokenized_corpus: dict[str, list[str]],
        pattern: Optional[str], 
        lowercase: Optional[bool],
        normalize: Optional[bool],
        normalization_from: Optional[str],
    ):
        """ Init QAAnnotator
    
        Args:
            docid_to_tokenized_corpus (HashMap<String, Vec<String>>): docid -> List of str (Pre-Tokenized Corpus)
            pattern (Option<&str>): Default to cut Word Boundary.
            lowercase (Option<bool>): Default true, lowercase inputs.
            normalize: (Option<bool>): Default true, using normalize.
            normalization_from (Option<String>): Default to use "NFD"
        """
        pass

    def annotate(
        self,
        qid_to_docids: dict[str, list[str]],    # qid -> [doc_id]. All retrieval results
        qid_to_answers: dict[str, list[str]],   # qid -> [answer_str]
    ) -> dict[str, dict[str, int]]:
        """ Multi-threaded version of Annotate the documents, judge whether the certain documents 
            contains the answers by sub-string list matching, return qrels.

        Pipelines:
            1. Tokenize answers with regex-based tokenizer.
            2. Judge whether there is at least one answer in answers that is sub-strings of tokenized_corpus.
            3. Collecting and return query revelences (qrels).

        Args:
            qid_to_docids (HashMap<String, Vec<String>>): qid -> [doc_id]. All retrieval results
            qid_to_answers (HashMap<String, Vec<String>>): qid -> [answer_str].

        Returns:
            qrels (HashMap<String, HashMap<String, u32>>): qid -> doc_id -> 1/0 (revelent/irrevelent)
        """
        pass


class ICUWordPreTokenizer:
    """ `ICUWordPreTokenizer` cuts the word boundary with [ICU4X](https://github.com/unicode-org/icu4x) 
        International Components for Unicode. It supports cutting any language, backed by a LSTM model
        and the dictionary model for Chinese and Japanese. It will return the words list without whitespaces.
    """
    def __init__(self, stopword_sets: set[str]) -> None: 
        """ Init func
    
            Args:
                stopword_sets (HashSet<String>): Set of stopwords str.
        """
        ...
    
    def tokenize(self, text: str, remove_stopwords=False, lowercase=True) -> list[str]:
        """ Pre-Tokenize the text by cutting with word boundary.

            Processing pipeline:
                1. Remove all non-visable control sequences, regex r"[\p{Cc}\p{Cs}\p{Cn}]+"
                2. Lowercase the text if set. (Default True. Aligning with the needs of sparse reps.)
                3. Cutting the texts by itering through the word boundary defined by International Components 
                for Unicode, removing all whitespaces. Then return the words list without whitespaces.
            
            Args:
                text (String): String text.
                remove_stopwords (bool): Whether to remove stopwords defined in `self.stopword_sets`. Default `false`.
                lowercase (bool): Whether to lowercase the inputs. Default `true`.
        """
        ...
    
    def batch_tokenize(self, texts: list[str], remove_stopwords=False, lowercase=True) -> list[list[str]]:
        """ A multi-threaded version of Pre-Tokenizing the texts by cutting with word boundary.
            
            Processing pipeline:
                1. Remove all non-visable control sequences, regex r"[\p{Cc}\p{Cs}\p{Cn}]+"
                2. Lowercase the text if set. (Default True. Aligning with the needs of sparse reps.)
                3. Cutting the texts by itering through the word boundary defined by International Components 
                   for Unicode, removing all whitespaces. Then return the words list without whitespaces.
            
            Args:
                texts (Vec<String>): List of texts.
                remove_stopwords (bool): Whether to remove stopwords defined in `self.stopword_sets`. Default `false`.
                lowercase (bool): Whether to lowercase the inputs. Default `true`.
        """
        ...
    
    def __call__(self, texts: list[str], remove_stopwords=False, lowercase=True) -> list[list[str]]:
        """ Same as `self.batch_tokenize`. 
        A multi-threaded version of Pre-Tokenizing the texts by cutting with word boundary.
            
            Processing pipeline:
                1. Remove all non-visable control sequences, regex r"[\p{Cc}\p{Cs}\p{Cn}]+"
                2. Lowercase the text if set. (Default True. Aligning with the needs of sparse reps.)
                3. Cutting the texts by itering through the word boundary defined by International Components 
                   for Unicode, removing all whitespaces. Then return the words list without whitespaces.
            
            Args:
                texts (Vec<String>): List of texts.
                remove_stopwords (bool): Whether to remove stopwords defined in `self.stopword_sets`. Default `false`.
                lowercase (bool): Whether to lowercase the inputs. Default `true`.
        """
        ...

