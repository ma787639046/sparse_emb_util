use pyo3::prelude::*;
use pyo3::exceptions::*;
use icu_segmenter::WordSegmenter;
use itertools::Itertools;
use onig::Regex;
// use rayon::prelude::*;
use std::collections::HashSet;

/// PyO3 wrapper of ICUWordPreTokenizer.
/// 
/// `ICUWordPreTokenizer` cuts the word boundary with [ICU4X](https://github.com/unicode-org/icu4x) 
/// International Components for Unicode. It supports cutting any language, backed by a LSTM model
/// and the dictionary model for Chinese and Japanese. It will return the words list without whitespaces.
#[pyclass(name = "ICUWordPreTokenizer")]
pub struct PyICUWordPreTokenizer {
    tokenizer: ICUWordPreTokenizer
}

#[pymethods]
impl PyICUWordPreTokenizer {
    #[new]
    #[pyo3(signature = (stopword_sets = HashSet::new()))]
    pub fn new(stopword_sets: HashSet<String>) -> PyResult<Self> {
        let tokenizer = ICUWordPreTokenizer::new(stopword_sets).map_err(|err| PyValueError::new_err(err))?;

        Ok(Self {tokenizer})
    }

    #[pyo3(signature = (text, remove_stopwords = false, lowercase = true))]
    pub fn tokenize(
        &self,
        py: Python,
        text: String,
        remove_stopwords: bool,
        lowercase: bool,
    ) -> Vec<String> {
        py.allow_threads(|| {
            self.tokenizer.tokenize(text, remove_stopwords, lowercase)
        })
    }

    #[pyo3(signature = (texts, remove_stopwords = false, lowercase = true))]
    pub fn batch_tokenize(
        &self,
        py: Python,
        texts: Vec<String>,
        remove_stopwords: bool,
        lowercase: bool,
    ) -> Vec<Vec<String>> {
        py.allow_threads(|| {
            self.tokenizer.batch_tokenize(texts, remove_stopwords, lowercase)
        })
    }

    #[pyo3(signature = (texts, remove_stopwords = false, lowercase = true))]
    pub fn __call__(
        &self,
        py: Python,
        texts: Vec<String>,
        remove_stopwords: bool,
        lowercase: bool,
    ) -> Vec<Vec<String>> {
        self.batch_tokenize(py, texts, remove_stopwords, lowercase)
    }
}


/// `ICUWordPreTokenizer` cuts the word boundary with [ICU4X](https://github.com/unicode-org/icu4x) 
/// International Components for Unicode. It supports cutting any language, backed by a LSTM model
/// and the dictionary model for Chinese and Japanese. It will return the words list without whitespaces.
#[allow(unused)]
pub struct ICUWordPreTokenizer {
    word_segmenter: WordSegmenter,
    re_bad_chars: Regex,
    stopword_sets: HashSet<String>,
}

#[allow(unused)]
impl ICUWordPreTokenizer {
    /// Init func
    /// 
    /// ### Args:
    ///     
    ///      stopword_sets (HashSet<String>): Set of stopwords str.
    pub fn new(stopword_sets: HashSet<String>) -> Result<Self, String> {
        let word_segmenter = WordSegmenter::new_auto();
        let re_bad_chars = Regex::new(r"[\p{Cc}\p{Cs}\p{Cn}]+")
            .map_err(|e| e.description().to_string())?;

        Ok(Self {
            word_segmenter,
            re_bad_chars,
            stopword_sets,
        })
    }

    /// Pre-Tokenize the text by cutting with word boundary.
    /// 
    /// ### Processing pipeline:
    ///     1. Remove all non-visable control sequences, regex r"[\p{Cc}\p{Cs}\p{Cn}]+"
    ///     2. Lowercase the text if set. (Default True. Aligning with the needs of sparse reps.)
    ///     3. Cutting the texts by itering through the word boundary defined by International Components 
    ///        for Unicode, removing all whitespaces. Then return the words list without whitespaces.
    /// 
    /// ### Args:
    ///     text (String): String text.
    ///     remove_stopwords (bool): Whether to remove stopwords defined in `self.stopword_sets`. Default `false`.
    ///     lowercase (bool): Whether to lowercase the inputs. Default `true`.
    pub fn tokenize(
        &self,
        text: String,
        remove_stopwords: bool,
        lowercase: bool,
    ) -> Vec<String> {
        // Remove invalid characters and trim
        let mut clean_text = self.re_bad_chars.replace_all(text.as_str(), "")
                                                      .trim()
                                                      .to_string();

        if clean_text.is_empty() {
            return vec![];
        }

        // Convert to lowercase if required
        if lowercase {
            clean_text = clean_text.to_lowercase();
        }

        // Perform word segmentation
        let mut words = Vec::new();

        for (start, end) in self.word_segmenter.segment_str(&clean_text).tuple_windows().into_iter() {
            let word = clean_text[start..end].trim();
            if !word.is_empty() {
                if remove_stopwords {
                    if !self.stopword_sets.contains(word) {
                        words.push(word.to_string());
                    }
                } else {
                    words.push(word.to_string());
                }
            }
        }

        words
    }

    /// A multi-threaded version of Pre-Tokenizing the texts by cutting with word boundary.
    /// 
    /// ### Processing pipeline:
    ///     1. Remove all non-visable control sequences, regex r"[\p{Cc}\p{Cs}\p{Cn}]+"
    ///     2. Lowercase the text if set. (Default True. Aligning with the needs of sparse reps.)
    ///     3. Cutting the texts by itering through the word boundary defined by International Components 
    ///        for Unicode, removing all whitespaces. Then return the words list without whitespaces.
    /// 
    /// ### Args:
    ///     texts (Vec<String>): List of texts.
    ///     remove_stopwords (bool): Whether to remove stopwords defined in `self.stopword_sets`. Default `false`.
    ///     lowercase (bool): Whether to lowercase the inputs. Default `true`.
    pub fn batch_tokenize(
        &self,
        texts: Vec<String>,
        remove_stopwords: bool,
        lowercase: bool,
    ) -> Vec<Vec<String>> {
        texts
            // .into_par_iter()     # This causes hang when using Python Multi-processing
            .into_iter()
            .map(|text| self.tokenize(text, remove_stopwords, lowercase))
            .collect()
    }
}

