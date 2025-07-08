use onig::Regex;
use unicode_normalization::UnicodeNormalization;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

/// PyO3 wrapper of RegexTokenizer. 
/// RegexTokenizer minic the tokenization code from Facebook/DPR & DrQA codebase,
/// performing a regex-based tokenization on the english string input.
#[pyclass(name = "RegexTokenizer")]
pub struct PyRegexTokenizer {
    pub inner: RegexTokenizer,
}

#[pymethods]
impl PyRegexTokenizer {
    #[new]
    #[pyo3(signature = (
        pattern=r"(?im)([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])".to_string(), 
        lowercase=true, 
        normalize=true, 
        normalization_from="nfd".to_string()
    ))]
    pub fn new(
        pattern: String, 
        lowercase: bool,
        normalize: bool,
        normalization_from: String,
    ) -> PyResult<Self> {
        // Default Args
        // + Word Boundary Regex
        // + lowercase
        // + NFD normalize
        let inner = RegexTokenizer::new(pattern, lowercase, normalize, normalization_from)
                                        .map_err(|err| PyValueError::new_err(err))?;
        Ok(Self { inner: inner })
    }

    pub fn tokenize(
        &self,
        py: Python,
        text: String,
    ) -> Vec<String> {
        py.allow_threads(|| self.inner.tokenize(text))
    }

    pub fn batch_tokenize(
        &self,
        py: Python,
        texts: Vec<String>,
    ) -> Vec<Vec<String>> {
        py.allow_threads(|| {
            texts.into_par_iter()
                 .map(|text| self.inner.tokenize(text))
                 .collect()
        })
    }
    
    pub fn __call__(
        &self,
        py: Python,
        texts: Vec<String>,
    ) -> Vec<Vec<String>> {
        self.batch_tokenize(py, texts)
    }
}

/// Judge whether `answer` is a sub-string of `text`
#[allow(unused)]
pub fn is_subsequence(answer: &Vec<String>, text: &Vec<String>) -> bool {
    if answer.is_empty() {
        return true; // Empty `answer`: Match
    }
    if answer.len() > text.len() {
        return false;
    }

    for i in 0..=(text.len() - answer.len()) {
        if &text[i..i + answer.len()] == answer.as_slice() {
            return true;    // Sub-string match
        }
    }
    false
}

/// Judge whether there exists one of the `answers` being a sub-string of `text`
#[allow(unused)]
pub fn is_subsequence_multi(answers: &Vec<Vec<String>>, text: &Vec<String>) -> bool {
    for answer in answers {
        if is_subsequence(answer, text) {
            return true;
        }
    }
    false
}

/// RegexTokenizer minic the tokenization code from Facebook/DPR & DrQA codebase,
/// performing a regex-based tokenization on the english string input.
pub struct RegexTokenizer {
    re: Regex,
    lowercase: bool,
    normalize: bool,
    normalization_from: String,
}

#[allow(unused)]
impl RegexTokenizer {
    /// Args:
    ///     pattern (String): Regex pattern to cut word boundary.
    ///     lowercase (bool): Whether to lowercase inputs.
    ///     normalize (bool): Whether to unicode normalize 
    ///     normalization_from (String):Normalization form.
    pub fn new(
        pattern: String, 
        lowercase: bool, 
        normalize: bool,
        normalization_from: String,
    ) -> Result<Self, String> {
        let re = Regex::new(pattern.as_str()).map_err(|e| e.description().to_string())?;
        let normalization_from = normalization_from.to_lowercase();
        let valid_forms = vec!["nfd", "nfc", "nfkd", "nfkc"];
        if !valid_forms.contains(&normalization_from.as_str()) {
            return Err(format!("Invalid normalization_from {}", normalization_from).to_string())
        }

        Ok(RegexTokenizer { re, lowercase, normalize, normalization_from })
    }

    /// Create a default regex tokenizer
    /// 
    /// Default Pattern:
    ///     (?i): IGNORECASE.   
    ///     (?m): MULTILINE.   
    ///     r'[\p{L}\p{N}\p{M}]+': L - Letter; N - Number; M - Mark.   
    ///     r'[^\p{Z}\p{C}]': Z - White Separator; C - Control.   
    pub fn new_default() -> Result<Self, String> {
        let pattern = r"(?im)([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])".to_string();
        let lowercase = true;
        let normalize = true;
        let normalization_from = "nfd".to_string();
        Self::new(pattern, lowercase, normalize, normalization_from)
    }

    /// Perform regex-based tokenization on `text`
    pub fn tokenize(&self, mut text: String) -> Vec<String> {
        if self.normalize {
            if self.normalization_from == "nfd" {
                text = text.nfd().collect();
            } 
            else if self.normalization_from == "nfc" {
                text = text.nfc().collect();
            }
            else if self.normalization_from == "nfkd" {
                text = text.nfkd().collect();
            }
            else if self.normalization_from == "nfkc" {
                text = text.nfkc().collect();
            }
        }

        let mut matches: Vec<String> = Vec::new();
        for cap in self.re.captures_iter(text.as_str()) {
            if let Some(matched) = cap.at(0) {
                if self.lowercase {
                    matches.push(matched.to_string().to_lowercase());
                } else {
                    matches.push(matched.to_string());
                }
            }
        }
        matches
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_word_tokenize() {
        let text = String::from("Hello, 世界! 123 😊\nAnother Line!");

        // Word Tokenize
        let tokenizer = RegexTokenizer::new_default().unwrap();
        let text_toks = tokenizer.tokenize(text);

        assert_eq!(text_toks, vec![
            "hello".to_string(), 
            ",".to_string(),
            "世界".to_string(), 
            "!".to_string(), 
            "123".to_string(), 
            "😊".to_string(), 
            "another".to_string(), 
            "line".to_string(), 
            "!".to_string()
        ]);
    }

    #[test]
    fn test_is_subsequence() {
        let text = String::from("Hello, 世界! 123 😊\nAnother Line!");
        let ans1 = String::from("Hello, 世");          // CJK words without space will not be seperated
        let ans2 = String::from("😊\nAnother Line!");
        let ans3 = String::from("\nAnother Lines!");
        let ans4 = String::from("Hello, 世界! 1234 😊");

        // Word Tokenize
        let tokenizer = RegexTokenizer::new_default().unwrap();
        let text_toks = tokenizer.tokenize(text);
        let ans1_toks = tokenizer.tokenize(ans1);
        let ans2_toks = tokenizer.tokenize(ans2);
        let ans3_toks = tokenizer.tokenize(ans3);
        let ans4_toks = tokenizer.tokenize(ans4);

        // [Test1] is_subsequence
        assert_eq!(is_subsequence(&ans1_toks, &text_toks), false);
        assert_eq!(is_subsequence(&ans2_toks, &text_toks), true);
        assert_eq!(is_subsequence(&ans3_toks, &text_toks), false);
        assert_eq!(is_subsequence(&ans4_toks, &text_toks), false);
    }

    #[test]
    fn test_is_subsequence_multi() {
        let text = String::from("Hello, 世界! 123 😊\nAnother Line!");
        let ans1 = String::from("Hello, 世");          // CJK words without space will not be seperated
        let ans2 = String::from("😊\nAnother Line!");
        let ans3 = String::from("\nAnother Lines!");
        let ans4 = String::from("Hello, 世界! 1234 😊");

        // Word Tokenize
        let tokenizer = RegexTokenizer::new_default().unwrap();
        let text_toks = tokenizer.tokenize(text);
        let ans1_toks = tokenizer.tokenize(ans1);
        let ans2_toks = tokenizer.tokenize(ans2);
        let ans3_toks = tokenizer.tokenize(ans3);
        let ans4_toks = tokenizer.tokenize(ans4);

        // [Test2] is_subsequence_multi
        let ans_toks_vec = vec![ans1_toks, ans2_toks, ans3_toks, ans4_toks];
        assert_eq!(is_subsequence_multi(&ans_toks_vec, &text_toks), true);
    }
}

