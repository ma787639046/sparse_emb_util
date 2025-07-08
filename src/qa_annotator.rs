use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use crate::regex_tokenizer;

/// QAAnnotator uses a Regex-based tokenizer to cut the english texts, and judge whether the certain
/// documents contains the answers by sub-string list matching.
#[pyclass(name = "QAAnnotator")]
pub struct PyQAAnnotator {
    docid_to_tokenized_corpus: HashMap<String, Vec<String>>,    // docid -> List of str (Pre-Tokenized Corpus)
    tokenizer: regex_tokenizer::PyRegexTokenizer,
}

#[pymethods]
impl PyQAAnnotator {
    /// Init QAAnnotator
    /// 
    /// ## Args:
    ///     docid_to_tokenized_corpus (HashMap<String, Vec<String>>): docid -> List of str (Pre-Tokenized Corpus)
    ///     pattern (Option<&str>): Default to cut Word Boundary.
    ///     lowercase (Option<bool>): Default true, lowercase inputs.
    ///     normalize: (Option<bool>): Default true, using normalize.
    ///     normalization_from (Option<String>): Default to use "NFD"
    #[new]
    #[pyo3(signature = (
        docid_to_tokenized_corpus, 
        pattern=r"(?im)([\p{L}\p{N}\p{M}]+)|([^\p{Z}\p{C}])".to_string(), 
        lowercase=true, 
        normalize=true, 
        normalization_from="nfd".to_string()
    ))]
    fn new(
        docid_to_tokenized_corpus: HashMap<String, Vec<String>>,
        pattern: String, 
        lowercase: bool,
        normalize: bool,
        normalization_from: String,
    ) -> PyResult<Self> {
        let tokenizer = regex_tokenizer::PyRegexTokenizer::new(pattern, lowercase, normalize, normalization_from)?;
        Ok(Self {docid_to_tokenized_corpus, tokenizer})
    }

    /// Annotate the documents, judge whether the certain documents contains the answers by sub-string list matching, return qrels.
    /// 
    /// ## Pipelines:
    ///     1. Tokenize answers with regex-based tokenizer.
    ///     2. Judge whether there is at least one answer in answers that is sub-strings of tokenized_corpus.
    ///     3. Collecting and return query revelences (qrels).
    /// 
    /// ## Args:
    ///     qid_to_docids (HashMap<String, Vec<String>>): qid -> [doc_id]. All retrieval results
    ///     qid_to_answers (HashMap<String, Vec<String>>): qid -> [answer_str].
    /// 
    /// ## Returns:
    ///     qrels (HashMap<String, HashMap<String, u32>>): qid -> doc_id -> 1/0 (revelent/irrevelent) 
    fn annotate_non_optim(
        &self,
        py: Python,
        qid_to_docids: HashMap<String, Vec<String>>,    // qid -> [doc_id]. All retrieval results
        qid_to_answers: HashMap<String, Vec<String>>,   // qid -> [answer_str]
    ) -> PyResult<HashMap<String, HashMap<String, u32>>> {       // Return: {qid -> pid -> has_answer}
        py.allow_threads(|| {
            let mut qrels: HashMap<String, HashMap<String, u32>> = HashMap::new();
            for (qid, docids) in qid_to_docids.into_iter() {
                let answer_texts = qid_to_answers.get(&qid).unwrap();

                // Tokenize answers
                let mut answers: Vec<Vec<String>> = Vec::new();
                for answer in answer_texts {
                    answers.push(self.tokenizer.inner.tokenize(answer.clone()));
                }
                
                // Judge whether there is at least one answer in answers that is sub-strings of tokenized_corpus
                let mut docid_to_hasanswer: HashMap<String, u32> = HashMap::new();
                for docid in docids {
                    let tokenized_corpus = self.docid_to_tokenized_corpus.get(&docid).unwrap();
                    let hasanser = regex_tokenizer::is_subsequence_multi(&answers, tokenized_corpus);
                    docid_to_hasanswer.insert(docid, hasanser as u32);
                }

                // Add to qrels
                qrels.insert(qid, docid_to_hasanswer);
            }
            Ok(qrels)
        })
    }


    /// Multi-threaded version of Annotate the documents, judge whether the certain documents 
    /// contains the answers by sub-string list matching, return qrels.
    /// 
    /// ## Pipelines:
    ///     1. Tokenize answers with regex-based tokenizer.
    ///     2. Judge whether there is at least one answer in answers that is sub-strings of tokenized_corpus.
    ///     3. Collecting and return query revelences (qrels).
    /// 
    /// ## Args:
    ///     qid_to_docids (HashMap<String, Vec<String>>): qid -> [doc_id]. All retrieval results
    ///     qid_to_answers (HashMap<String, Vec<String>>): qid -> [answer_str].
    /// 
    /// ## Returns:
    ///     qrels (HashMap<String, HashMap<String, u32>>): qid -> doc_id -> 1/0 (revelent/irrevelent) 
    fn annotate(
        &self,
        py: Python,
        qid_to_docids: HashMap<String, Vec<String>>,    // qid -> [doc_id]. All retrieval results
        qid_to_answers: HashMap<String, Vec<String>>,   // qid -> [answer_str]
    ) -> PyResult<HashMap<String, HashMap<String, u32>>> {       // Return: {qid -> pid -> has_answer}
        py.allow_threads(|| {
            let qrels: HashMap<String, HashMap<String, u32>> = qid_to_docids
                .into_par_iter()  // Parallel qid
                .map(|(qid, docids)| {
                    // Tokenize answers
                    let answer_texts = qid_to_answers
                        .get(&qid)
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Missing answers for QID: {}", qid)))?;
                    let answers: Vec<Vec<String>> = answer_texts
                        .par_iter()
                        .map(|answer| self.tokenizer.inner.tokenize(answer.clone()))
                        .collect();

                    // Judge whether there is at least one answer in answers that is sub-strings of tokenized_corpus
                    let docid_to_hasanswer: HashMap<String, u32> = docids
                        .into_par_iter()
                        .map(|docid| {
                            let tokenized_corpus = self
                                .docid_to_tokenized_corpus
                                .get(&docid)
                                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Missing tokenized corpus for docid: {}", docid)))?;
                            let has_answer = regex_tokenizer::is_subsequence_multi(&answers, tokenized_corpus);
                            Ok((docid, has_answer as u32))
                        })
                        .collect::<PyResult<HashMap<String, u32>>>()?;

                    // Return
                    Ok((qid, docid_to_hasanswer))
                })
                .collect::<PyResult<HashMap<String, HashMap<String, u32>>>>()?;

            Ok(qrels)
        })
    }
}
