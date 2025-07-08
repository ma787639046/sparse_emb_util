use pyo3::prelude::*;
mod converter;
mod icu;
mod regex_tokenizer;
mod qa_annotator;

/// Expose the classes to Python.
#[pymodule]
fn sparse_emb_util(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<converter::PyConverter>()?;
    m.add_class::<regex_tokenizer::PyRegexTokenizer>()?;
    m.add_class::<qa_annotator::PyQAAnnotator>()?;
    m.add_class::<icu::PyICUWordPreTokenizer>()?;
    Ok(())
}
