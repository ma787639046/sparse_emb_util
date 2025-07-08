use half::f16;
use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyValueError, PyKeyError};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

/// ## Converter can convert sparse arrays to JSON / Pseudo String format efficiently
/// 
/// ### Args:
///     vocab_dict (Option<HashMap<i32, String>>): A map of `token_id -> token_str`
/// 
#[pyclass(name = "Converter")]
pub struct PyConverter {
    vocab_dict: HashMap<i32, String>,
}

#[pymethods]
impl PyConverter {
    #[new]
    #[pyo3(signature = (vocab_dict=HashMap::new()))]
    pub fn new(vocab_dict: HashMap<i32, String>) -> Self {
        Self { vocab_dict }
    }

    /// Same as `self.convert_sparse_reps_to_json_f32`
    /// A float32 multi-threaded version of Convert sparse representations to quantized JSON format.
    /// Format: `{token_id / token: int frequency}`, all keys are `str`.
    /// 
    /// ### Args:
    ///     reps (PyReadonlyArray2<f32>): Numpy f32 array, shape [batch_size, vocab_dim]
    ///     quantization_factor (i32): Upscaling factor. Quantized reps = (reps * quantization_factor).floor()
    ///     convert_id_to_token (bool): True - Return token str; False - Return token_id str
    ///     allow_negative_values (bool): Whether to preserve negative values.
    ///     negative_prefix (&str): Sparse reps doesn't allow negative values. Thus the entry of negative value
    ///                             will be `{negative_prefix}_{token}`.
    #[pyo3(signature = (reps, quantization_factor=100, convert_id_to_token=false, allow_negative_values=false, negative_prefix="neg_"))]
    pub fn convert_sparse_reps_to_json(
        &self,
        py: Python,
        reps: PyReadonlyArray2<f32>,
        quantization_factor: i32,
        convert_id_to_token: bool,
        allow_negative_values: bool,
        negative_prefix: &str,
    ) -> PyResult<Vec<HashMap<String, i32>>> {
        self.convert_sparse_reps_to_json_f32(py, reps, quantization_factor, convert_id_to_token, allow_negative_values, negative_prefix)
    }

    /// A float32 multi-threaded version of Convert sparse representations to quantized JSON format.
    /// Format: `{token_id / token: int frequency}`, all keys are `str`.
    /// 
    /// ### Args:
    ///     reps (PyReadonlyArray2<f32>): Numpy f32 array, shape [batch_size, vocab_dim]
    ///     quantization_factor (i32): Upscaling factor. Quantized reps = (reps * quantization_factor).floor()
    ///     convert_id_to_token (bool): True - Return token str; False - Return token_id str
    ///     allow_negative_values (bool): Whether to preserve negative values.
    ///     negative_prefix (&str): Sparse reps doesn't allow negative values. Thus the entry of negative value
    ///                             will be `{negative_prefix}_{token}`.
    #[pyo3(signature = (reps, quantization_factor=100, convert_id_to_token=false, allow_negative_values=false, negative_prefix="neg_"))]
    pub fn convert_sparse_reps_to_json_f32(
        &self,
        py: Python,
        reps: PyReadonlyArray2<f32>,
        quantization_factor: i32,
        convert_id_to_token: bool,
        allow_negative_values: bool,
        negative_prefix: &str,
    ) -> PyResult<Vec<HashMap<String, i32>>> {
        let reps = reps.as_array();

        if reps.shape().len() != 2 {
            return Err(PyValueError::new_err("Input numpy array must be 2-dimensional."));
        }

        let result = py.allow_threads(|| {
            let quant_factor = quantization_factor as f32;

            (0..reps.shape()[0])
                .into_par_iter()
                .map(|batch_id| {
                    let mut sparse_rep: HashMap<String, i32> = HashMap::new();

                    for (vocab_id, &value) in reps.row(batch_id).indexed_iter() {
                        // Value is not zero
                        // Large margin (1e-4) is used here, because the reps may be casted from
                        // other reps with lower precision
                        if value > 1e-4 || value < -1e-4 { 
                            if value < -1e-4 && !allow_negative_values {
                                continue;
                            }

                            // Quantize it
                            let quantized_value = (value * quant_factor).floor() as i32;

                            let token_key = if convert_id_to_token {
                                                        self.vocab_dict
                                                            .get(&(vocab_id as i32))
                                                            .ok_or_else(|| PyErr::new::<PyKeyError, _>(
                                                                format!("Token id {} not found in vocab_dict.", vocab_id)))?
                                                            .clone()
                                                    } else {
                                                        vocab_id.to_string()
                                                    };

                            if quantized_value > 0 {
                                sparse_rep.insert(token_key, quantized_value);
                            } else if quantized_value < 0 && allow_negative_values {
                                sparse_rep.insert(negative_prefix.to_string() + &token_key, -quantized_value);
                            }
                        }
                    }

                    if sparse_rep.is_empty() {
                        let pad_key = if convert_id_to_token {
                                        "[PAD]".to_string()
                                    } else {
                                        "-1".to_string()
                                    };
                        sparse_rep.insert(pad_key,1);
                    }

                    Ok(sparse_rep)
                })
                .collect::<PyResult<Vec<_>>>() // Collect into Vec<HashMap<String, i32>>
        });

        result
    }

    /// A float16 multi-threaded version of Convert sparse representations to quantized JSON format.
    /// Format: `{token_id / token: int frequency}`, all keys are `str`.
    /// 
    /// ### Args:
    ///     reps (PyReadonlyArray2<f16>): Numpy f16 array, shape [batch_size, vocab_dim]
    ///     quantization_factor (i32): Upscaling factor. Quantized reps = (reps * quantization_factor).floor()
    ///     convert_id_to_token (bool): True - Return token str; False - Return token_id str
    ///     allow_negative_values (bool): Whether to preserve negative values.
    ///     negative_prefix (&str): Sparse reps doesn't allow negative values. Thus the entry of negative value
    ///                             will be `{negative_prefix}_{token}`.
    #[pyo3(signature = (reps, quantization_factor=100, convert_id_to_token=false, allow_negative_values=false, negative_prefix="neg_"))]
    pub fn convert_sparse_reps_to_json_f16(
        &self,
        py: Python,
        reps: PyReadonlyArray2<f16>,
        quantization_factor: i32,
        convert_id_to_token: bool,
        allow_negative_values: bool,
        negative_prefix: &str,
    ) -> PyResult<Vec<HashMap<String, i32>>> {
        let reps = reps.as_array();

        if reps.shape().len() != 2 {
            return Err(PyValueError::new_err("Input numpy array must be 2-dimensional."));
        }

        let result = py.allow_threads(|| {
            let quant_factor = quantization_factor as f32;

            (0..reps.shape()[0])
                .into_par_iter()
                .map(|batch_id| {
                    let mut sparse_rep: HashMap<String, i32> = HashMap::new();

                    for (vocab_id, &value) in reps.row(batch_id).indexed_iter() {
                        if value > f16::ZERO || value < f16::NEG_ZERO { // Value is not zero
                            if value < f16::NEG_ZERO && !allow_negative_values {
                                continue;
                            }

                            let value = value.to_f32();
                            // Quantize it
                            let quantized_value = (value * quant_factor).floor() as i32;

                            let token_key = if convert_id_to_token {
                                                        self.vocab_dict
                                                            .get(&(vocab_id as i32))
                                                            .ok_or_else(|| PyErr::new::<PyKeyError, _>(
                                                                format!("Token id {} not found in vocab_dict.", vocab_id)))?
                                                            .clone()
                                                    } else {
                                                        vocab_id.to_string()
                                                    };

                            if quantized_value > 0 {
                                sparse_rep.insert(token_key, quantized_value);
                            } else if quantized_value < 0 && allow_negative_values {
                                sparse_rep.insert(negative_prefix.to_string() + &token_key, -quantized_value);
                            }
                        }
                    }

                    if sparse_rep.is_empty() {
                        let pad_key = if convert_id_to_token {
                                        "[PAD]".to_string()
                                    } else {
                                        "-1".to_string()
                                    };
                        sparse_rep.insert(pad_key,1);
                    }

                    Ok(sparse_rep)
                })
                .collect::<PyResult<Vec<_>>>() // Collect into Vec<HashMap<String, i32>>
        });

        result
    }



    /// Same as `self.convert_sparse_reps_to_pseudo_text_f32`
    /// A float32 multi-threaded version of Convert sparse representations to quantized pseudo text.
    /// Format: `token1 token1 ... token1 token2 token2 ... token2 ...`, each `tokenx` will be repeated `frequency times`.
    /// 
    /// ### Args:
    /// 
    ///     reps (PyReadonlyArray2<f32>): Numpy f32 array, shape [batch_size, vocab_dim]
    ///     quantization_factor (i32): Upscaling factor. Quantized reps = (reps * quantization_factor).floor()
    ///     convert_id_to_token (bool): True - Return token str; False - Return token_id str
    ///     allow_negative_values (bool): Whether to preserve negative values.
    ///     negative_prefix (&str): Sparse reps doesn't allow negative values. Thus the entry of negative value
    ///                             will be `{negative_prefix}_{token}`.
    #[pyo3(signature = (reps, quantization_factor=100, convert_id_to_token=false, allow_negative_values=false, negative_prefix="neg_"))]
    pub fn convert_sparse_reps_to_pseudo_text(
        &self,
        py: Python,
        reps: PyReadonlyArray2<f32>,
        quantization_factor: i32,
        convert_id_to_token: bool,
        allow_negative_values: bool,
        negative_prefix: &str,
    ) -> PyResult<Vec<String>> {
        self.convert_sparse_reps_to_pseudo_text_f32(py, reps, quantization_factor, convert_id_to_token, allow_negative_values, negative_prefix)
    }

    /// A float32 multi-threaded version of Convert sparse representations to quantized pseudo text.
    /// Format: `token1 token1 ... token1 token2 token2 ... token2 ...`, each `tokenx` will be repeated `frequency times`.
    /// 
    /// ### Args:
    /// 
    ///     reps (PyReadonlyArray2<f32>): Numpy f32 array, shape [batch_size, vocab_dim]
    ///     quantization_factor (i32): Upscaling factor. Quantized reps = (reps * quantization_factor).floor()
    ///     convert_id_to_token (bool): True - Return token str; False - Return token_id str
    ///     allow_negative_values (bool): Whether to preserve negative values.
    ///     negative_prefix (&str): Sparse reps doesn't allow negative values. Thus the entry of negative value
    ///                             will be `{negative_prefix}_{token}`.
    #[pyo3(signature = (reps, quantization_factor=100, convert_id_to_token=false, allow_negative_values=false, negative_prefix="neg_"))]
    pub fn convert_sparse_reps_to_pseudo_text_f32(
        &self,
        py: Python,
        reps: PyReadonlyArray2<f32>,
        quantization_factor: i32,
        convert_id_to_token: bool,
        allow_negative_values: bool,
        negative_prefix: &str,
    ) -> PyResult<Vec<String>> {
        let json_reps = self.convert_sparse_reps_to_json_f32(py, reps, quantization_factor, convert_id_to_token, allow_negative_values, negative_prefix)?;
        let result = self.convert_json_reps_to_pseudo_text(py, json_reps); // Create pseudo text format
        Ok(result)
    }

    /// A float16 multi-threaded version of Convert sparse representations to quantized pseudo text.
    /// Format: `token1 token1 ... token1 token2 token2 ... token2 ...`, each `tokenx` will be repeated `frequency times`.
    /// 
    /// ### Args:
    /// 
    ///     reps (PyReadonlyArray2<f16>): Numpy f16 array, shape [batch_size, vocab_dim]
    ///     quantization_factor (i32): Upscaling factor. Quantized reps = (reps * quantization_factor).floor()
    ///     convert_id_to_token (bool): True - Return token str; False - Return token_id str
    ///     allow_negative_values (bool): Whether to preserve negative values.
    ///     negative_prefix (&str): Sparse reps doesn't allow negative values. Thus the entry of negative value
    ///                             will be `{negative_prefix}_{token}`.
    #[pyo3(signature = (reps, quantization_factor=100, convert_id_to_token=false, allow_negative_values=false, negative_prefix="neg_"))]
    pub fn convert_sparse_reps_to_pseudo_text_f16(
        &self,
        py: Python,
        reps: PyReadonlyArray2<f16>,
        quantization_factor: i32,
        convert_id_to_token: bool,
        allow_negative_values: bool,
        negative_prefix: &str,
    ) -> PyResult<Vec<String>> {
        let json_reps = self.convert_sparse_reps_to_json_f16(py, reps, quantization_factor, convert_id_to_token, allow_negative_values, negative_prefix)?;
        let result = self.convert_json_reps_to_pseudo_text(py, json_reps); // Create pseudo text format
        Ok(result)
    }

    /// Convert json reps to pseudo text
    /// 
    /// ### Args:
    ///     
    ///     json_reps (Vec<HashMap<String, i32>>): Format: `{token_id / token: int frequency}`, all keys are `str`.
    /// 
    pub fn convert_json_reps_to_pseudo_text(
        &self,
        py: Python,
        json_reps: Vec<HashMap<String, i32>>
    ) -> Vec<String> {
        py.allow_threads(|| {
            json_reps
                .into_par_iter() // Parallelize over batch
                .map(|dict_rep| {
                    // Pre-allocate a String with a reasonable capacity to reduce reallocations
                    // Adjust capacity based on expected token lengths
                    let mut text_rep = String::with_capacity(dict_rep.len() * 100);
    
                    for (token, freq) in dict_rep {
                        if freq > 0 {
                            for _ in 0..freq {
                                if !text_rep.is_empty() {
                                    text_rep.push(' '); // Add a space before each token (except the first one)
                                }
                                text_rep.push_str(&token);
                            }
                        }
                    }
                    text_rep
                })
                .collect::<Vec<String>>() // Collect into Vec<String>
        })
    }
}