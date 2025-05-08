use crate::ai::engine::tensor::Tensor;
use crate::ai::engine::autodiff::ComputationGraph;

// Element-wise vector operations
pub fn add(a: &[f64], b: &[f64]) -> Vec<f64> {
    if a.len() != b.len() {
        panic!("Vector lengths must match");
    }
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

#[no_mangle]
pub extern "C" fn luma_add_vectors(a: *const f64, b: *const f64, len: i32, out: *mut f64) -> i32 {
    if a.is_null() || b.is_null() || out.is_null() || len <= 0 {
        return -1;
    }
    unsafe {
        let a_slice = std::slice::from_raw_parts(a, len as usize);
        let b_slice = std::slice::from_raw_parts(b, len as usize);
        let result = add(a_slice, b_slice);
        std::ptr::copy_nonoverlapping(result.as_ptr(), out, len as usize);
    }
    0
}

// Tensor operations
pub fn tensor_add(a: &Tensor, b: &Tensor, graph: Option<&mut ComputationGraph>) -> Tensor {
    if a.get_shape() != b.get_shape() {
        panic!("Tensor shapes must match for addition: {:?} vs {:?}", 
               a.get_shape(), b.get_shape());
    }

    let result_data = add(a.get_data(), b.get_data());
    let requires_grad = a.requires_grad() || b.requires_grad();

    let result = if requires_grad {
        Tensor::with_grad(result_data, a.get_shape().to_vec())
    } else {
        Tensor::new(result_data, a.get_shape().to_vec())
    };

    if let Some(graph) = graph {
        if requires_grad {
            graph.add_operation(
                "add", 
                vec![a.clone(), b.clone()], 
                result.clone()
            );
        }
    }

    result
}

pub fn tensor_mul(a: &Tensor, b: &Tensor, graph: Option<&mut ComputationGraph>) -> Tensor {
    if a.get_shape() != b.get_shape() {
        panic!("Tensor shapes must match for element-wise multiplication: {:?} vs {:?}", 
               a.get_shape(), b.get_shape());
    }

    let result_data: Vec<f64> = a.get_data()
        .iter()
        .zip(b.get_data().iter())
        .map(|(x, y)| x * y)
        .collect();

    let requires_grad = a.requires_grad() || b.requires_grad();

    let result = if requires_grad {
        Tensor::with_grad(result_data, a.get_shape().to_vec())
    } else {
        Tensor::new(result_data, a.get_shape().to_vec())
    };

    if let Some(graph) = graph {
        if requires_grad {
            graph.add_operation(
                "mul", 
                vec![a.clone(), b.clone()], 
                result.clone()
            );
        }
    }

    result
}

pub fn tensor_matmul(a: &Tensor, b: &Tensor, graph: Option<&mut ComputationGraph>) -> Tensor {
    if a.get_shape().len() != 2 || b.get_shape().len() != 2 {
        panic!("Matrix multiplication requires 2D tensors");
    }

    let a_shape = a.get_shape();
    let b_shape = b.get_shape();

    if a_shape[1] != b_shape[0] {
        panic!("Incompatible shapes for matrix multiplication: {:?} and {:?}", 
               a_shape, b_shape);
    }

    let m = a_shape[0];
    let n = a_shape[1];
    let p = b_shape[1];

    let mut result_data = vec![0.0; m * p];

    for i in 0..m {
        for j in 0..p {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a.get_data()[i * n + k] * b.get_data()[k * p + j];
            }
            result_data[i * p + j] = sum;
        }
    }

    let result_shape = vec![m, p];
    let requires_grad = a.requires_grad() || b.requires_grad();

    let result = if requires_grad {
        Tensor::with_grad(result_data, result_shape)
    } else {
        Tensor::new(result_data, result_shape)
    };

    if let Some(graph) = graph {
        if requires_grad {
            graph.add_operation(
                "matmul", 
                vec![a.clone(), b.clone()], 
                result.clone()
            );
        }
    }

    result
}

pub fn tensor_sum(a: &Tensor, axis: Option<usize>, graph: Option<&mut ComputationGraph>) -> Tensor {
    let a_shape = a.get_shape();

    match axis {
        Some(dim) => {
            if dim >= a_shape.len() {
                panic!("Sum axis {} is out of bounds for tensor of rank {}", dim, a_shape.len());
            }

            let mut result_shape = a_shape.to_vec();
            result_shape[dim] = 1;

            let total_elements: usize = result_shape.iter().product();
            let mut result_data = vec![0.0; total_elements];

            let stride: usize = a_shape.iter().skip(dim + 1).product();
            let axis_size = a_shape[dim];
            let outer_size: usize = a_shape.iter().take(dim).product();

            for outer in 0..outer_size {
                for inner in 0..stride {
                    let mut sum = 0.0;
                    for i in 0..axis_size {
                        let idx = outer * axis_size * stride + i * stride + inner;
                        sum += a.get_data()[idx];
                    }
                    let result_idx = outer * stride + inner;
                    result_data[result_idx] = sum;
                }
            }

            let result = if a.requires_grad() {
                Tensor::with_grad(result_data, result_shape)
            } else {
                Tensor::new(result_data, result_shape)
            };

            if let Some(graph) = graph {
                if a.requires_grad() {
                    graph.add_operation(
                        "sum", 
                        vec![a.clone()], 
                        result.clone()
                    );
                }
            }

            result
        },
        None => {
            let sum: f64 = a.get_data().iter().sum();
            let result = if a.requires_grad() {
                Tensor::with_grad(vec![sum], vec![1])
            } else {
                Tensor::new(vec![sum], vec![1])
            };

            if let Some(graph) = graph {
                if a.requires_grad() {
                    graph.add_operation(
                        "sum_all", 
                        vec![a.clone()], 
                        result.clone()
                    );
                }
            }

            result
        }
    }
}

pub fn tensor_scalar_mul(a: &Tensor, scalar: f64, graph: Option<&mut ComputationGraph>) -> Tensor {
    let result_data: Vec<f64> = a.get_data().iter().map(|&x| x * scalar).collect();

    let result = if a.requires_grad() {
        Tensor::with_grad(result_data, a.get_shape().to_vec())
    } else {
        Tensor::new(result_data, a.get_shape().to_vec())
    };

    if let Some(graph) = graph {
        if a.requires_grad() {
            let op_id = graph.add_operation(
                "scalar_mul", 
                vec![a.clone()], 
                result.clone()
            );
            graph.set_op_metadata(op_id, "scalar", scalar.to_string());
        }
    }

    result
}