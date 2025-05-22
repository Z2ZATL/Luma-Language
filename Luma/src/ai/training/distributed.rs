#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub fn parallel_process(data: &[f64]) -> Vec<f64> {
    #[cfg(feature = "parallel")]
    {
        data.par_iter().map(|x| *x * 2.0).collect() // Parallel computation
    }
    #[cfg(not(feature = "parallel"))]
    {
        data.iter().map(|x| *x * 2.0).collect() // Sequential computation
    }
}

/// Distributes training across multiple threads
pub fn distributed_train(data: &[f64], labels: &[f64]) -> Result<Vec<f64>, String> {
    if data.len() != labels.len() {
        return Err("Data and labels must have the same length".to_string());
    }

    #[cfg(feature = "parallel")]
    {
        let weights: Vec<f64> = data
            .par_iter()
            .zip(labels.par_iter())
            .map(|(x, y)| x * y) // Simple weight update
            .collect();
        Ok(weights)
    }
    #[cfg(not(feature = "parallel"))]
    {
        let weights: Vec<f64> = data
            .iter()
            .zip(labels.iter())
            .map(|(x, y)| x * y) // Simple weight update
            .collect();
        Ok(weights)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_process() {
        let data = vec![1.0, 2.0, 3.0];
        let result = parallel_process(&data);
        assert_eq!(result, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_distributed_train() {
        let data = vec![1.0, 2.0, 3.0];
        let labels = vec![2.0, 4.0, 6.0];
        let result = distributed_train(&data, &labels).unwrap();
        assert_eq!(result, vec![2.0, 8.0, 18.0]);
    }
}