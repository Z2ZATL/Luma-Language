/// Normalizes a dataset by subtracting the mean and dividing by the standard deviation
pub fn normalize_data(data: &mut [Vec<f64>]) {
    for vec_row in data {
        let sum: f64 = vec_row.iter().sum();
        let mean = sum / vec_row.len() as f64;
        let variance: f64 = vec_row.iter().map(|x| (x - mean).powi(2)).sum();
        let std_dev = (variance / vec_row.len() as f64).sqrt();

        for val in vec_row.iter_mut() {
            if std_dev != 0.0 {
                *val = (*val - mean) / std_dev;
            } else {
                *val = *val - mean; // Avoid division by zero
            }
        }
    }
}

/// Preprocesses a dataset by applying normalization
pub fn preprocess(data: &mut Vec<Vec<f64>>) {
    let data_slice = &mut data[..];
    normalize_data(data_slice);
}

/// Scales data to a specific range (min-max scaling)
pub fn scale_data(data: &mut [Vec<f64>], min: f64, max: f64) {
    for vec_row in data {
        let row_min = *vec_row.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&0.0);
        let row_max = *vec_row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(&1.0);
        let range = row_max - row_min;

        if range != 0.0 {
            for val in vec_row.iter_mut() {
                *val = min + (*val - row_min) * (max - min) / range;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_data() {
        let mut data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        normalize_data(&mut data);
        for row in data {
            let mean: f64 = row.iter().sum::<f64>() / row.len() as f64;
            assert!((mean - 0.0).abs() < 1e-10, "Mean should be close to 0");
        }
    }

    #[test]
    fn test_scale_data() {
        let mut data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        scale_data(&mut data, 0.0, 1.0);
        for row in data {
            let min = row.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            let max = row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            assert!((min - 0.0).abs() < 1e-10, "Min should be 0");
            assert!((max - 1.0).abs() < 1e-10, "Max should be 1");
        }
    }
}