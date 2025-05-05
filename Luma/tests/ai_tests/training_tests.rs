#[cfg(test)]
mod tests {
    use crate::ai::training::trainers;

    #[test]
    fn test_train_model() {
        let data = vec![1.0, 2.0, 3.0];
        let labels = vec![0, 1, 0];
        let mut output = vec![0.0; 3];
        let result = unsafe {
            trainers::luma_train(1, data.as_ptr(), labels.as_ptr(), 3, 10, 0.01, output.as_mut_ptr())
        };
        assert_eq!(result, 0);
    }
}