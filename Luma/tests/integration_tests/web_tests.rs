#[cfg(test)]
mod tests {
    use crate::integrations::web;

    #[test]
    fn test_init_web() {
        #[cfg(target_arch = "wasm32")]
        {
            let result = web::luma_init_web();
            assert_eq!(result, 0);
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let result = web::luma_init_web();
            assert_eq!(result, -1);
        }
    }
}