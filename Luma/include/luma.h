#ifndef LUMA_H
#define LUMA_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version information
#define LUMA_VERSION_MAJOR 1
#define LUMA_VERSION_MINOR 0
#define LUMA_VERSION_PATCH 0
#define LUMA_VERSION "1.0.0"

// Error codes
#define LUMA_SUCCESS 0
#define LUMA_ERROR -1
#define LUMA_ERROR_INVALID_INPUT -2
#define LUMA_ERROR_OUT_OF_MEMORY -3
#define LUMA_ERROR_NOT_IMPLEMENTED -4
#define LUMA_ERROR_CUDA_ERROR -5
#define LUMA_ERROR_FILE_NOT_FOUND -6
#define LUMA_ERROR_INVALID_MODEL -7

// Data types
typedef struct LumaTensor LumaTensor;
typedef struct LumaModel LumaModel;
typedef struct LumaDataset LumaDataset;
typedef struct LumaOptimizer LumaOptimizer;

// Build configuration detection
typedef enum {
    LUMA_BUILD_NATIVE = 0,
    LUMA_BUILD_WASM = 1,
    LUMA_BUILD_COLAB = 2
} LumaBuildType;

// Device types
typedef enum {
    LUMA_DEVICE_CPU = 0,
    LUMA_DEVICE_CUDA = 1,
    LUMA_DEVICE_OPENCL = 2
} LumaDeviceType;

// Tensor operations
typedef enum {
    LUMA_DTYPE_FLOAT32 = 0,
    LUMA_DTYPE_FLOAT64 = 1,
    LUMA_DTYPE_INT32 = 2,
    LUMA_DTYPE_INT64 = 3
} LumaDataType;

// === Core Runtime Functions ===

// Initialize Luma runtime with build type detection
int luma_init(void);

// Get version information
const char* luma_get_version(void);

// Get build type
LumaBuildType luma_get_build_type(void);

// Cleanup all resources
void luma_cleanup(void);

// === Tensor Operations ===

// Create a new tensor
LumaTensor* luma_tensor_create(const float* data, const int64_t* shape, int ndim, LumaDataType dtype);

// Create tensor with zeros
LumaTensor* luma_tensor_zeros(const int64_t* shape, int ndim, LumaDataType dtype);

// Create tensor with ones
LumaTensor* luma_tensor_ones(const int64_t* shape, int ndim, LumaDataType dtype);

// Free tensor memory
void luma_tensor_free(LumaTensor* tensor);

// Get tensor data pointer
const float* luma_tensor_get_data(const LumaTensor* tensor);

// Get tensor shape
const int64_t* luma_tensor_get_shape(const LumaTensor* tensor);

// Get tensor dimensions
int luma_tensor_get_ndim(const LumaTensor* tensor);

// Copy tensor
LumaTensor* luma_tensor_copy(const LumaTensor* tensor);

// === Dataset Operations ===

// Load dataset from file (returns dataset handle, NULL on error)
LumaDataset* luma_dataset_load(const char* path, const char* name, int lazy_loading);

// Load dataset from memory
LumaDataset* luma_dataset_from_memory(const float* data, const int64_t* labels, 
                                      int num_samples, int num_features);

// Get dataset size
int luma_dataset_get_size(const LumaDataset* dataset);

// Get dataset features count
int luma_dataset_get_num_features(const LumaDataset* dataset);

// Get batch from dataset
int luma_dataset_get_batch(const LumaDataset* dataset, int start_idx, int batch_size,
                          LumaTensor** batch_data, LumaTensor** batch_labels);

// Free dataset memory
void luma_dataset_free(LumaDataset* dataset);

// === Model Operations ===

// Create a neural network model
LumaModel* luma_model_create_neural_network(const char* architecture);

// Create model from configuration
LumaModel* luma_model_create_from_config(const char* config_json);

// Load model from file
LumaModel* luma_model_load(const char* path);

// Save model to file
int luma_model_save(const LumaModel* model, const char* path);

// Forward pass
LumaTensor* luma_model_forward(LumaModel* model, const LumaTensor* input);

// Set model to training mode
void luma_model_train_mode(LumaModel* model);

// Set model to evaluation mode
void luma_model_eval_mode(LumaModel* model);

// Free model memory
void luma_model_free(LumaModel* model);

// === Training Operations ===

// Create optimizer
LumaOptimizer* luma_optimizer_create(const char* optimizer_type, float learning_rate);

// Train model for one epoch
int luma_train_epoch(LumaModel* model, LumaOptimizer* optimizer, 
                     const LumaDataset* dataset, int batch_size);

// Train model for multiple epochs
int luma_train_model(LumaModel* model, LumaOptimizer* optimizer,
                     const LumaDataset* train_dataset, const LumaDataset* val_dataset,
                     int epochs, int batch_size);

// Evaluate model
float luma_evaluate_model(LumaModel* model, const LumaDataset* test_dataset, const char* metric);

// Free optimizer memory
void luma_optimizer_free(LumaOptimizer* optimizer);

// === Export Operations ===

// Export model to ONNX format
int luma_export_onnx(const LumaModel* model, const char* output_path);

// Export model to TensorFlow format
int luma_export_tensorflow(const LumaModel* model, const char* output_path);

// Export model to WebAssembly
int luma_export_wasm(const LumaModel* model, const char* output_path);

// Export model to JSON format
int luma_export_json(const LumaModel* model, const char* output_path);

// === Device Management ===

// Set device for computations
int luma_set_device(LumaDeviceType device_type, int device_id);

// Get current device
LumaDeviceType luma_get_current_device(void);

// Check if CUDA is available
int luma_cuda_is_available(void);

// Get number of CUDA devices
int luma_cuda_device_count(void);

// === Utility Functions ===

// Get last error message
const char* luma_get_last_error(void);

// Set logging level (0=off, 1=error, 2=warn, 3=info, 4=debug)
void luma_set_log_level(int level);

// Log message
void luma_log(int level, const char* message);

// === Platform-specific Functions ===

#ifdef LUMA_BUILD_WASM
// WebAssembly specific functions
int luma_wasm_init(void);
void luma_wasm_set_memory_limit(size_t limit_mb);
#endif

#ifdef LUMA_BUILD_COLAB
// Google Colab specific functions
int luma_colab_init(void);
int luma_colab_display_model(const LumaModel* model);
int luma_colab_plot_training_history(const float* losses, int num_epochs);
#endif

#ifdef LUMA_BUILD_NATIVE
// Native build specific functions
int luma_native_enable_openmp(void);
int luma_native_set_thread_count(int num_threads);
#endif

// === Deprecated Functions (for backward compatibility) ===

// Legacy dataset loading (deprecated, use luma_dataset_load instead)
int luma_load_dataset(const char* path, const char* name, int lazy) __attribute__((deprecated));

// Legacy model creation (deprecated, use luma_model_create_neural_network instead)  
int luma_create_model(const char* model_type) __attribute__((deprecated));

// Legacy training function (deprecated, use luma_train_model instead)
int luma_train(int model_id, int epochs, int batch_size, float learning_rate) __attribute__((deprecated));

// Legacy evaluation function (deprecated, use luma_evaluate_model instead)
int luma_evaluate(int model_id, const char* metrics) __attribute__((deprecated));

// Legacy save function (deprecated, use luma_model_save instead)
int luma_save_model(int model_id, const char* path) __attribute__((deprecated));

#ifdef __cplusplus
}
#endif

#endif // LUMA_H