/**
 * @file luma.h
 * @brief C bindings for the Luma AI framework
 */

#ifndef LUMA_H
#define LUMA_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Opaque handle to a Luma Model
 */
typedef struct LumaModel* LumaModelHandle;

/**
 * Opaque handle to a Luma Tensor
 */
typedef struct LumaTensor* LumaTensorHandle;

/**
 * Opaque handle to a Luma Computation Graph
 */
typedef struct LumaComputationGraph* LumaGraphHandle;

/**
 * Error codes for Luma C API
 */
typedef enum LumaStatus {
    LUMA_STATUS_SUCCESS = 0,
    LUMA_STATUS_ERROR_INVALID_ARGUMENT = 1,
    LUMA_STATUS_ERROR_RUNTIME = 2,
    LUMA_STATUS_ERROR_NOT_IMPLEMENTED = 3,
    LUMA_STATUS_ERROR_MEMORY = 4
} LumaStatus;

/**
 * Initialize the Luma framework
 * 
 * This function must be called before any other Luma function.
 * 
 * @return LUMA_STATUS_SUCCESS on success, other status code on failure
 */
LumaStatus luma_initialize();

/**
 * Shutdown the Luma framework and free all resources
 * 
 * This function should be called when Luma is no longer needed.
 * 
 * @return LUMA_STATUS_SUCCESS on success, other status code on failure
 */
LumaStatus luma_shutdown();

/**
 * Create a new Luma model
 * 
 * @param id Model identifier
 * @param model Pointer to receive the model handle
 * @return LUMA_STATUS_SUCCESS on success, other status code on failure
 */
LumaStatus luma_model_create(const char* id, LumaModelHandle* model);

/**
 * Load a Luma model from file
 * 
 * @param path Path to the model file
 * @param model Pointer to receive the model handle
 * @return LUMA_STATUS_SUCCESS on success, other status code on failure
 */
LumaStatus luma_model_load(const char* path, LumaModelHandle* model);

/**
 * Save a Luma model to file
 * 
 * @param model The model handle
 * @param path Path where the model should be saved
 * @return LUMA_STATUS_SUCCESS on success, other status code on failure
 */
LumaStatus luma_model_save(LumaModelHandle model, const char* path);

/**
 * Free resources associated with a Luma model
 * 
 * @param model Model handle to free
 * @return LUMA_STATUS_SUCCESS on success, other status code on failure
 */
LumaStatus luma_model_free(LumaModelHandle model);

/**
 * Create a new Luma tensor
 * 
 * @param data Pointer to tensor data
 * @param shape Array of dimension sizes
 * @param ndim Number of dimensions
 * @param tensor Pointer to receive the tensor handle
 * @return LUMA_STATUS_SUCCESS on success, other status code on failure
 */
LumaStatus luma_tensor_create(const float* data, const int64_t* shape, int ndim, LumaTensorHandle* tensor);

/**
 * Free resources associated with a Luma tensor
 * 
 * @param tensor Tensor handle to free
 * @return LUMA_STATUS_SUCCESS on success, other status code on failure
 */
LumaStatus luma_tensor_free(LumaTensorHandle tensor);

/**
 * Run model inference on input tensor
 * 
 * @param model Model handle
 * @param input Input tensor handle
 * @param output Pointer to receive output tensor handle
 * @return LUMA_STATUS_SUCCESS on success, other status code on failure
 */
LumaStatus luma_model_predict(LumaModelHandle model, LumaTensorHandle input, LumaTensorHandle* output);

/**
 * Export model to specific format
 * 
 * @param model Model handle
 * @param format Format to export to ("tensorflow", "pytorch", "onnx", etc.)
 * @param path Path where the exported model should be saved
 * @return LUMA_STATUS_SUCCESS on success, other status code on failure
 */
LumaStatus luma_model_export(LumaModelHandle model, const char* format, const char* path);

/**
 * Get last error message
 * 
 * @return Pointer to null-terminated error message string
 */
const char* luma_get_last_error();

#ifdef __cplusplus
}
#endif

#endif /* LUMA_H */