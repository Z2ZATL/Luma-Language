#ifndef LUMA_H
#define LUMA_H

#ifdef __cplusplus
extern "C" {
#endif

// Error codes
#define LUMA_SUCCESS 0
#define LUMA_ERROR -1

// Initialize Luma runtime
void luma_init();

// Load a dataset (returns dataset_id, -1 on error)
int luma_load_dataset(const char* path, const char* name, int lazy);

// Create a model (returns model_id, -1 on error)
int luma_create_model(const char* model_type);

// Train the model (returns 0 on success, -1 on error)
int luma_train(int model_id, int epochs, int batch_size, float learning_rate);

// Evaluate the model (returns 0 on success, -1 on error)
int luma_evaluate(int model_id, const char* metrics);

// Save the model (returns 0 on success, -1 on error)
int luma_save_model(int model_id, const char* path);

// Free resources
void luma_cleanup();

#ifdef __cplusplus
}
#endif

#endif // LUMA_H