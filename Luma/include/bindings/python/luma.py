"""
Luma AI Framework - Python Bindings

This module provides Python bindings for the Luma AI/ML framework.
"""

import os
import sys
import ctypes
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np

# Load the Luma shared library
def _load_luma_library():
    """Load the Luma shared library."""
    if sys.platform.startswith('linux'):
        lib_name = 'libluma.so'
    elif sys.platform.startswith('darwin'):
        lib_name = 'libluma.dylib'
    elif sys.platform.startswith('win'):
        lib_name = 'luma.dll'
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")
    
    # Try to find the library in standard locations
    lib_paths = [
        # Current directory
        os.path.join(os.path.dirname(os.path.abspath(__file__)), lib_name),
        # Parent directories
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib', lib_name),
        # System paths
        os.path.join('/usr/lib', lib_name),
        os.path.join('/usr/local/lib', lib_name),
    ]
    
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            try:
                return ctypes.CDLL(lib_path)
            except Exception as e:
                print(f"Failed to load {lib_path}: {e}")
    
    raise RuntimeError(f"Could not find Luma library ({lib_name})")

# Define Luma error codes
LUMA_STATUS_SUCCESS = 0
LUMA_STATUS_ERROR_INVALID_ARGUMENT = 1
LUMA_STATUS_ERROR_RUNTIME = 2
LUMA_STATUS_ERROR_NOT_IMPLEMENTED = 3
LUMA_STATUS_ERROR_MEMORY = 4

class LumaError(Exception):
    """Exception raised for Luma library errors."""
    pass

def _check_status(status):
    """Check the status returned by a Luma C API function."""
    if status != LUMA_STATUS_SUCCESS:
        error_msg = _lib.luma_get_last_error()
        if error_msg:
            error_msg = ctypes.cast(error_msg, ctypes.c_char_p).value.decode('utf-8')
        else:
            error_msg = f"Unknown error (code: {status})"
        raise LumaError(error_msg)

# Try to load the library
try:
    _lib = _load_luma_library()
except Exception as e:
    print(f"Warning: Failed to load Luma library: {e}")
    print("Luma will run in mock mode with limited functionality")
    _lib = None

class Tensor:
    """Represents a multi-dimensional array in Luma."""
    
    def __init__(self, data=None, shape=None):
        """Initialize a new Tensor."""
        self._handle = None
        
        if _lib is None:
            # Mock mode
            self._data = np.array(data if data is not None else [], dtype=np.float32)
            return
        
        if data is not None:
            # Convert data to numpy array if it's not already
            if not isinstance(data, np.ndarray):
                data = np.array(data, dtype=np.float32)
            else:
                data = data.astype(np.float32)
            
            # Create tensor from numpy array
            shape_array = (ctypes.c_int64 * len(data.shape))(*data.shape)
            handle = ctypes.c_void_p()
            
            # Create a contiguous array in memory
            data_contiguous = np.ascontiguousarray(data, dtype=np.float32)
            
            # Pass the data to the C API
            status = _lib.luma_tensor_create(
                data_contiguous.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                shape_array,
                len(data.shape),
                ctypes.byref(handle)
            )
            _check_status(status)
            self._handle = handle
        elif shape is not None:
            # Create empty tensor with the given shape
            shape_array = (ctypes.c_int64 * len(shape))(*shape)
            handle = ctypes.c_void_p()
            
            # Create a zero-filled array
            data = np.zeros(shape, dtype=np.float32)
            
            # Pass the data to the C API
            status = _lib.luma_tensor_create(
                data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                shape_array,
                len(shape),
                ctypes.byref(handle)
            )
            _check_status(status)
            self._handle = handle
    
    def __del__(self):
        """Clean up the tensor."""
        if _lib is not None and self._handle is not None:
            _lib.luma_tensor_free(self._handle)
    
    @property
    def shape(self):
        """Get the shape of the tensor."""
        # This would call into the C API to get the tensor shape
        # For now, we'll return a mock shape
        return (1, 1) if _lib is None else (1, 1)  # Placeholder
    
    @property
    def size(self):
        """Get the total size of the tensor."""
        return np.prod(self.shape)

class Model:
    """Represents a Luma AI model."""
    
    def __init__(self, id=None, path=None):
        """Initialize a new Model."""
        self._handle = None
        
        if _lib is None:
            # Mock mode
            self.id = id or "mock_model"
            return
        
        handle = ctypes.c_void_p()
        
        if path is not None:
            # Load model from file
            status = _lib.luma_model_load(
                path.encode('utf-8'),
                ctypes.byref(handle)
            )
            _check_status(status)
        elif id is not None:
            # Create a new model
            status = _lib.luma_model_create(
                id.encode('utf-8'),
                ctypes.byref(handle)
            )
            _check_status(status)
        else:
            raise ValueError("Either id or path must be provided")
        
        self._handle = handle
    
    def __del__(self):
        """Clean up the model."""
        if _lib is not None and self._handle is not None:
            _lib.luma_model_free(self._handle)
    
    def save(self, path):
        """Save the model to a file."""
        if _lib is None:
            print(f"Mock: Saving model to {path}")
            return
        
        status = _lib.luma_model_save(self._handle, path.encode('utf-8'))
        _check_status(status)
    
    def predict(self, input_tensor):
        """Run inference on an input tensor."""
        if _lib is None:
            print("Mock: Running inference")
            return Tensor(np.zeros((1, 10), dtype=np.float32))
        
        output_handle = ctypes.c_void_p()
        status = _lib.luma_model_predict(self._handle, input_tensor._handle, ctypes.byref(output_handle))
        _check_status(status)
        
        # Wrap the output handle in a Python tensor
        output_tensor = Tensor()
        output_tensor._handle = output_handle
        return output_tensor
    
    def export(self, format, path):
        """Export the model to a specific format."""
        if _lib is None:
            print(f"Mock: Exporting model to {format} format at {path}")
            return
        
        status = _lib.luma_model_export(self._handle, format.encode('utf-8'), path.encode('utf-8'))
        _check_status(status)

# Initialize the library if available
if _lib is not None:
    _lib.luma_get_last_error.restype = ctypes.c_char_p
    
    # Initialize the Luma framework
    status = _lib.luma_initialize()
    try:
        _check_status(status)
    except LumaError as e:
        print(f"Warning: Failed to initialize Luma: {e}")
        print("Luma will run in mock mode with limited functionality")
        _lib = None

# Ensure the library is shutdown when the Python process exits
import atexit

@atexit.register
def _shutdown_luma():
    if _lib is not None:
        _lib.luma_shutdown()

# Module exports
__all__ = ['Tensor', 'Model', 'LumaError']