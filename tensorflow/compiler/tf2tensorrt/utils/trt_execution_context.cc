/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/tf2tensorrt/utils/trt_execution_context.h"
#include "tensorflow/compiler/tf2tensorrt/utils/trt_allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/stream_executor/lib/statusor.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT
#include "third_party/tensorrt/NvInfer.h"

namespace tensorflow {
namespace tensorrt {

using ::stream_executor::port::StatusOr;

class TRTBaseAllocator;

ExecutionContext::~ExecutionContext() {
  if (device_memory_) {
    DCHECK(memory_allocator_) << "Internal error: Device memory with address "
                              << (char*)device_memory_ << "is not freed";
    memory_allocator_->free(device_memory_);
  }
  if (execution_context_) {
    execution_context_->destroy();
  }
}

StatusOr<ExecutionContext> ExecutionContext::Create(
    nvinfer1::ICudaEngine* cuda_engine, TRTBaseAllocator* allocator) {
  void* device_memory = nullptr;
  nvinfer1::IExecutionContext* execution_context;
  if (allocator == nullptr) {
    execution_context = cuda_engine->createExecutionContext();
  } else {
    execution_context =
        cuda_engine->createExecutionContextWithoutDeviceMemory();
    size_t device_memory_size = cuda_engine->getDeviceMemorySize();
    VLOG(2) << "Device memory size for cuda engine " << device_memory_size;

    if (device_memory_size > 0) {
      device_memory = allocator->allocate(device_memory_size,
                                          /*unused alignment=*/0, /*flags=*/0);
      if (device_memory == nullptr) {
        return errors::InvalidArgument(
            "Out of GPU memory when creating execution context");
      }
      execution_context->setDeviceMemory(device_memory);
    }
  }
  return ExecutionContext(allocator, device_memory, execution_context);
}
};  // namespace tensorrt
};  // namespace tensorflow
#endif
