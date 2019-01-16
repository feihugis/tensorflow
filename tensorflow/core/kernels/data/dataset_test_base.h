/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_DATASET_TEST_BASE_H
#define TENSORFLOW_DATASET_TEST_BASE_H

#include <memory>
#include <tuple>
#include <vector>

#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {

class DatasetOpsTestBase : public ::testing::Test {
 public:
  DatasetOpsTestBase()
      : device_(DeviceFactory::NewDevice("CPU", {}, "/job:a/replica:0/task:0")),
        device_type_(DEVICE_CPU) {
    CHECK(device_.get()) << "Could not create CPU device";
    allocator_ = device_->GetAllocator(AllocatorAttributes());
  }

  ~DatasetOpsTestBase() {
    gtl::STLDeleteElements(&tensors_);
    gtl::STLDeleteElements(&managed_outputs_);
  }

  Status InitThreadPool(int thread_num);

  Status InitFunctionLibraryRuntime(const std::vector<FunctionDef>& flib,
                                    int cpu_num);

  Status InitDatasetFromContext(OpKernelContext* context, int output_index,
                                DatasetBase** dataset);

  Tensor* GetOutput(OpKernelContext* context, int output_index, Status* status);

  Status InitOpKernelContext(OpKernel* kernel,
                             gtl::InlinedVector<TensorValue, 4>* inputs,
                             std::unique_ptr<OpKernelContext>* context);

  Status RunOpKernel(OpKernel* op_kernel, OpKernelContext* context);

  void InitSerializationContext(
      std::unique_ptr<SerializationContext>& serialization_context_);

  Status MakeOpKernel(NodeDef node_def, std::unique_ptr<OpKernel>& kernel);

  template <typename T>
  void AddDatasetInputFromArray(gtl::InlinedVector<TensorValue, 4>* inputs,
                                DataTypeVector input_types,
                                const TensorShape& shape,
                                const gtl::ArraySlice<T>& data) {
    Tensor* input =
        AddDatasetInput(inputs, input_types, DataTypeToEnum<T>::v(), shape);
    test::FillValues<T>(input, data);
  }

  Tensor* AddDatasetInput(gtl::InlinedVector<TensorValue, 4>* inputs,
                          DataTypeVector input_types, DataType dtype,
                          const TensorShape& shape);

  Status MakeDataset(OpKernel* kernel, OpKernelContext* context);

  Status MakeIterator(OpKernelContext* context, DatasetBase* dataset,
                      std::unique_ptr<IteratorContext>* iterator_context,
                      std::unique_ptr<IteratorBase>* iterator);

  Status GetNext(IteratorBase* iterator, IteratorContext* iterator_context,
                 std::vector<Tensor>* out_tensors, bool* end_of_sequence);

  Allocator* allocator();

  void SetOutputAttrs();

  Status CheckOpKernelInput(const OpKernel& kernel,
                            const gtl::InlinedVector<TensorValue, 4>& inputs);

 protected:
  std::unique_ptr<Device> device_;
  DeviceType device_type_;
  Allocator* allocator_;
  std::vector<AllocatorAttributes> allocator_attrs_;
  std::unique_ptr<ScopedStepContainer> step_container_;

  FunctionLibraryRuntime* flr_;  // not owned
  std::function<void(std::function<void()>)> runner_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<OpKernelContext::Params> params_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::vector<Tensor*> tensors_;  // owns Tensors.
  // Copies of the outputs in unified memory (host and device accessible).
  std::vector<Tensor*> managed_outputs_;
  mutex lock_for_refs_;  // Used as the Mutex for inputs added as refs
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_DATASET_TEST_BASE_H
