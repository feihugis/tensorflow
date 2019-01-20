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

#ifndef TENSORFLOW_CORE_KERNELS_DATA_DATASET_TEST_BASE_H
#define TENSORFLOW_CORE_KERNELS_DATA_DATASET_TEST_BASE_H

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

// Helpful functions to test dataset kernels.
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

  // Creates a new operation based on the node definition.
  std::unique_ptr<OpKernel> CreateOpKernel(const NodeDef& node_def);

  // Creates a new dataset. Here we assume that the dataset operation has only
  // one output stored in the OpKernelContext.
  DatasetBase* CreateDataset(OpKernel* kernel, OpKernelContext* context);

  // Creates a new iterator for iterating over the range of elements in this
  // dataset. Meanwhile, `iterator_context` will be initialized internally.
  //
  // This method may be called multiple times on the same dataset, and the
  // resulting iterators will have distinct state.
  std::unique_ptr<IteratorBase> CreateIterator(
      OpKernelContext* context, DatasetBase* dataset,
      std::unique_ptr<IteratorContext>* iterator_context);

  // Gets the next output from the range that this iterator is traversing.
  //
  // If at least one output remains in this iterator's range, that
  // output will be stored in `*out_tensors` and `false` will be
  // stored in `*end_of_sequence`.
  //
  // If no more outputs remain in this iterator's range, `true` will
  // be stored in `*end_of_sequence`, and the content of
  // `*out_tensors` will be undefined.
  Status GetNext(IteratorBase* iterator, IteratorContext* iterator_context,
                 std::vector<Tensor>* out_tensors, bool* end_of_sequence);

  // Creates a new range dataset operation.
  template <typename T>
  std::unique_ptr<OpKernel> CreateRangeDatasetOpKernel(StringPiece node_name) {
    DataTypeVector dtypes({tensorflow::DataTypeToEnum<T>::value});
    std::vector<PartialTensorShape> shapes({{}});
    NodeDef node_def = test::function::NDef(
        node_name, "RangeDataset", {"start", "stop", "step"},
        {{"output_types", dtypes}, {"output_shapes", shapes}});
    return CreateOpKernel(node_def);
  }

  // Creates a new range dataset.
  template <typename U>
  DatasetBase* CreateRangeDataset(int64 start, int64 end, int64 step,
                                  StringPiece node_name) {
    std::unique_ptr<OpKernel> range_kernel =
        CreateRangeDatasetOpKernel<U>(node_name);
    std::unique_ptr<OpKernelContext> range_context;
    gtl::InlinedVector<TensorValue, 4> range_inputs;
    AddDatasetInputFromArray<int64>(&range_inputs, range_kernel->input_types(),
                                    TensorShape({}), {start});
    AddDatasetInputFromArray<int64>(&range_inputs, range_kernel->input_types(),
                                    TensorShape({}), {end});
    AddDatasetInputFromArray<int64>(&range_inputs, range_kernel->input_types(),
                                    TensorShape({}), {step});
    range_context = CreateOpKernelContext(range_kernel.get(), &range_inputs);
    TF_CHECK_OK(CheckOpKernelInput(*range_kernel, range_inputs));
    TF_CHECK_OK(RunOpKernel(range_kernel.get(), range_context.get()));
    return GetDatasetFromContext(range_context.get(), 0);
  }

  // Fetches the dataset from the operation context.
  DatasetBase* GetDatasetFromContext(OpKernelContext* context,
                                     int output_index);

 protected:
  // Creates a thread pool for parallel tasks.
  Status InitThreadPool(int thread_num);

  // Initializes the runtime for computing the dataset operation and registers
  // the input function definitions. `InitThreadPool()' needs to be called
  // before this method if we want to run the tasks in parallel.
  Status InitFunctionLibraryRuntime(const std::vector<FunctionDef>& flib,
                                    int cpu_num);

  // Runs an operation producing outputs. The `context` need to be initialized
  // (see `CreateOpKernelContext()` ) before running this method.
  Status RunOpKernel(OpKernel* op_kernel, OpKernelContext* context);

  // Checks that the size of `inputs` matches the requirement of the operation.
  Status CheckOpKernelInput(const OpKernel& kernel,
                            const gtl::InlinedVector<TensorValue, 4>& inputs);

  // Creates a new context for running the dataset operation.
  std::unique_ptr<OpKernelContext> CreateOpKernelContext(
      OpKernel* kernel, gtl::InlinedVector<TensorValue, 4>* inputs);

  // Returns a new serialization context for serializing the dataset and
  // iterator.
  std::unique_ptr<SerializationContext> CreateSerializationContext();

  // Adds an arrayslice of data into the input vector. `input_types` describes
  // the required data type for each input tensor. `shape` and `data` describes
  // the shape and values of the current input tensor.
  template <typename T>
  void AddDatasetInputFromArray(gtl::InlinedVector<TensorValue, 4>* inputs,
                                DataTypeVector input_types,
                                const TensorShape& shape,
                                const gtl::ArraySlice<T>& data) {
    Tensor* input =
        AddDatasetInput(inputs, input_types, DataTypeToEnum<T>::v(), shape);
    test::FillValues<T>(input, data);
  }

 private:
  // Adds an empty tensor to the input vector. The returned pointer can be used
  // to fill in the value for the tensor.
  Tensor* AddDatasetInput(gtl::InlinedVector<TensorValue, 4>* inputs,
                          DataTypeVector input_types, DataType dtype,
                          const TensorShape& shape);

  // Sets the allocator attributes for the operation outputs.
  void SetOutputAttrs();

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

#endif  // TENSORFLOW_CORE_KERNELS_DATA_DATASET_TEST_BASE_H
