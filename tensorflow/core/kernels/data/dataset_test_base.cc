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

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {

std::unique_ptr<OpKernel> DatasetOpsTestBase::CreateOpKernel(
    const NodeDef& node_def) {
  Status status;
  std::unique_ptr<OpKernel> kernel =
      tensorflow::CreateOpKernel(device_type_, device_.get(), allocator_,
                                 node_def, TF_GRAPH_DEF_VERSION, &status);
  TF_CHECK_OK(status) << status;
  return kernel;
}

DatasetBase* DatasetOpsTestBase::CreateDataset(OpKernel* kernel,
                                               OpKernelContext* context) {
  TF_CHECK_OK(RunOpKernel(kernel, context));
  // Assume that DatasetOp has only one output.
  return GetDatasetFromContext(context, 0);
}

std::unique_ptr<IteratorBase> DatasetOpsTestBase::CreateIterator(
    OpKernelContext* context, DatasetBase* dataset,
    std::unique_ptr<IteratorContext>* iterator_context) {
  *iterator_context = MakeUnique<IteratorContext>(context);
  IteratorContext::Params params(iterator_context->get());
  params.function_handle_cache = new FunctionHandleCache(flr_);
  iterator_context->reset(new IteratorContext(params));
  std::unique_ptr<IteratorBase> iterator;
  TF_CHECK_OK(
      dataset->MakeIterator(iterator_context->get(), "Iterator", &iterator));
  return iterator;
}

Status DatasetOpsTestBase::GetNext(IteratorBase* iterator,
                                   IteratorContext* iterator_context,
                                   std::vector<Tensor>* out_tensors,
                                   bool* end_of_sequence) {
  return iterator->GetNext(iterator_context, out_tensors, end_of_sequence);
}

DatasetBase* DatasetOpsTestBase::GetDatasetFromContext(OpKernelContext* context,
                                                       int output_index) {
  DatasetBase* dataset;
  Tensor* output = context->mutable_output(output_index);
  TF_CHECK_OK(GetDatasetFromVariantTensor(*output, &dataset));
  dataset->Ref();
  return dataset;
}

Status DatasetOpsTestBase::InitThreadPool(int thread_num) {
  if (thread_num < 1) {
    return errors::InvalidArgument(
        "The `thread_num` argument should be but got: ", thread_num);
  }
  thread_pool_ = MakeUnique<thread::ThreadPool>(Env::Default(), ThreadOptions(),
                                                "inter_op", thread_num);
  return Status::OK();
}

Status DatasetOpsTestBase::InitFunctionLibraryRuntime(
    const std::vector<FunctionDef>& flib, int cpu_num) {
  if (cpu_num < 1) {
    return errors::InvalidArgument(
        "The `cpu_num` argument should be positive but got: ", cpu_num);
  }
  SessionOptions options;
  auto* device_count = options.config.mutable_device_count();
  device_count->insert({"CPU", cpu_num});
  std::vector<std::unique_ptr<Device>> devices;
  TF_RETURN_IF_ERROR(DeviceFactory::AddDevices(
      options, "/job:localhost/replica:0/task:0", &devices));
  device_mgr_ = MakeUnique<DeviceMgr>(std::move(devices));

  FunctionDefLibrary proto;
  for (const auto& fdef : flib) *(proto.add_function()) = fdef;
  lib_def_ = MakeUnique<FunctionLibraryDefinition>(OpRegistry::Global(), proto);

  OptimizerOptions opts;
  pflr_ = MakeUnique<ProcessFunctionLibraryRuntime>(
      device_mgr_.get(), Env::Default(), TF_GRAPH_DEF_VERSION, lib_def_.get(),
      opts, thread_pool_.get(), nullptr /* cluster_flr */);
  flr_ = pflr_->GetFLR("/job:localhost/replica:0/task:0/cpu:0");
  if (thread_pool_ == nullptr) {
    runner_ = [](std::function<void()> fn) { fn(); };
  } else {
    runner_ = [this](std::function<void()> fn) {
      thread_pool_->Schedule(std::move(fn));
    };
  }
  return Status::OK();
}

Status DatasetOpsTestBase::RunOpKernel(OpKernel* op_kernel,
                                       OpKernelContext* context) {
  device_->Compute(op_kernel, context);
  return context->status();
}

std::unique_ptr<OpKernelContext> DatasetOpsTestBase::CreateOpKernelContext(
    OpKernel* kernel, gtl::InlinedVector<TensorValue, 4>* inputs) {
  params_ = MakeUnique<OpKernelContext::Params>();
  params_->device = device_.get();
  params_->resource_manager = device_->resource_manager();
  params_->frame_iter = FrameAndIter(0, 0);
  params_->inputs = inputs;
  params_->op_kernel = kernel;
  params_->function_library = flr_;
  params_->runner = &runner_;
  step_container_ = MakeUnique<ScopedStepContainer>(0, [](const string&) {});
  params_->step_container = step_container_.get();
  checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
  params_->slice_reader_cache = &slice_reader_cache_wrapper;
  SetOutputAttrs();
  return MakeUnique<OpKernelContext>(params_.get());
}

std::unique_ptr<SerializationContext>
DatasetOpsTestBase::CreateSerializationContext() {
  SerializationContext::Params params;
  params.flib_def = lib_def_.get();
  return MakeUnique<SerializationContext>(params);
}

Status DatasetOpsTestBase::CheckOpKernelInput(
    const OpKernel& kernel, const gtl::InlinedVector<TensorValue, 4>& inputs) {
  if (kernel.input_types().size() != inputs.size()) {
    return errors::Internal("The number of input elements should be ",
                            kernel.input_types().size(),
                            ", but got: ", inputs.size());
  }
  return Status::OK();
}

Tensor* DatasetOpsTestBase::AddDatasetInput(
    gtl::InlinedVector<TensorValue, 4>* inputs, DataTypeVector input_types,
    DataType dtype, const TensorShape& shape) {
  CHECK_GT(input_types.size(), inputs->size())
      << "Adding more inputs than types.";
  bool is_ref = IsRefType(input_types[inputs->size()]);
  Tensor* input = new Tensor(allocator_, dtype, shape);
  tensors_.push_back(input);
  if (is_ref) {
    CHECK_EQ(RemoveRefType(input_types[inputs->size()]), dtype);
    inputs->push_back({&lock_for_refs_, input});
  } else {
    CHECK_EQ(input_types[inputs->size()], dtype);
    inputs->push_back({nullptr, input});
  }
  return input;
}

void DatasetOpsTestBase::SetOutputAttrs() {
  allocator_attrs_.clear();
  for (int index = 0; index < params_->op_kernel->num_outputs(); index++) {
    AllocatorAttributes attr;
    const bool on_host =
        (params_->op_kernel->output_memory_types()[index] == HOST_MEMORY);
    attr.set_on_host(on_host);
    allocator_attrs_.emplace_back(attr);
  }
  params_->output_attr_array = gtl::vector_as_array(&allocator_attrs_);
}

}  // namespace data
}  // namespace tensorflow
