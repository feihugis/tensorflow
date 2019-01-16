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

#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/common_runtime/gpu/gpu_managed_allocator.h"
#endif

#include "dataset_test_base.h"

namespace tensorflow {
namespace data {

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

Status DatasetOpsTestBase::InitDatasetFromContext(OpKernelContext* context,
                                                  int output_index,
                                                  DatasetBase** dataset) {
  Status status;
  Tensor* output = GetOutput(context, output_index, &status);
  TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(*output, dataset));
  (*dataset)->Ref();
  return status;
}

Tensor* DatasetOpsTestBase::GetOutput(OpKernelContext* context,
                                      int output_index, Status* status) {
  if (output_index >= context->num_outputs()) {
    *status = errors::InvalidArgument(
        "The 'output_index' should be less than the output number( ",
        context->num_outputs(), ") but got: ", output_index);
  }
  Tensor* output = context->mutable_output(output_index);
#ifdef GOOGLE_CUDA
  if (device_type_ == DEVICE_GPU) {
    managed_outputs_.resize(context_->num_outputs());
    // Copy the output tensor to managed memory if we haven't done so.
    if (!managed_outputs_[output_index]) {
      Tensor* managed_output =
          new Tensor(allocator(), output->dtype(), output->shape());
      auto src = output->tensor_data();
      auto dst = managed_output->tensor_data();
      context_->eigen_gpu_device().memcpy(const_cast<char*>(dst.data()),
                                          src.data(), src.size());
      context_->eigen_gpu_device().synchronize();
      managed_outputs_[output_index] = managed_output;
    }
    output = managed_outputs_[output_index];
  }
#endif
  return output;
}

Status DatasetOpsTestBase::InitOpKernelContext(
    OpKernel* kernel, gtl::InlinedVector<TensorValue, 4>* inputs,
    std::unique_ptr<OpKernelContext>* context) {
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
  *context = MakeUnique<OpKernelContext>(params_.get());
  return Status::OK();
}

Status DatasetOpsTestBase::RunOpKernel(OpKernel* op_kernel,
                                       OpKernelContext* context) {
  device_->Compute(op_kernel, context);
  return context->status();
}

void DatasetOpsTestBase::InitSerializationContext(
    std::unique_ptr<SerializationContext>& serialization_context_) {
  SerializationContext::Params params;
  params.flib_def = lib_def_.get();
  serialization_context_ = MakeUnique<SerializationContext>(params);
}

Status DatasetOpsTestBase::MakeOpKernel(NodeDef node_def,
                                        std::unique_ptr<OpKernel>& kernel) {
  Status status;
  kernel = CreateOpKernel(device_type_, device_.get(), allocator(), node_def,
                          TF_GRAPH_DEF_VERSION, &status);
  TF_RETURN_IF_ERROR(status);
  return status;
}

Tensor* DatasetOpsTestBase::AddDatasetInput(
    gtl::InlinedVector<TensorValue, 4>* inputs, DataTypeVector input_types,
    DataType dtype, const TensorShape& shape) {
  CHECK_GT(input_types.size(), inputs->size())
      << "Adding more inputs than types; perhaps you need to call MakeOp";
  bool is_ref = IsRefType(input_types[inputs->size()]);
  Tensor* input = new Tensor(allocator(), dtype, shape);
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

Status DatasetOpsTestBase::MakeDataset(OpKernel* kernel,
                                       OpKernelContext* context) {
  device_->Compute(kernel, context);
  return context->status();
}

Status DatasetOpsTestBase::MakeIterator(
    OpKernelContext* context, DatasetBase* dataset,
    std::unique_ptr<IteratorContext>* iterator_context,
    std::unique_ptr<IteratorBase>* iterator) {
  *iterator_context = MakeUnique<IteratorContext>(context);
  IteratorContext::Params params(iterator_context->get());
  params.function_handle_cache = new FunctionHandleCache(flr_);
  iterator_context->reset(new IteratorContext(params));
  return dataset->MakeIterator(iterator_context->get(), "Iterator", iterator);
}

Status DatasetOpsTestBase::GetNext(IteratorBase* iterator,
                                   IteratorContext* iterator_context,
                                   std::vector<Tensor>* out_tensors,
                                   bool* end_of_sequence) {
  return iterator->GetNext(iterator_context, out_tensors, end_of_sequence);
}

Allocator* DatasetOpsTestBase::allocator() { return allocator_; }

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

Status DatasetOpsTestBase::CheckOpKernelInput(
    const OpKernel& kernel, const gtl::InlinedVector<TensorValue, 4>& inputs) {
  if (kernel.input_types().size() != inputs.size()) {
    return errors::Internal("The number of input elements should be ",
                            kernel.input_types().size(),
                            ", but got: ", inputs.size());
  }
  return Status::OK();
}

}  // namespace data
}  // namespace tensorflow
