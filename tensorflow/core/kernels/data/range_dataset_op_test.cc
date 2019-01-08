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

#include <tuple>
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace {

class RangeDatasetOpTest : public OpsTestBase {
 public:
  ~RangeDatasetOpTest() override {
    gtl::STLDeleteElements(&tensors_);
    gtl::STLDeleteElements(&managed_outputs_);
    dataset_->Unref();
  }

  Status InitOp() {
    Status status;
    kernel_ = CreateOpKernel(device_type_, device_.get(), allocator(),
                             node_def_, TF_GRAPH_DEF_VERSION, &status);
    TF_RETURN_IF_ERROR(status);
    if (kernel_ != nullptr) input_types_ = kernel_->input_types();
    return status;
  }

  Status InitThreadPool(int thread_num) {
    if (thread_num < 1) {
      return errors::InvalidArgument(
          "The `thread_num` argument should be but got: ", thread_num);
    }
    thread_pool_ = MakeUnique<thread::ThreadPool>(
        Env::Default(), ThreadOptions(), "inter_op", thread_num);
    return Status::OK();
  }

  Status InitFunctionLibraryRuntime(const std::vector<FunctionDef>& flib,
                                    int cpu_num) {
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
    lib_def_ =
        MakeUnique<FunctionLibraryDefinition>(OpRegistry::Global(), proto);

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

  Status InitDatasetFromContext(int output_index) {
    auto* tensor = GetOutput(output_index);
    TF_RETURN_IF_ERROR(GetDatasetFromVariantTensor(*tensor, &dataset_));
    dataset_->Ref();
    return Status::OK();
  }

  Status InitSerializationContext() {
    SerializationContext::Params params;
    params.flib_def = lib_def_.get();
    serialization_context_ = MakeUnique<SerializationContext>(params);
    return Status::OK();
  }

  Status RunOpKernel() {
    // Make sure the old OpKernelContext is deleted before the Params it was
    // using.
    context_.reset();
    params_ = MakeUnique<OpKernelContext::Params>();
    params_->device = device_.get();
    params_->frame_iter = FrameAndIter(0, 0);
    params_->inputs = &inputs_;
    params_->op_kernel = kernel_.get();
    params_->function_library = flr_;
    params_->runner = &runner_;
    step_container_ = MakeUnique<ScopedStepContainer>(0, [](const string&) {});
    params_->step_container = step_container_.get();
    checkpoint::TensorSliceReaderCacheWrapper slice_reader_cache_wrapper;
    params_->slice_reader_cache = &slice_reader_cache_wrapper;
    params_->resource_manager = device_.get()->resource_manager();
    std::vector<AllocatorAttributes> attrs;
    test::SetOutputAttrs(params_.get(), &attrs);

    context_ = MakeUnique<OpKernelContext>(params_.get());
    device_->Compute(kernel_.get(), context_.get());
    return context_->status();
  }

 protected:
  template <typename T>
  Status MakeOpDef() {
    DataType value_type = tensorflow::DataTypeToEnum<T>::value;
    std::vector<PartialTensorShape> shapes({{}});
    DataTypeVector dtypes({value_type});

    TF_RETURN_IF_ERROR(NodeDefBuilder(kNodeName, kOpName)
                           .Input(FakeInput(DT_INT64))
                           .Input(FakeInput(DT_INT64))
                           .Input(FakeInput(DT_INT64))
                           .Attr("output_types", dtypes)
                           .Attr("output_shapes", shapes)
                           .Finalize(node_def()));
    TF_RETURN_IF_ERROR(InitOp());
    return Status::OK();
  }

  Status MakeDataset(int64 start, int64 end, int64 step, int output_index,
                     int thread_num, int cpu_num) {
    // Clear inputs_ to enable the repeated update of dataset inputs.
    inputs_.clear();
    AddInputFromArray<int64>(TensorShape({}), {start});
    AddInputFromArray<int64>(TensorShape({}), {end});
    AddInputFromArray<int64>(TensorShape({}), {step});

    TF_RETURN_IF_ERROR(InitThreadPool(thread_num));
    TF_RETURN_IF_ERROR(InitFunctionLibraryRuntime({}, cpu_num));
    TF_RETURN_IF_ERROR(RunOpKernel());
    TF_RETURN_IF_ERROR(InitDatasetFromContext(output_index));
    return Status::OK();
  }

  Status MakeIterator() {
    iterator_context_.reset();
    iterator_context_ = MakeUnique<IteratorContext>(context_.get());
    iterator_.reset();
    return dataset_->MakeIterator(iterator_context_.get(), "Iterator",
                                  &iterator_);
  }

  Status GetNext(std::vector<Tensor>* out_tensors, bool* end_of_sequence) {
    return iterator_->GetNext(iterator_context_.get(), out_tensors,
                              end_of_sequence);
  }

  const string& name() const { return dataset_->name(); }

 protected:
  DatasetBase* dataset_;
  FunctionLibraryRuntime* flr_;  // not owned
  std::function<void(std::function<void()>)> runner_;
  std::unique_ptr<DeviceMgr> device_mgr_;
  std::unique_ptr<FunctionLibraryDefinition> lib_def_;
  std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  std::unique_ptr<IteratorContext> iterator_context_;
  std::unique_ptr<IteratorBase> iterator_;
  std::unique_ptr<SerializationContext> serialization_context_;
  static const string kNodeName;
  static const string kOpName;
};

const string RangeDatasetOpTest::kNodeName = "range_dataset";
const string RangeDatasetOpTest::kOpName = "RangeDataset";

struct DatasetGetNextTest
    : RangeDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<int, int, int>> {};

TEST_P(DatasetGetNextTest, GetNext) {
  TF_ASSERT_OK(MakeOpDef<int64>());
  int output_index = 0, thread_num = 2, cpu_num = 4;
  int start = std::get<0>(GetParam());
  int end = std::get<1>(GetParam());
  int step = std::get<2>(GetParam());
  TF_ASSERT_OK(
      MakeDataset(start, end, step, output_index, thread_num, cpu_num));
  TF_ASSERT_OK(MakeIterator());
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(GetNext(&out_tensors, &end_of_sequence));
  }
  std::vector<int64> expected_values;
  for (int i = start; (end - i) * step > 0; i = i + step) {
    expected_values.reserve(1);
    expected_values.emplace_back(i);
  }
  EXPECT_EQ(out_tensors.size(), expected_values.size());
  for (size_t i = 0; i < out_tensors.size(); ++i) {
    int64 actual_value = out_tensors[i].flat<int64>()(0);
    int64 expect_value = expected_values[i];
    EXPECT_EQ(actual_value, expect_value);
  }
}

INSTANTIATE_TEST_CASE_P(RangeDatasetOpTest, DatasetGetNextTest,
                        ::testing::Values(std::make_tuple(0, 10, 1),
                                          std::make_tuple(0, 10, 3),
                                          std::make_tuple(10, 0, -1),
                                          std::make_tuple(10, 0, -3)));

TEST_F(RangeDatasetOpTest, DatasetName) {
  TF_ASSERT_OK(MakeOpDef<int64>());
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 4;
  TF_ASSERT_OK(
      MakeDataset(start, end, step, output_index, thread_num, cpu_num));
  EXPECT_EQ(dataset_->name(), kOpName);
}

TEST_F(RangeDatasetOpTest, DatasetOutputDtypes) {
  TF_ASSERT_OK(MakeOpDef<int64>());
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 4;
  TF_ASSERT_OK(
      MakeDataset(start, end, step, output_index, thread_num, cpu_num));
  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(dataset_->output_dtypes(), expected_dtypes);
}

TEST_F(RangeDatasetOpTest, DatasetOutputShapes) {
  TF_ASSERT_OK(MakeOpDef<int64>());
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 4;
  TF_ASSERT_OK(
      MakeDataset(start, end, step, output_index, thread_num, cpu_num));
  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(dataset_->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < dataset_->output_shapes().size(); ++i) {
    dataset_->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
  }
}

struct DatasetCardinalityTest
    : RangeDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

TEST_P(DatasetCardinalityTest, Cardinality) {
  TF_ASSERT_OK(MakeOpDef<int64>());
  int output_index = 0, thread_num = 2, cpu_num = 4;
  int start = std::get<0>(GetParam());
  int end = std::get<1>(GetParam());
  int step = std::get<2>(GetParam());
  int expected_cardinality = std::get<3>(GetParam());
  TF_ASSERT_OK(
      MakeDataset(start, end, step, output_index, thread_num, cpu_num));
  EXPECT_EQ(dataset_->Cardinality(), expected_cardinality);
}

INSTANTIATE_TEST_CASE_P(RangeDatasetOpTest, DatasetCardinalityTest,
                        ::testing::Values(std::make_tuple(0, 10, 1, 10),
                                          std::make_tuple(0, 10, 3, 4),
                                          std::make_tuple(10, 0, -3, 4)));

TEST_F(RangeDatasetOpTest, DatasetSave) {
  TF_ASSERT_OK(MakeOpDef<int64>());
  int output_index = 0, thread_num = 2, cpu_num = 4;
  int start = 0, end = 10, step = 1;
  TF_ASSERT_OK(
      MakeDataset(start, end, step, output_index, thread_num, cpu_num));
  TF_ASSERT_OK(InitSerializationContext());
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(dataset_->Save(serialization_context_.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_F(RangeDatasetOpTest, IteratorOutputDtypes) {
  TF_ASSERT_OK(MakeOpDef<int64>());
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 4;
  TF_ASSERT_OK(
      MakeDataset(start, end, step, output_index, thread_num, cpu_num));
  TF_ASSERT_OK(MakeIterator());
  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(iterator_->output_dtypes(), expected_dtypes);
}

TEST_F(RangeDatasetOpTest, IteratorOutputShapes) {
  TF_ASSERT_OK(MakeOpDef<int64>());
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 4;
  TF_ASSERT_OK(
      MakeDataset(start, end, step, output_index, thread_num, cpu_num));
  TF_ASSERT_OK(MakeIterator());
  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(iterator_->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < dataset_->output_shapes().size(); ++i) {
    iterator_->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
  }
}

TEST_F(RangeDatasetOpTest, IteratorOutputPrefix) {
  TF_ASSERT_OK(MakeOpDef<int64>());
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 4;
  TF_ASSERT_OK(
      MakeDataset(start, end, step, output_index, thread_num, cpu_num));
  TF_ASSERT_OK(MakeIterator());
  EXPECT_EQ(iterator_->prefix(), "Iterator::Range");
}

struct IteratorRoundtripTest
    : RangeDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

TEST_P(IteratorRoundtripTest, Roundtrip) {
  TF_ASSERT_OK(MakeOpDef<int64>());
  int output_index = 0, thread_num = 2, cpu_num = 4;
  int start = std::get<0>(GetParam());
  int end = std::get<1>(GetParam());
  int step = std::get<2>(GetParam());
  int breakpoint = std::get<3>(GetParam());
  TF_ASSERT_OK(
      MakeDataset(start, end, step, output_index, thread_num, cpu_num));
  TF_ASSERT_OK(MakeIterator());
  std::vector<Tensor> out_tensors;
  bool end_of_sequence = false;
  int cur_val = start - step;
  for (int i = 0; i < breakpoint; i++) {
    if (!end_of_sequence) {
      TF_EXPECT_OK(GetNext(&out_tensors, &end_of_sequence));
      cur_val = ((end - cur_val - step) * step > 0) ? cur_val + step : cur_val;
    }
  }
  TF_ASSERT_OK(InitSerializationContext());
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(iterator_->Save(serialization_context_.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
  VariantTensorDataReader reader(&data);
  TF_ASSERT_OK(iterator_->Restore(iterator_context_.get(), &reader));
  TF_EXPECT_OK(GetNext(&out_tensors, &end_of_sequence));
  int expect_next =
      ((end - cur_val - step) * step > 0) ? cur_val + step : cur_val;
  EXPECT_EQ(out_tensors.back().flat<int64>()(0), expect_next);
}

INSTANTIATE_TEST_CASE_P(
    RangeDatasetOpTest, IteratorRoundtripTest,
    ::testing::Values(
        std::make_tuple(0, 10, 2, 0),    // unused_iterator
        std::make_tuple(0, 10, 2, 4),    // fully_used_iterator_increase
        std::make_tuple(10, 0, -2, 4),   // fully_used_iterator_decrease
        std::make_tuple(0, 10, 2, 6)));  // exhausted_iterator

}  // namespace
}  // namespace data
}  // namespace tensorflow
