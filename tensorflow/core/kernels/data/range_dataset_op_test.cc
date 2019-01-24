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
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/kernels/data/dataset_test_base.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/ptr_util.h"

namespace tensorflow {
namespace data {
namespace {

class RangeDatasetOpTest : public DatasetOpsTestBase {
 public:
  static const char* kNodeName;
  static const char* kOpName;

  ~RangeDatasetOpTest() override {
    gtl::STLDeleteElements(&tensors_);
    dataset_->Unref();
  }

 protected:
  // Creates a new RangeDataset, meanwhile initializing the operation kernel and
  // context internally. `T` specifies the ouput dtype of RangDataset op kernel.
  template <typename T>
  DatasetBase* CreateRangeDataset(int64 start, int64 end, int64 step) {
    inputs_.clear();
    TF_CHECK_OK(CreateRangeDatasetOpKernel<T>("range", &kernel_));
    TF_CHECK_OK(AddDatasetInputFromArray<int64>(
        &inputs_, kernel_->input_types(), TensorShape({}), {start}));
    TF_CHECK_OK(AddDatasetInputFromArray<int64>(
        &inputs_, kernel_->input_types(), TensorShape({}), {end}));
    TF_CHECK_OK(AddDatasetInputFromArray<int64>(
        &inputs_, kernel_->input_types(), TensorShape({}), {step}));

    TF_CHECK_OK(CreateOpKernelContext(kernel_.get(), &inputs_, &context_));
    TF_CHECK_OK(CheckOpKernelInput(*kernel_, inputs_));
    DatasetBase* dataset;
    TF_CHECK_OK(CreateDataset(kernel_.get(), context_.get(), &dataset));
    return dataset;
  }

 protected:
  DatasetBase* dataset_;
  std::unique_ptr<OpKernel> kernel_;
  std::unique_ptr<OpKernelContext> context_;
  std::unique_ptr<IteratorContext> iterator_context_;
  std::unique_ptr<IteratorBase> iterator_;
  std::unique_ptr<SerializationContext> serialization_context_;

 private:
  gtl::InlinedVector<TensorValue, 4> inputs_;
};

const char* RangeDatasetOpTest::kNodeName = "range_dataset";
const char* RangeDatasetOpTest::kOpName = "RangeDataset";

struct DatasetGetNextTest
    : RangeDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<int64, int64, int64>> {};

TEST_P(DatasetGetNextTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  int64 start = std::get<0>(GetParam());
  int64 end = std::get<1>(GetParam());
  int64 step = std::get<2>(GetParam());

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  dataset_ = CreateRangeDataset<int>(start, end, step);
  TF_ASSERT_OK(CreateIteratorContext(context_.get(), &iterator_context_));
  TF_ASSERT_OK(CreateIterator(iterator_context_.get(), dataset_, &iterator_));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(GetNext(iterator_.get(), iterator_context_.get(), &out_tensors,
                         &end_of_sequence));
  }
  std::vector<int> expected_values;
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
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  dataset_ = CreateRangeDataset<int64>(start, end, step);

  EXPECT_EQ(dataset_->name(), kOpName);
}

TEST_F(RangeDatasetOpTest, DatasetOutputDtypes) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  dataset_ = CreateRangeDataset<int64>(start, end, step);

  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(dataset_->output_dtypes(), expected_dtypes);
}

TEST_F(RangeDatasetOpTest, DatasetOutputShapes) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  dataset_ = CreateRangeDataset<int64>(start, end, step);

  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(dataset_->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < dataset_->output_shapes().size(); ++i) {
    dataset_->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
  }
}

struct DatasetCardinalityTest
    : RangeDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<int64, int64, int64, int>> {};

TEST_P(DatasetCardinalityTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  int64 start = std::get<0>(GetParam());
  int64 end = std::get<1>(GetParam());
  int64 step = std::get<2>(GetParam());
  int expected_cardinality = std::get<3>(GetParam());

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  dataset_ = CreateRangeDataset<int64>(start, end, step);

  EXPECT_EQ(dataset_->Cardinality(), expected_cardinality);
}

INSTANTIATE_TEST_CASE_P(RangeDatasetOpTest, DatasetCardinalityTest,
                        ::testing::Values(std::make_tuple(0, 10, 1, 10),
                                          std::make_tuple(0, 10, 3, 4),
                                          std::make_tuple(10, 0, -3, 4)));

TEST_F(RangeDatasetOpTest, DatasetSave) {
  int64 thread_num = 2, cpu_num = 2;
  int start = 0, end = 10, step = 1;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  dataset_ = CreateRangeDataset<int64>(start, end, step);
  TF_ASSERT_OK(CreateSerializationContext(&serialization_context_));

  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(dataset_->Save(serialization_context_.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_F(RangeDatasetOpTest, IteratorOutputDtypes) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  dataset_ = CreateRangeDataset<int64>(start, end, step);
  TF_ASSERT_OK(CreateIteratorContext(context_.get(), &iterator_context_));
  TF_ASSERT_OK(CreateIterator(iterator_context_.get(), dataset_, &iterator_));

  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(iterator_->output_dtypes(), expected_dtypes);
}

TEST_F(RangeDatasetOpTest, IteratorOutputShapes) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  dataset_ = CreateRangeDataset<int64>(start, end, step);
  TF_ASSERT_OK(CreateIteratorContext(context_.get(), &iterator_context_));
  TF_ASSERT_OK(CreateIterator(iterator_context_.get(), dataset_, &iterator_));

  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(iterator_->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < dataset_->output_shapes().size(); ++i) {
    iterator_->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
  }
}

TEST_F(RangeDatasetOpTest, IteratorOutputPrefix) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  dataset_ = CreateRangeDataset<int64>(start, end, step);
  TF_ASSERT_OK(CreateIteratorContext(context_.get(), &iterator_context_));
  TF_ASSERT_OK(CreateIterator(iterator_context_.get(), dataset_, &iterator_));

  EXPECT_EQ(iterator_->prefix(), "Iterator::Range");
}

struct IteratorRoundtripTest
    : RangeDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

TEST_P(IteratorRoundtripTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  int64 start = std::get<0>(GetParam());
  int64 end = std::get<1>(GetParam());
  int64 step = std::get<2>(GetParam());
  int breakpoint = std::get<3>(GetParam());

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  dataset_ = CreateRangeDataset<int64>(start, end, step);
  TF_ASSERT_OK(CreateIteratorContext(context_.get(), &iterator_context_));
  TF_ASSERT_OK(CreateIterator(iterator_context_.get(), dataset_, &iterator_));

  std::vector<Tensor> out_tensors;
  bool end_of_sequence = false;
  int cur_val = start - step;
  for (int i = 0; i < breakpoint; i++) {
    if (!end_of_sequence) {
      TF_EXPECT_OK(GetNext(iterator_.get(), iterator_context_.get(),
                           &out_tensors, &end_of_sequence));
      cur_val = ((end - cur_val - step) * step > 0) ? cur_val + step : cur_val;
    }
  }
  TF_CHECK_OK(CreateSerializationContext(&serialization_context_));
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(iterator_->Save(serialization_context_.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
  VariantTensorDataReader reader(&data);
  TF_ASSERT_OK(iterator_->Restore(iterator_context_.get(), &reader));
  TF_EXPECT_OK(GetNext(iterator_.get(), iterator_context_.get(), &out_tensors,
                       &end_of_sequence));
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
