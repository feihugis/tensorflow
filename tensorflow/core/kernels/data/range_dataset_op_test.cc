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
  template <typename T>
  Status MakeDatasetOpKernel() {
    DataType value_type = tensorflow::DataTypeToEnum<T>::value;
    DataTypeVector dtypes({value_type});
    std::vector<PartialTensorShape> shapes({{}});

    TF_RETURN_IF_ERROR(NodeDefBuilder(kNodeName, kOpName)
                           .Input(FakeInput(DT_INT64))
                           .Input(FakeInput(DT_INT64))
                           .Input(FakeInput(DT_INT64))
                           .Attr("output_types", dtypes)
                           .Attr("output_shapes", shapes)
                           .Finalize(&node_def_));
    TF_RETURN_IF_ERROR(MakeOpKernel(node_def_, kernel_));
    return Status::OK();
  }

  Status MakeDataset(int64 start, int64 end, int64 step, int output_index) {
    inputs_.clear();
    AddDatasetInputFromArray<int64>(&inputs_, kernel_->input_types(),
                                    TensorShape({}), {start});
    AddDatasetInputFromArray<int64>(&inputs_, kernel_->input_types(),
                                    TensorShape({}), {end});
    AddDatasetInputFromArray<int64>(&inputs_, kernel_->input_types(),
                                    TensorShape({}), {step});

    TF_RETURN_IF_ERROR(InitOpKernelContext(kernel_.get(), &inputs_, &context_));

    TF_RETURN_IF_ERROR(CheckOpKernelInput(*kernel_, inputs_));

    TF_RETURN_IF_ERROR(RunOpKernel(kernel_.get(), context_.get()));
    TF_RETURN_IF_ERROR(
        InitDatasetFromContext(context_.get(), output_index, &dataset_));
    return Status::OK();
  }

  const string& name() const { return dataset_->name(); }

 protected:
  NodeDef node_def_;
  DatasetBase* dataset_;
  gtl::InlinedVector<TensorValue, 4> inputs_;
  std::unique_ptr<OpKernel> kernel_;
  std::unique_ptr<OpKernelContext> context_;
  std::unique_ptr<IteratorContext> iterator_context_;
  std::unique_ptr<IteratorBase> iterator_;
  std::unique_ptr<SerializationContext> serialization_context_;
};

const char* RangeDatasetOpTest::kNodeName = "range_dataset";
const char* RangeDatasetOpTest::kOpName = "RangeDataset";

struct DatasetGetNextTest
    : RangeDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<int, int, int>> {};

TEST_P(DatasetGetNextTest, GetNext) {
  int output_index = 0, thread_num = 2, cpu_num = 2;
  int start = std::get<0>(GetParam());
  int end = std::get<1>(GetParam());
  int step = std::get<2>(GetParam());

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));

  TF_ASSERT_OK(MakeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeDataset(start, end, step, output_index));
  TF_ASSERT_OK(
      MakeIterator(context_.get(), dataset_, &iterator_context_, &iterator_));
  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(GetNext(iterator_.get(), iterator_context_.get(), &out_tensors,
                         &end_of_sequence));
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
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TF_ASSERT_OK(MakeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeDataset(start, end, step, output_index));
  EXPECT_EQ(dataset_->name(), kOpName);
}

TEST_F(RangeDatasetOpTest, DatasetOutputDtypes) {
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TF_ASSERT_OK(MakeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeDataset(start, end, step, output_index));
  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(dataset_->output_dtypes(), expected_dtypes);
}

TEST_F(RangeDatasetOpTest, DatasetOutputShapes) {
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TF_ASSERT_OK(MakeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeDataset(start, end, step, output_index));
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
  int output_index = 0, thread_num = 2, cpu_num = 2;
  int start = std::get<0>(GetParam());
  int end = std::get<1>(GetParam());
  int step = std::get<2>(GetParam());
  int expected_cardinality = std::get<3>(GetParam());
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TF_ASSERT_OK(MakeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeDataset(start, end, step, output_index));
  EXPECT_EQ(dataset_->Cardinality(), expected_cardinality);
}

INSTANTIATE_TEST_CASE_P(RangeDatasetOpTest, DatasetCardinalityTest,
                        ::testing::Values(std::make_tuple(0, 10, 1, 10),
                                          std::make_tuple(0, 10, 3, 4),
                                          std::make_tuple(10, 0, -3, 4)));

TEST_F(RangeDatasetOpTest, DatasetSave) {
  int output_index = 0, thread_num = 2, cpu_num = 2;
  int start = 0, end = 10, step = 1;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TF_ASSERT_OK(MakeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeDataset(start, end, step, output_index));
  InitSerializationContext(serialization_context_);
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(dataset_->Save(serialization_context_.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_F(RangeDatasetOpTest, IteratorOutputDtypes) {
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TF_ASSERT_OK(MakeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeDataset(start, end, step, output_index));
  TF_ASSERT_OK(
      MakeIterator(context_.get(), dataset_, &iterator_context_, &iterator_));
  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(iterator_->output_dtypes(), expected_dtypes);
}

TEST_F(RangeDatasetOpTest, IteratorOutputShapes) {
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TF_ASSERT_OK(MakeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeDataset(start, end, step, output_index));
  TF_ASSERT_OK(
      MakeIterator(context_.get(), dataset_, &iterator_context_, &iterator_));
  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(iterator_->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < dataset_->output_shapes().size(); ++i) {
    iterator_->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
  }
}

TEST_F(RangeDatasetOpTest, IteratorOutputPrefix) {
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TF_ASSERT_OK(MakeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeDataset(start, end, step, output_index));
  TF_ASSERT_OK(
      MakeIterator(context_.get(), dataset_, &iterator_context_, &iterator_));
  EXPECT_EQ(iterator_->prefix(), "Iterator::Range");
}

struct IteratorRoundtripTest
    : RangeDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

TEST_P(IteratorRoundtripTest, Roundtrip) {
  int output_index = 0, thread_num = 2, cpu_num = 2;
  int start = std::get<0>(GetParam());
  int end = std::get<1>(GetParam());
  int step = std::get<2>(GetParam());
  int breakpoint = std::get<3>(GetParam());
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({}, cpu_num));
  TF_ASSERT_OK(MakeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeDataset(start, end, step, output_index));
  TF_ASSERT_OK(
      MakeIterator(context_.get(), dataset_, &iterator_context_, &iterator_));
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
  InitSerializationContext(serialization_context_);
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
