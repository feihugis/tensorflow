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
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
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

class MapDatasetOpTest : public DatasetOpsTestBase {
 public:
  static const char* kNodeName;
  static const char* kOpName;

  ~MapDatasetOpTest() override {
    gtl::STLDeleteElements(&tensors_);
    range_dataset_->Unref();
  }

 public:
  FunctionDef MakeFuncDef() { return test::function::XTimesTwo(); }

  string FuncName() { return "XTimesTwo"; }
  template <typename T>
  Status MakeRangeDatasetOpKernel() {
    DataTypeVector dtypes({tensorflow::DataTypeToEnum<T>::value});
    std::vector<PartialTensorShape> shapes({{}});
    range_node_def_ = test::function::NDef(
        "range", "RangeDataset", {"start", "stop", "step"},
        {{"output_types", dtypes}, {"output_shapes", shapes}});
    TF_RETURN_IF_ERROR(MakeOpKernel(range_node_def_, range_kernel_));
    return Status::OK();
  }

  Status MakeRangeDataset(int start, int end, int step, int output_index) {
    range_inputs_.clear();
    AddDatasetInputFromArray<int64>(
        &range_inputs_, range_kernel_->input_types(), TensorShape({}), {start});
    AddDatasetInputFromArray<int64>(
        &range_inputs_, range_kernel_->input_types(), TensorShape({}), {end});
    AddDatasetInputFromArray<int64>(
        &range_inputs_, range_kernel_->input_types(), TensorShape({}), {step});
    TF_RETURN_IF_ERROR(InitOpKernelContext(range_kernel_.get(), &range_inputs_,
                                           &range_context_));
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*range_kernel_, range_inputs_));
    TF_RETURN_IF_ERROR(RunOpKernel(range_kernel_.get(), range_context_.get()));
    TF_RETURN_IF_ERROR(InitDatasetFromContext(range_context_.get(),
                                              output_index, &range_dataset_));
    return Status::OK();
  }

  template <typename T>
  Status MakeMapDatasetOpKernel() {
    DataTypeVector map_output_dtypes({tensorflow::DataTypeToEnum<T>::value});
    gtl::ArraySlice<TensorShape> map_output_shapes({{}});
    FunctionDefHelper::AttrValueWrapper func =
        FunctionDefHelper::FunctionRef(FuncName(), {{"T", DT_INT64}});

    map_node_def_ = test::function::NDef(kNodeName, kOpName, {"range"},
                                         {{"f", func},
                                          {"Targuments", {}},
                                          {"output_shapes", map_output_shapes},
                                          {"output_types", map_output_dtypes},
                                          {"use_inter_op_parallelism", true},
                                          {"preserve_cardinality", false}});
    TF_RETURN_IF_ERROR(MakeOpKernel(map_node_def_, map_kernel_));
    return Status::OK();
  }

  Status MakeMapDataset(DatasetBase* input_dataset, int output_index) {
    map_inputs_.clear();
    range_context_.reset();
    Tensor dataset_tensor(DT_VARIANT, TensorShape({}));
    TF_RETURN_IF_ERROR(
        StoreDatasetInVariantTensor(input_dataset, &dataset_tensor));
    Variant variant = dataset_tensor.scalar<Variant>()();
    AddDatasetInputFromArray<Variant>(&map_inputs_, map_kernel_->input_types(),
                                      TensorShape({}), {variant});

    TF_RETURN_IF_ERROR(
        InitOpKernelContext(map_kernel_.get(), &map_inputs_, &map_context_));
    TF_RETURN_IF_ERROR(CheckOpKernelInput(*map_kernel_, map_inputs_));
    TF_RETURN_IF_ERROR(RunOpKernel(map_kernel_.get(), map_context_.get()));
    TF_RETURN_IF_ERROR(InitDatasetFromContext(map_context_.get(), output_index,
                                              &map_dataset_));
    return Status::OK();
  }

 protected:
  NodeDef range_node_def_;
  gtl::InlinedVector<TensorValue, 4> range_inputs_;
  std::unique_ptr<OpKernel> range_kernel_;
  DatasetBase* range_dataset_;
  std::unique_ptr<OpKernelContext> range_context_;

  NodeDef map_node_def_;
  DatasetBase* map_dataset_;
  gtl::InlinedVector<TensorValue, 4> map_inputs_;
  std::unique_ptr<OpKernel> map_kernel_;
  std::unique_ptr<OpKernelContext> map_context_;
  std::unique_ptr<IteratorContext> iterator_context_;
  std::unique_ptr<IteratorBase> iterator_;

  std::unique_ptr<SerializationContext> serialization_context_;
};

const char* MapDatasetOpTest::kNodeName = "map_dataset";
const char* MapDatasetOpTest::kOpName = "MapDataset";

struct DatasetGetNextTest
    : MapDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<int, int, int>> {};

TEST_P(DatasetGetNextTest, GetNext) {
  int output_index = 0, thread_num = 2, cpu_num = 2;
  int start = std::get<0>(GetParam());
  int end = std::get<1>(GetParam());
  int step = std::get<2>(GetParam());
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({MakeFuncDef()}, cpu_num));

  TF_ASSERT_OK(MakeRangeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeRangeDataset(start, end, step, output_index));

  TF_ASSERT_OK(MakeMapDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeMapDataset(range_dataset_, 0));
  TF_ASSERT_OK(MakeIterator(map_context_.get(), map_dataset_,
                            &iterator_context_, &iterator_));

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(GetNext(iterator_.get(), iterator_context_.get(), &out_tensors,
                         &end_of_sequence));
  }
  std::vector<int64> expected_values;
  for (int i = start; (end - i) * step > 0; i = i + step) {
    expected_values.reserve(1);
    expected_values.emplace_back(i * 2);
  }
  EXPECT_EQ(out_tensors.size(), expected_values.size());
  for (size_t i = 0; i < out_tensors.size(); ++i) {
    int64 actual_value = out_tensors[i].flat<int64>()(0);
    int64 expect_value = expected_values[i];
    EXPECT_EQ(actual_value, expect_value);
  }
}

INSTANTIATE_TEST_CASE_P(MapDatasetOpTest, DatasetGetNextTest,
                        ::testing::Values(std::make_tuple(0, 10, 1),
                                          std::make_tuple(0, 10, 3),
                                          std::make_tuple(10, 0, -1),
                                          std::make_tuple(10, 0, -3)));

TEST_F(MapDatasetOpTest, DatasetName) {
  int output_index = 0, thread_num = 2, cpu_num = 2;
  int start = 0, end = 10, step = 1;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({MakeFuncDef()}, cpu_num));

  TF_ASSERT_OK(MakeRangeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeRangeDataset(start, end, step, output_index));

  TF_ASSERT_OK(MakeMapDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeMapDataset(range_dataset_, 0));
  EXPECT_EQ(map_dataset_->name(), kOpName);
}

TEST_F(MapDatasetOpTest, DatasetOutputDtypes) {
  int output_index = 0, thread_num = 2, cpu_num = 2;
  int start = 0, end = 10, step = 1;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({MakeFuncDef()}, cpu_num));

  TF_ASSERT_OK(MakeRangeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeRangeDataset(start, end, step, output_index));

  TF_ASSERT_OK(MakeMapDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeMapDataset(range_dataset_, 0));
  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(map_dataset_->output_dtypes(), expected_dtypes);
}

TEST_F(MapDatasetOpTest, DatasetOutputShapes) {
  int output_index = 0, thread_num = 2, cpu_num = 2;
  int start = 0, end = 10, step = 1;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({MakeFuncDef()}, cpu_num));

  TF_ASSERT_OK(MakeRangeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeRangeDataset(start, end, step, output_index));

  TF_ASSERT_OK(MakeMapDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeMapDataset(range_dataset_, 0));
  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(map_dataset_->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < map_dataset_->output_shapes().size(); ++i) {
    map_dataset_->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
  }
}

struct DatasetCardinalityTest
    : MapDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

TEST_P(DatasetCardinalityTest, Cardinality) {
  int output_index = 0, thread_num = 2, cpu_num = 2;
  int start = std::get<0>(GetParam());
  int end = std::get<1>(GetParam());
  int step = std::get<2>(GetParam());
  int expected_cardinality = std::get<3>(GetParam());
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({MakeFuncDef()}, cpu_num));

  TF_ASSERT_OK(MakeRangeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeRangeDataset(start, end, step, output_index));

  TF_ASSERT_OK(MakeMapDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeMapDataset(range_dataset_, 0));
  EXPECT_EQ(map_dataset_->Cardinality(), expected_cardinality);
}

INSTANTIATE_TEST_CASE_P(MapDatasetOpTest, DatasetCardinalityTest,
                        ::testing::Values(std::make_tuple(0, 10, 1, 10),
                                          std::make_tuple(0, 10, 3, 4),
                                          std::make_tuple(10, 0, -3, 4)));

TEST_F(MapDatasetOpTest, DatasetSave) {
  int output_index = 0, thread_num = 2, cpu_num = 2;
  int start = 0, end = 10, step = 1;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({MakeFuncDef()}, cpu_num));

  TF_ASSERT_OK(MakeRangeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeRangeDataset(start, end, step, output_index));

  TF_ASSERT_OK(MakeMapDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeMapDataset(range_dataset_, 0));

  InitSerializationContext(serialization_context_);
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(map_dataset_->Save(serialization_context_.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_F(MapDatasetOpTest, IteratorOutputDtypes) {
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({MakeFuncDef()}, cpu_num));

  TF_ASSERT_OK(MakeRangeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeRangeDataset(start, end, step, output_index));

  TF_ASSERT_OK(MakeMapDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeMapDataset(range_dataset_, 0));
  TF_ASSERT_OK(MakeIterator(map_context_.get(), map_dataset_,
                            &iterator_context_, &iterator_));
  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(iterator_->output_dtypes(), expected_dtypes);
}

TEST_F(MapDatasetOpTest, IteratorOutputShapes) {
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({MakeFuncDef()}, cpu_num));

  TF_ASSERT_OK(MakeRangeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeRangeDataset(start, end, step, output_index));

  TF_ASSERT_OK(MakeMapDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeMapDataset(range_dataset_, 0));
  TF_ASSERT_OK(MakeIterator(map_context_.get(), map_dataset_,
                            &iterator_context_, &iterator_));
  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(iterator_->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < map_dataset_->output_shapes().size(); ++i) {
    iterator_->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
  }
}

TEST_F(MapDatasetOpTest, IteratorOutputPrefix) {
  int start = 0, end = 10, step = 1;
  int output_index = 0, thread_num = 2, cpu_num = 2;
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({MakeFuncDef()}, cpu_num));

  TF_ASSERT_OK(MakeRangeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeRangeDataset(start, end, step, output_index));

  TF_ASSERT_OK(MakeMapDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeMapDataset(range_dataset_, 0));
  TF_ASSERT_OK(MakeIterator(map_context_.get(), map_dataset_,
                            &iterator_context_, &iterator_));
  EXPECT_EQ(iterator_->prefix(), "Iterator::Map");
}

struct IteratorRoundtripTest
    : MapDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<int, int, int, int>> {};

TEST_P(IteratorRoundtripTest, Roundtrip) {
  int output_index = 0, thread_num = 2, cpu_num = 2;
  int start = std::get<0>(GetParam());
  int end = std::get<1>(GetParam());
  int step = std::get<2>(GetParam());
  int breakpoint = std::get<3>(GetParam());
  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({MakeFuncDef()}, cpu_num));

  TF_ASSERT_OK(MakeRangeDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeRangeDataset(start, end, step, output_index));

  TF_ASSERT_OK(MakeMapDatasetOpKernel<int64>());
  TF_ASSERT_OK(MakeMapDataset(range_dataset_, 0));
  TF_ASSERT_OK(MakeIterator(map_context_.get(), map_dataset_,
                            &iterator_context_, &iterator_));
  std::vector<Tensor> out_tensors;
  bool end_of_sequence = false;
  int cur_range_val = start - step;
  for (int i = 0; i < breakpoint; i++) {
    if (!end_of_sequence) {
      TF_EXPECT_OK(GetNext(iterator_.get(), iterator_context_.get(),
                           &out_tensors, &end_of_sequence));
      cur_range_val = ((end - cur_range_val - step) * step > 0)
                          ? cur_range_val + step
                          : cur_range_val;
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
  int expect_range_next = ((end - cur_range_val - step) * step > 0)
                              ? cur_range_val + step
                              : cur_range_val;
  EXPECT_EQ(out_tensors.back().flat<int64>()(0), expect_range_next * 2);
}

INSTANTIATE_TEST_CASE_P(
    MapDatasetOpTest, IteratorRoundtripTest,
    ::testing::Values(
        std::make_tuple(0, 10, 2, 0),    // unused_iterator
        std::make_tuple(0, 10, 2, 4),    // fully_used_iterator_increase
        std::make_tuple(10, 0, -2, 4),   // fully_used_iterator_decrease
        std::make_tuple(0, 10, 2, 6)));  // exhausted_iterator

}  // namespace
}  // namespace data
}  // namespace tensorflow
