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

 protected:
  // Creates a new MapDataset, meanwhile initializing the operation kernel and
  // context internally.
  DatasetBase* CreateMapDataset(DatasetBase* input_dataset,
                                const string& func_name) {
    CHECK_EQ(map_inputs_.size(), 0);
    map_kernel_ =
        CreateMapDatasetOpKernel<int64>(input_dataset->name(), func_name);

    // Save the input dataset into a variant tensor as the input of MapDataset.
    Tensor dataset_tensor(DT_VARIANT, TensorShape({}));
    TF_CHECK_OK(StoreDatasetInVariantTensor(input_dataset, &dataset_tensor));
    Variant variant = dataset_tensor.scalar<Variant>()();
    AddDatasetInputFromArray<Variant>(&map_inputs_, map_kernel_->input_types(),
                                      TensorShape({}), {variant});

    map_context_ = CreateOpKernelContext(map_kernel_.get(), &map_inputs_);
    TF_CHECK_OK(CheckOpKernelInput(*map_kernel_, map_inputs_));
    return CreateDataset(map_kernel_.get(), map_context_.get());
  }

 private:
  // Creates a new MapDatasetOp kernel. The `input_dataset` parameter should be
  // same with the node name of the input dataset for the method
  // `CreateMapDataset()`.
  template <typename T>
  std::unique_ptr<OpKernel> CreateMapDatasetOpKernel(
      const string& input_dataset, const string& func_name) {
    DataTypeVector map_output_dtypes({tensorflow::DataTypeToEnum<T>::value});
    gtl::ArraySlice<TensorShape> map_output_shapes({{}});
    FunctionDefHelper::AttrValueWrapper func =
        FunctionDefHelper::FunctionRef(func_name, {{"T", DT_INT64}});

    map_node_def_ = test::function::NDef(kNodeName, kOpName, {input_dataset},
                                         {{"f", func},
                                          {"Targuments", {}},
                                          {"output_shapes", map_output_shapes},
                                          {"output_types", map_output_dtypes},
                                          {"use_inter_op_parallelism", true},
                                          {"preserve_cardinality", false}});
    return CreateOpKernel(map_node_def_);
  }

 protected:
  DatasetBase* range_dataset_;
  DatasetBase* map_dataset_;
  std::unique_ptr<OpKernel> map_kernel_;
  std::unique_ptr<OpKernelContext> map_context_;
  std::unique_ptr<IteratorContext> iterator_context_;
  std::unique_ptr<IteratorBase> iterator_;
  std::unique_ptr<SerializationContext> serialization_context_;

 private:
  NodeDef map_node_def_;
  gtl::InlinedVector<TensorValue, 4> map_inputs_;
};

const char* MapDatasetOpTest::kNodeName = "map_dataset";
const char* MapDatasetOpTest::kOpName = "MapDataset";

struct DatasetGetNextTest
    : MapDatasetOpTest,
      ::testing::WithParamInterface<
          std::tuple<int64, int64, int64, string, std::vector<int64>,
                     std::vector<FunctionDef>>> {};

TEST_P(DatasetGetNextTest, GetNext) {
  int thread_num = 2, cpu_num = 2;
  int64 start = std::get<0>(GetParam());
  int64 end = std::get<1>(GetParam());
  int64 step = std::get<2>(GetParam());
  string func_name = std::get<3>(GetParam());
  std::vector<int64> expected_values = std::get<4>(GetParam());
  std::vector<FunctionDef> func_lib = std::get<5>(GetParam());

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(func_lib, cpu_num));
  range_dataset_ = CreateRangeDataset<int64>(start, end, step, "range");
  map_dataset_ = CreateMapDataset(range_dataset_, func_name);
  iterator_ =
      CreateIterator(map_context_.get(), map_dataset_, &iterator_context_);

  bool end_of_sequence = false;
  std::vector<Tensor> out_tensors;
  while (!end_of_sequence) {
    TF_EXPECT_OK(GetNext(iterator_.get(), iterator_context_.get(), &out_tensors,
                         &end_of_sequence));
  }

  EXPECT_EQ(out_tensors.size(), expected_values.size());
  for (size_t i = 0; i < out_tensors.size(); ++i) {
    int64 actual_value = out_tensors[i].flat<int64>()(0);
    int64 expect_value = expected_values[i];
    EXPECT_EQ(actual_value, expect_value);
  }
}

INSTANTIATE_TEST_CASE_P(
    MapDatasetOpTest, DatasetGetNextTest,
    ::testing::Values(
        std::make_tuple(0, 10, 3, "XTimesTwo", std::vector<int64>{0, 6, 12, 18},
                        std::vector<FunctionDef>{test::function::XTimesTwo()}),
        std::make_tuple(0, 10, 3, "XAddX", std::vector<int64>{0, 6, 12, 18},
                        std::vector<FunctionDef>{test::function::XAddX()}),
        std::make_tuple(
            10, 0, -3, "XTimesFour", std::vector<int64>{40, 28, 16, 4},
            std::vector<FunctionDef>{test::function::XTimesTwo(),
                                     test::function::XTimesFour()})));

TEST_F(MapDatasetOpTest, DatasetName) {
  int thread_num = 2, cpu_num = 2;
  int64 start = 0, end = 10, step = 1;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));
  range_dataset_ = CreateRangeDataset<int64>(start, end, step, "range");
  map_dataset_ = CreateMapDataset(range_dataset_, func_def.signature().name());
  EXPECT_EQ(map_dataset_->name(), kOpName);
}

TEST_F(MapDatasetOpTest, DatasetOutputDtypes) {
  int thread_num = 2, cpu_num = 2;
  int64 start = 0, end = 10, step = 1;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));
  range_dataset_ = CreateRangeDataset<int64>(start, end, step, "range");
  map_dataset_ = CreateMapDataset(range_dataset_, func_def.signature().name());
  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(map_dataset_->output_dtypes(), expected_dtypes);
}

TEST_F(MapDatasetOpTest, DatasetOutputShapes) {
  int thread_num = 2, cpu_num = 2;
  int64 start = 0, end = 10, step = 1;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));
  range_dataset_ = CreateRangeDataset<int64>(start, end, step, "range");
  map_dataset_ = CreateMapDataset(range_dataset_, func_def.signature().name());
  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(map_dataset_->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < map_dataset_->output_shapes().size(); ++i) {
    map_dataset_->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
  }
}

struct DatasetCardinalityTest
    : MapDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<int64, int64, int64, int>> {};

TEST_P(DatasetCardinalityTest, Cardinality) {
  int thread_num = 2, cpu_num = 2;
  int64 start = std::get<0>(GetParam());
  int64 end = std::get<1>(GetParam());
  int64 step = std::get<2>(GetParam());
  int expected_cardinality = std::get<3>(GetParam());
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));
  range_dataset_ = CreateRangeDataset<int64>(start, end, step, "range");
  map_dataset_ = CreateMapDataset(range_dataset_, func_def.signature().name());
  EXPECT_EQ(map_dataset_->Cardinality(), expected_cardinality);
}

INSTANTIATE_TEST_CASE_P(MapDatasetOpTest, DatasetCardinalityTest,
                        ::testing::Values(std::make_tuple(0, 10, 1, 10),
                                          std::make_tuple(0, 10, 3, 4),
                                          std::make_tuple(10, 0, -3, 4)));

TEST_F(MapDatasetOpTest, DatasetSave) {
  int thread_num = 2, cpu_num = 2;
  int64 start = 0, end = 10, step = 1;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));
  range_dataset_ = CreateRangeDataset<int64>(start, end, step, "range");
  map_dataset_ = CreateMapDataset(range_dataset_, func_def.signature().name());
  serialization_context_ = CreateSerializationContext();
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(map_dataset_->Save(serialization_context_.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
}

TEST_F(MapDatasetOpTest, IteratorOutputDtypes) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));

  range_dataset_ = CreateRangeDataset<int64>(start, end, step, "range");
  map_dataset_ = CreateMapDataset(range_dataset_, func_def.signature().name());
  iterator_ =
      CreateIterator(map_context_.get(), map_dataset_, &iterator_context_);
  DataTypeVector expected_dtypes({DT_INT64});
  EXPECT_EQ(iterator_->output_dtypes(), expected_dtypes);
}

TEST_F(MapDatasetOpTest, IteratorOutputShapes) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));

  range_dataset_ = CreateRangeDataset<int64>(start, end, step, "range");
  map_dataset_ = CreateMapDataset(range_dataset_, func_def.signature().name());
  iterator_ =
      CreateIterator(map_context_.get(), map_dataset_, &iterator_context_);
  std::vector<PartialTensorShape> expected_shapes({{}});
  EXPECT_EQ(iterator_->output_shapes().size(), expected_shapes.size());
  for (int i = 0; i < map_dataset_->output_shapes().size(); ++i) {
    iterator_->output_shapes()[i].IsIdenticalTo(expected_shapes[i]);
  }
}

TEST_F(MapDatasetOpTest, IteratorOutputPrefix) {
  int64 start = 0, end = 10, step = 1;
  int thread_num = 2, cpu_num = 2;
  FunctionDef func_def = test::function::XTimesTwo();

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime({func_def}, cpu_num));

  range_dataset_ = CreateRangeDataset<int64>(start, end, step, "range");
  map_dataset_ = CreateMapDataset(range_dataset_, func_def.signature().name());
  iterator_ =
      CreateIterator(map_context_.get(), map_dataset_, &iterator_context_);
  EXPECT_EQ(iterator_->prefix(), "Iterator::Map");
}

struct IteratorRoundtripTest
    : MapDatasetOpTest,
      ::testing::WithParamInterface<std::tuple<
          int64, int64, int64, int, int64, string, std::vector<FunctionDef>>> {
};

TEST_P(IteratorRoundtripTest, Roundtrip) {
  int thread_num = 2, cpu_num = 2;
  int64 start = std::get<0>(GetParam());
  int64 end = std::get<1>(GetParam());
  int64 step = std::get<2>(GetParam());
  int breakpoint = std::get<3>(GetParam());
  int64 expected_value = std::get<4>(GetParam());
  string func_name = std::get<5>(GetParam());
  std::vector<FunctionDef> func_lib = std::get<6>(GetParam());

  TF_ASSERT_OK(InitThreadPool(thread_num));
  TF_ASSERT_OK(InitFunctionLibraryRuntime(func_lib, cpu_num));

  range_dataset_ = CreateRangeDataset<int64>(start, end, step, "range");
  map_dataset_ = CreateMapDataset(range_dataset_, func_name);
  iterator_ =
      CreateIterator(map_context_.get(), map_dataset_, &iterator_context_);
  std::vector<Tensor> out_tensors;
  bool end_of_sequence = false;
  for (int i = 0; i < breakpoint; i++) {
    TF_EXPECT_OK(GetNext(iterator_.get(), iterator_context_.get(), &out_tensors,
                         &end_of_sequence));
  }
  serialization_context_ = CreateSerializationContext();
  VariantTensorData data;
  VariantTensorDataWriter writer(&data);
  TF_ASSERT_OK(iterator_->Save(serialization_context_.get(), &writer));
  TF_ASSERT_OK(writer.Flush());
  VariantTensorDataReader reader(&data);
  TF_ASSERT_OK(iterator_->Restore(iterator_context_.get(), &reader));
  TF_EXPECT_OK(GetNext(iterator_.get(), iterator_context_.get(), &out_tensors,
                       &end_of_sequence));
  EXPECT_EQ(out_tensors.back().flat<int64>()(0), expected_value);
}

INSTANTIATE_TEST_CASE_P(
    MapDatasetOpTest, IteratorRoundtripTest,
    ::testing::Values(
        std::make_tuple(0, 10, 2, 0, 0, "XTimesTwo",
                        std::vector<FunctionDef>{test::function::XTimesTwo()}),
        std::make_tuple(0, 10, 2, 4, 16, "XAddX",
                        std::vector<FunctionDef>{test::function::XAddX()}),
        std::make_tuple(0, 10, 2, 6, 32, "XTimesFour",
                        std::vector<FunctionDef>{
                            test::function::XTimesTwo(),
                            test::function::XTimesFour()})));

}  // namespace
}  // namespace data
}  // namespace tensorflow
