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
#include "tensorflow/core/kernels/data/batch_dataset_op.h"

#include "tensorflow/core/kernels/data/dataset_test_base.h"

namespace tensorflow {
namespace data {
namespace {

constexpr char kNodeName[] = "batch_dataset_v2";
constexpr int kOpVersion = 2;
constexpr char kIteratorPrefix[] = "Iterator";

class BatchDatasetOpTest : public DatasetOpsTestBaseV2<BatchDatasetParams> {
 protected:
  // Creates a new `BatchDataset` op kernel.
  Status MakeDatasetOpKernel(
      const BatchDatasetParams& dataset_params,
      std::unique_ptr<OpKernel>* batch_dataset_op_kernel) override {
    name_utils::OpNameParams params;
    params.op_version = kOpVersion;
    NodeDef node_def = test::function::NDef(
        dataset_params.node_name,
        name_utils::OpName(BatchDatasetOp::kDatasetType, params),
        {BatchDatasetOp::kInputDataset, BatchDatasetOp::kBatchSize,
         BatchDatasetOp::kDropRemainder},
        {{BatchDatasetOp::kParallelCopy, dataset_params.parallel_copy},
         {BatchDatasetOp::kOutputTypes, dataset_params.output_dtypes},
         {BatchDatasetOp::kOutputShapes, dataset_params.output_shapes}});
    TF_RETURN_IF_ERROR(CreateOpKernel(node_def, batch_dataset_op_kernel));
    return Status::OK();
  }
};

// Test Case 1: test BatchDatasetV2 with `drop_remainder` = false and a batch
// size that can evenly split the input dataset.
BatchDatasetParams BatchDatasetParams1() {
  return {{/*start=*/0, /*stop=*/12, /*step=*/1},
          /*batch_size=*/4,
          /*drop_remainder=*/false,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({4})},
          /*node_name=*/kNodeName};
}

// Test Case 2: test BatchDatasetV2 with `drop_remainder` = true and a batch
// size that can evenly split the input dataset.
BatchDatasetParams BatchDatasetParams2() {
  return {{/*start=*/0, /*stop=*/12, /*step=*/1},
          /*batch_size=*/4,
          /*drop_remainder=*/true,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({4})},
          /*node_name=*/kNodeName};
}

// Test Case 3: test BatchDatasetV2 with `drop_remainder` = false and a batch
// size that can not evenly split the input dataset.
BatchDatasetParams BatchDatasetParams3() {
  return {{/*start=*/0, /*stop=*/10, /*step=*/1},
          /*batch_size=*/3,
          /*drop_remainder=*/false,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({-1})},
          /*node_name=*/kNodeName};
}

// Test Case 4: test BatchDatasetV2 with `drop_remainder` = true and a batch
// size that can not evenly split the input dataset.
BatchDatasetParams BatchDatasetParams4() {
  return {{/*start=*/0, /*stop=*/10, /*step=*/1},
          /*batch_size=*/3,
          /*drop_remainder=*/true,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({3})},
          /*node_name=*/kNodeName};
}

// Test Case 5: test BatchDatasetV2 with `drop_remainder` = true and
// `batch_size` > the cardinality of the input dataset.
BatchDatasetParams BatchDatasetParams5() {
  return {{/*start=*/0, /*stop=*/10, /*step=*/1},
          /*batch_size=*/12,
          /*drop_remainder=*/true,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({12})},
          /*node_name=*/kNodeName};
}

// Test Case 6: test BatchDatasetV2 with `drop_remainder` = false and
// `batch_size` > the cardinality of the input dataset.
BatchDatasetParams BatchDatasetParams6() {
  return {{/*start=*/0, /*stop=*/10, /*step=*/1},
          /*batch_size=*/12,
          /*drop_remainder=*/false,
          /*parallel_copy=*/true,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({-1})},
          /*node_name=*/kNodeName};
}

// Test Case 7: test BatchDatasetV2 with `drop_remainder` = false and
// the output of the input dataset is empty.
BatchDatasetParams BatchDatasetParams7() {
  return {{/*start=*/0, /*stop=*/0, /*step=*/1},
          /*batch_size=*/4,
          /*drop_remainder=*/false,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({4})},
          /*node_name=*/kNodeName};
}

// Test Case 8: test BatchDatasetV2 with an invalid batch size
BatchDatasetParams InvalidBatchSizeBatchDatasetParams() {
  return {{/*start=*/0, /*stop=*/10, /*step=*/1},
          /*batch_size=*/-1,
          /*drop_remainder=*/false,
          /*parallel_copy=*/false,
          /*output_dtypes=*/{DT_INT64},
          /*output_shapes=*/{PartialTensorShape({3})},
          /*node_name=*/kNodeName};
}

std::vector<GetNextTestCase<BatchDatasetParams>> GetNextTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({4}),
                                {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDatasetParams2(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({4}),
                                {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDatasetParams3(),
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
            CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
            CreateTensor<int64>(TensorShape({3}), {6, 7, 8}),
            CreateTensor<int64>(TensorShape({1}), {9})}},
          {/*dataset_params=*/BatchDatasetParams4(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({3}),
                                {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}})},
          {/*dataset_params=*/BatchDatasetParams5(),
           /*expected_outputs=*/{}},
          {/*dataset_params=*/BatchDatasetParams6(),
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({10}),
                                {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}})},

          {/*dataset_params=*/BatchDatasetParams7(),
           /*expected_outputs=*/{}}};
}

ITERATOR_GET_NEXT_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                         GetNextTestCases())

std::vector<DatasetNodeNameTestCase<BatchDatasetParams>>
DatasetNodeNameTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_node_name=*/kNodeName}};
}

DATASET_NODE_NAME_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                         DatasetNodeNameTestCases())

std::vector<DatasetTypeStringTestCase<BatchDatasetParams>>
DatasetTypeStringTestCases() {
  name_utils::OpNameParams params;
  params.op_version = kOpVersion;
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_dataset_type_string=*/name_utils::OpName(
               BatchDatasetOp::kDatasetType, params)}};
}

DATASET_TYPE_STRING_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                           DatasetTypeStringTestCases())

std::vector<DatasetOutputDtypesTestCase<BatchDatasetParams>>
DatasetOutputDtypesTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

DATASET_OUTPUT_DTYPES_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                             DatasetOutputDtypesTestCases())

std::vector<DatasetOutputShapesTestCase<BatchDatasetParams>>
DatasetOutputShapesTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({3})}},
          {/*dataset_params=*/BatchDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({12})}},
          {/*dataset_params=*/BatchDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}}};
}

DATASET_OUTPUT_SHAPES_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                             DatasetOutputShapesTestCases())

std::vector<CardinalityTestCase<BatchDatasetParams>> CardinalityTestCases() {
  return {
      {/*dataset_params=*/BatchDatasetParams1(), /*expected_cardinality=*/3},
      {/*dataset_params=*/BatchDatasetParams2(), /*expected_cardinality=*/3},
      {/*dataset_params=*/BatchDatasetParams3(), /*expected_cardinality=*/4},
      {/*dataset_params=*/BatchDatasetParams4(), /*expected_cardinality=*/3},
      {/*dataset_params=*/BatchDatasetParams5(), /*expected_cardinality=*/0},
      {/*dataset_params=*/BatchDatasetParams6(), /*expected_cardinality=*/1},
      {/*dataset_params=*/BatchDatasetParams7(), /*expected_cardinality=*/0}};
}

DATASET_CARDINALITY_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                           CardinalityTestCases())

std::vector<IteratorOutputDtypesTestCase<BatchDatasetParams>>
IteratorOutputDtypesTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_output_dtypes=*/{DT_INT64}}};
}

ITERATOR_OUTPUT_DTYPES_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                              IteratorOutputDtypesTestCases())

std::vector<IteratorOutputShapesTestCase<BatchDatasetParams>>
IteratorOutputShapesTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDatasetParams2(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}},
          {/*dataset_params=*/BatchDatasetParams3(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDatasetParams4(),
           /*expected_output_shapes=*/{PartialTensorShape({3})}},
          {/*dataset_params=*/BatchDatasetParams5(),
           /*expected_output_shapes=*/{PartialTensorShape({12})}},
          {/*dataset_params=*/BatchDatasetParams6(),
           /*expected_output_shapes=*/{PartialTensorShape({-1})}},
          {/*dataset_params=*/BatchDatasetParams7(),
           /*expected_output_shapes=*/{PartialTensorShape({4})}}};
}

ITERATOR_OUTPUT_SHAPES_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                              IteratorOutputShapesTestCases())

std::vector<IteratorPrefixTestCase<BatchDatasetParams>>
IteratorPrefixTestCases() {
  name_utils::IteratorPrefixParams params;
  params.op_version = kOpVersion;
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*expected_iterator_prefix=*/name_utils::IteratorPrefix(
               BatchDatasetOp::kDatasetType,
               BatchDatasetParams1().iterator_prefix, params)}};
}

ITERATOR_PREFIX_TEST_P(BatchDatasetOpTest, BatchDatasetParams,
                       IteratorPrefixTestCases())

std::vector<IteratorSaveAndRestoreTestCase<BatchDatasetParams>>
IteratorSaveAndRestoreTestCases() {
  return {{/*dataset_params=*/BatchDatasetParams1(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({4}),
                                {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDatasetParams2(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           CreateTensors<int64>(TensorShape({4}),
                                {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}})},
          {/*dataset_params=*/BatchDatasetParams3(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
            CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
            CreateTensor<int64>(TensorShape({3}), {6, 7, 8}),
            CreateTensor<int64>(TensorShape({1}), {9})}},
          {/*dataset_params=*/BatchDatasetParams4(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({3}), {0, 1, 2}),
            CreateTensor<int64>(TensorShape({3}), {3, 4, 5}),
            CreateTensor<int64>(TensorShape({3}), {6, 7, 8})}},
          {/*dataset_params=*/BatchDatasetParams5(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/{}},
          {/*dataset_params=*/BatchDatasetParams6(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/
           {CreateTensor<int64>(TensorShape({10}),
                                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})}},
          {/*dataset_params=*/BatchDatasetParams7(),
           /*breakpoints=*/{0, 1, 5},
           /*expected_outputs=*/{}}};
}

ITERATOR_SAVE_AND_RESTORE_TEST_P(
    BatchDatasetOpTest, IteratorSaveAndRestoreTestCase<BatchDatasetParams>,
    IteratorSaveAndRestoreTestCases())

TEST_F(BatchDatasetOpTest, InvalidBatchSize) {
  auto batch_dataset_params = InvalidBatchSizeBatchDatasetParams();
  EXPECT_EQ(Initialize(&batch_dataset_params).code(),
            tensorflow::error::INVALID_ARGUMENT);
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
