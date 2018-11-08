# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests which set DEBUG_SAVEALL and assert no garbage was created.

This flag seems to be sticky, so these tests have been isolated for now.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '1'

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test

from tensorflow.python.keras.layers.convolutional import Conv3D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.core import Flatten
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.engine.training import Model


from tensorflow.python.eager import backprop
from tensorflow.python.training.gradient_descent import GradientDescentOptimizer
from tensorflow.python.training.training_util import get_or_create_global_step

from tensorflow.python.ops.random_ops import random_uniform


class NoReferenceCycleTests(test_util.TensorFlowTestCase):
  """
  @test_util.assert_no_garbage_created
  def testEagerResourceVariables(self):
    with context.eager_mode():
      resource_variable_ops.ResourceVariable(1.0, name="a")

  @test_util.assert_no_garbage_created
  def testTensorArrays(self):
    with context.eager_mode():
      ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32,
          tensor_array_name="foo",
          size=3,
          infer_shape=False)

      w0 = ta.write(0, [[4.0, 5.0]])
      w1 = w0.write(1, [[1.0]])
      w2 = w1.write(2, -3.0)

      r0 = w2.read(0)
      r1 = w2.read(1)
      r2 = w2.read(2)

      d0, d1, d2 = self.evaluate([r0, r1, r2])
      self.assertAllEqual([[4.0, 5.0]], d0)
      self.assertAllEqual([[1.0]], d1)
      self.assertAllEqual(-3.0, d2)
  """

  def testMaxPool3DGradient(self):
    class MyModel(Model):
      def __init__(self):
        super(MyModel, self).__init__()
        self.conv3d_1 = Conv3D(filters=6, kernel_size=(3, 3, 3))
        self.max_pool_1 = MaxPooling3D(pool_size=(2, 2, 2))
        self.flatten = Flatten()
        self.dense_1 = Dense(5)

      def call(self, input, training=False):
        model = self.conv3d_1(input)
        model = self.max_pool_1(model)
        model = self.flatten(model)
        model = self.dense_1(model)
        return model

    def loss(model, x, y):
      logits = model(x)
      return losses.softmax_cross_entropy(onehot_labels=y, logits=logits), logits

    def grad(model, inputs, targets):
      with backprop.GradientTape() as tape:
        loss_value, logits = loss(model, inputs, targets)
      return loss_value, logits, tape.gradient(loss_value,
                                               model.trainable_variables)

    with context.eager_mode():
      model = MyModel()
      optimizer = GradientDescentOptimizer(learning_rate=0.01)
      global_step = get_or_create_global_step()

      x = random_uniform((10, 5, 10, 10, 3))
      y = random_uniform((10, 5))
      loss_value, logits, grads = grad(model, x, y)
      var_grad = zip(model.variables, grads)
      print(list(var_grad))
      optimizer.apply_gradients(zip(grads, model.variables), global_step)


if __name__ == "__main__":
  test.main()
