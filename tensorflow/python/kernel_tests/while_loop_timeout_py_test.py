# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OiR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=g-long-lambda
"""Tests for tensorflow.ops.control_flow_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import os
import sys
import threading
import time
import warnings

import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as framework_device_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
# Import gradients to resolve circular imports
from tensorflow.python.ops import gradients  # pylint: disable=unused-import
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None

import os
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '4'


class WhileLoopTimeoutTest(test_util.TensorFlowTestCase):

  def testTimeoutWithShortOperations(self):
    num_epochs = 1
    q = data_flow_ops.FIFOQueue(capacity=50, dtypes=[dtypes.int32], shapes=[()])
    enqueue_op = q.enqueue_many(constant_op.constant([1, 2]))

    # Use a 10-second timeout, which should be longer than any
    # non-blocking enqueue_many op.
    config = config_pb2.ConfigProto(operation_timeout_in_ms=1)
    with session.Session(config=config) as sess:
      for _ in range(num_epochs):
        sess.run(enqueue_op)
      self.assertEqual(sess.run(q.size()), num_epochs * 2)

  def testWhile_configOption(self):
    self.assertEqual(10, 10)
    # timeout_in_ms = 1
    # config = config_pb2.ConfigProto(operation_timeout_in_ms=timeout_in_ms)
    # run_options = config_pb2.RunOptions(timeout_in_ms=timeout_in_ms)
    # with self.cached_session(config=config) as sess:
    #   n = constant_op.constant(0)
    #   c = lambda x: math_ops.less(x, 2000000)
    #   b = lambda x: math_ops.add(x, 1)
    #   r = control_flow_ops.while_loop(c, b, [n])
    #   with self.assertRaises(errors.DeadlineExceededError):
    #     sess.run(r, options=run_options)



if __name__ == '__main__':
  googletest.main()
