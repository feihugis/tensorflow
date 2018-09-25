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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for the experimental input pipeline ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
import shutil
import tempfile

from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops.dataset_ops import MatchingFilesDataset
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops.gen_io_ops import matching_files
from tensorflow.python.framework import errors

import os
import time
from functools import partial


def timeit(fn, msg, N=0):
  start = time.time()
  res = fn()
  end = time.time()
  runtime = (end - start) * 1000
  msg = '{}: time: {:.2f} ms'.format(msg, runtime)
  if N:
    msg += ' ({:.2f} ms per iteration)'.format(runtime / N)
  print(msg)
  return res


width = 32
depth = 16


class MatchingFilesDatasetTest(test.TestCase):

  def setUp(self):
    self.tmp_dir = tempfile.mkdtemp()

  def tearDown(self):
    shutil.rmtree(self.tmp_dir, ignore_errors=True)

  def _touchTempFiles(self, filenames):
    for filename in filenames:
      open(path.join(self.tmp_dir, filename), 'a').close()

  def testEmptyDirectory(self):
    dataset = MatchingFilesDataset(path.join(self.tmp_dir, '*'))
    with self.cached_session() as sess:
      next_element = dataset.make_one_shot_iterator().get_next()
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testSimpleDirectory(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    dataset = MatchingFilesDataset(path.join(self.tmp_dir, '*'))
    with self.cached_session() as sess:
      next_element = dataset.make_one_shot_iterator().get_next()

      expected_filenames = []
      actual_filenames = []
      for filename in filenames:
        expected_filenames.append(
          compat.as_bytes(path.join(self.tmp_dir, filename)))
        actual_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(expected_filenames, actual_filenames)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testSimpleDirectoryInitializer(self):
    filenames = ['a', 'b', 'c']
    self._touchTempFiles(filenames)

    dataset = MatchingFilesDataset(path.join(self.tmp_dir, '*'))
    with self.cached_session() as sess:
      next_element = dataset.make_one_shot_iterator().get_next()
      expected_filenames = []
      actual_filenames = []
      for filename in filenames:
        expected_filenames.append(
          compat.as_bytes(path.join(self.tmp_dir, filename)))
        actual_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(expected_filenames, actual_filenames)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testFileSuffixes(self):
    filenames = ['a.txt', 'b.py', 'c.py', 'd.pyc']
    self._touchTempFiles(filenames)

    dataset = MatchingFilesDataset(path.join(self.tmp_dir, '*.py'))
    with self.cached_session() as sess:
      next_element = dataset.make_one_shot_iterator().get_next()
      expected_filenames = []
      actual_filenames = []
      for filename in filenames[1:-1]:
        expected_filenames.append(
          compat.as_bytes(path.join(self.tmp_dir, filename)))
        actual_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(expected_filenames, actual_filenames)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testFileMiddles(self):
    filenames = ['a.txt', 'b.py', 'c.pyc']
    self._touchTempFiles(filenames)

    dataset = MatchingFilesDataset(path.join(self.tmp_dir, '*.py*'))
    with self.cached_session() as sess:
      next_element = dataset.make_one_shot_iterator().get_next()
      expected_filenames = []
      actual_filenames = []
      for filename in filenames[1:]:
        expected_filenames.append(
          compat.as_bytes(path.join(self.tmp_dir, filename)))
        actual_filenames.append(compat.as_bytes(sess.run(next_element)))

      self.assertItemsEqual(expected_filenames, actual_filenames)
      with self.assertRaises(errors.OutOfRangeError):
        sess.run(next_element)

  def testNestedDirectory(self):
    self.maxDiff = None
    filenames = []
    width = 8
    depth = 4
    for i in range(width):
      for j in range(depth):
        new_base = os.path.join(self.tmp_dir, str(i),
          *[str(dir_name) for dir_name in range(j)])
        os.makedirs(new_base, exist_ok=True)
        for f in ['a.txt', 'b.py', 'c.pyc']:
          filename = os.path.join(new_base, f)
          filenames.append(filename)
          open(filename, 'w').close()

    patterns = []
    for i in range(depth):
      pattern = '{}/{}/*.txt'.format(
        self.tmp_dir, os.path.join(*['**' for _ in range(i + 1)]))
      patterns.append(pattern)

    dataset = MatchingFilesDataset(patterns)
    with self.cached_session() as sess:
      next_element = dataset.make_one_shot_iterator().get_next()
      expected_filenames = [compat.as_bytes(file)
                            for file in filenames if file.endswith('.txt')]
      actual_filenames = []
      while True:
        try:
          actual_filenames.append(compat.as_bytes(sess.run(next_element)))
        except errors.OutOfRangeError:
          break

      self.assertItemsEqual(expected_filenames, actual_filenames)


if __name__ == "__main__":
  test.main()
