# Copyright 2022 The TensorFlow NUFFT Authors. All Rights Reserved.
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
"""Tests for module `nufft_options`."""

import tensorflow as tf

from tensorflow_nufft.python.ops import nufft_options


class OptionsTest(tf.test.TestCase):
  """Tests for Options structure."""
  def test_options_proto(self):
    """Test (de)serialization of Options to/from proto."""
    # Create example data.
    options = nufft_options.Options()
    # Test default values.
    self.assertEqual(options.points_range, nufft_options.PointsRange.EXTENDED)
    self.assertEqual(options.debugging.check_bounds, False)
    # Change some values.
    options.max_batch_size = 4
    options.fftw.planning_rigor = nufft_options.FftwPlanningRigor.PATIENT
    options.debugging.check_bounds = True
    options.points_range = nufft_options.PointsRange.INFINITE
    # Test round-trip options -> proto -> options.
    options2 = nufft_options.Options.from_proto(options.to_proto())
    self.assertEqual(options2, options)

  def test_invalid_value(self):
    """Test that invalid values are rejected."""
    # Test validation on object creation.
    with self.assertRaises(ValueError):
      options = nufft_options.Options(max_batch_size='banana')
    # Test validation on assignment.
    options = nufft_options.Options()
    with self.assertRaises(ValueError):
      options.max_batch_size = 'kiwi'


if __name__ == '__main__':
  tf.test.main()
