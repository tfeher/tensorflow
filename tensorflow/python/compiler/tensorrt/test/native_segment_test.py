# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Model script to test TF-TensorRT integration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class AddTest(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, inp):
    """Create a graph containing single segment."""
    dtype = inp.dtype
    val = inp + inp
    return math_ops.abs(val)

  def GetConversionParams(self, run_params):
    """Return a ConversionParams for test."""
    conversion_params = super(AddTest,
                              self).GetConversionParams(run_params)

    # Asking for 128 GB of TensorRT workspace size, which will fail.
    # We expect that the native segment fallback will be used in TRTEngineOp
    return conversion_params._replace(max_workspace_size_bytes=1<<37)

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[1,1]], [[1,1]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]

class MulTest(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, inp1, inp2):
    """Create a graph containing single segment."""
    dtype = inp1.dtype
    val = inp1 * inp2
    return math_ops.abs(val)

  def GetConversionParams(self, run_params):
    """Return a ConversionParams for test."""
    conversion_params = super(MulTest,
                              self).GetConversionParams(run_params)
    # Asking for 128 GB of TensorRT workspace size, which will fail.
    # We expect that the native segment fallback will be used in TRTEngineOp
    return conversion_params._replace(max_workspace_size_bytes=1<<37)

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32, [[2,2,2],[2,2,2]],
                            [[2,2,2]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]

class MulTest3(trt_test.TfTrtIntegrationTestBase):
  def GraphFn(self, inp1, inp2, inp3):
    """Create a graph containing single segment."""
    dtype = inp1.dtype
    val = inp1 * inp2 + inp3
    return math_ops.abs(val)

  def GetConversionParams(self, run_params):
    """Return a ConversionParams for test."""
    conversion_params = super(MulTest3,
                              self).GetConversionParams(run_params)
    # Asking for 128 GB of TensorRT workspace size, which will fail.
    # We expect that the native segment fallback will be used in TRTEngineOp
    return conversion_params._replace(max_workspace_size_bytes=1<<37)

  def GetParams(self):
    return self.BuildParams(self.GraphFn, dtypes.float32,
                            [[2,2,2],[2,2,2],[2,2,2]],
                            [[2,2,2]])

  def ExpectedEnginesToBuild(self, run_params):
    """Return the expected engines to build."""
    return ["TRTEngineOp_0"]

if __name__ == "__main__":
  test.main()
