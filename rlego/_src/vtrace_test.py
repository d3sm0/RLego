# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Tests for `value_learning.py`."""

import functools

import functorch
import torch
from absl.testing import absltest
from absl.testing import parameterized
from rlego._src import utils
from rlego._src import vtrace


class VTraceTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

        behavior_policy_logits = torch.tensor(
            [[[8.9, 0.7], [5.0, 1.0], [0.6, 0.1], [-0.9, -0.1]],
             [[0.3, -5.0], [1.0, -8.0], [0.3, 1.7], [4.7, 3.3]]],
            dtype=torch.float32)
        target_policy_logits = torch.tensor(
            [[[0.4, 0.5], [9.2, 8.8], [0.7, 4.4], [7.9, 1.4]],
             [[1.0, 0.9], [1.0, -1.0], [-4.3, 8.7], [0.8, 0.3]]],
            dtype=torch.float32)
        actions = torch.tensor([[0, 1, 0, 0], [1, 0, 0, 1]], dtype=torch.int64)

        rho_tm1 = (target_policy_logits.softmax(dim=-1) / behavior_policy_logits.softmax(dim=-1))
        self._rho_tm1 = utils.batched_index(rho_tm1, actions)

        self._rewards = torch.tensor(
            [[-1.3, -1.3, 2.3, 42.0],
             [1.3, 5.3, -3.3, -5.0]],
            dtype=torch.float32)
        self._discounts = torch.tensor(
            [[0., 0.89, 0.85, 0.99],
             [0.88, 1., 0.83, 0.95]],
            dtype=torch.float32)
        self._values = torch.tensor(
            [[2.1, 1.1, -3.1, 0.0],
             [3.1, 0.1, -1.1, 7.4]],
            dtype=torch.float32)
        self._bootstrap_value = torch.tensor([8.4, -1.2], dtype=torch.float32)
        self._inputs = [
            self._rewards, self._discounts, self._rho_tm1,
            self._values, self._bootstrap_value]

        self._clip_rho_threshold = 1.0
        self._clip_pg_rho_threshold = 5.0
        self._lambda = 1.0

        self._expected_td = torch.tensor(
            [[-1.6155143, -3.4973226, 1.8670533, 5.0316002e1],
             [1.4662437, 3.6116405, -8.3327293e-5, -1.3540000e1]],
            dtype=torch.float32)
        self._expected_pg = torch.tensor(
            [[-1.6155143, -3.4973226, 1.8670534, 5.0316002e1],
             [1.4662433, 3.6116405, -8.3369283e-05, -1.3540000e+1]],
            dtype=torch.float32)

    def test_vtrace_td_error_and_advantage(self):
        """Tests for a full batch."""
        vtrace_td_error_and_advantage = functorch.vmap(functools.partial(
            vtrace.vtrace_td_error_and_advantage,
            clip_rho_threshold=self._clip_rho_threshold, lambda_=self._lambda))
        # Get function arguments.
        r_t, discount_t, rho_tm1, v_tm1, bootstrap_value = self._inputs
        v_t = torch.cat([v_tm1[:, 1:], bootstrap_value[:, None]], dim=1)
        # Compute vtrace output.
        vtrace_output = vtrace_td_error_and_advantage(
            v_tm1, v_t, r_t, discount_t, rho_tm1)
        # Test output.
        torch.testing.assert_close(
            self._expected_td, vtrace_output.errors, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(
            self._expected_pg, vtrace_output.pg_advantage, rtol=1e-3, atol=1e-3)

    @parameterized.named_parameters(
        ('vector_lambda',
         torch.tensor([[0., 1., 1., 0., 0., 1., 1., 1.]], dtype=torch.float32),
         torch.tensor([[1., 1., 1., 1., 1., 1., 1., 1.]], dtype=torch.float32)),
        ('vector_lambda_truncation',
         torch.tensor([[0., 1., 1., 1., 0., 1., 1., 1.]], dtype=torch.float32),
         torch.tensor([[1., 1., 1., 0., 1., 1., 1., 1.]], dtype=torch.float32)),
    )
    def test_vtrace_lambda_multiple_episodes_per_trace(self, discount_t, lambda_):
        """Tests for a full batch."""
        vtrace_ = functorch.vmap(
            functools.partial(
                vtrace.vtrace, clip_rho_threshold=self._clip_rho_threshold))
        # Get function arguments.
        r_t, rho_tm1, v_tm1 = torch.rand((3, 1, 8))
        bootstrap_value = torch.tensor([10.], dtype=torch.float32)
        v_t = torch.cat([v_tm1[:, 1:], bootstrap_value[:, None]], dim=1)
        # Full trace.
        vtrace_output = vtrace_(v_tm1, v_t, r_t, discount_t, rho_tm1, lambda_)
        # First episode trace.
        vtrace_output_ep1 = vtrace_(v_tm1[:, :4],
                                    v_t[:, :4], r_t[:, :4], discount_t[:, :4],
                                    rho_tm1[:, :4], lambda_[:, :4])
        vtrace_output_ep2 = vtrace_(v_tm1[:, 4:],
                                    v_t[:, 4:], r_t[:, 4:], discount_t[:, 4:],
                                    rho_tm1[:, 4:], lambda_[:, 4:])
        # Test output.
        torch.testing.assert_close(vtrace_output[:, :4], vtrace_output_ep1, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(vtrace_output[:, 4:], vtrace_output_ep2, rtol=1e-3, atol=1e-3)

    def test_lambda_q_estimate(self):
        """Tests for a full batch."""
        lambda_ = 0.8
        vtrace_td_error_and_advantage = functorch.vmap(functools.partial(
            vtrace.vtrace_td_error_and_advantage,
            clip_rho_threshold=self._clip_rho_threshold, lambda_=lambda_))
        # Get function arguments.
        r_t, discount_t, rho_tm1, v_tm1, bootstrap_value = self._inputs
        v_t = torch.cat([v_tm1[:, 1:], bootstrap_value[:, None]], dim=1)
        # Compute vtrace output.
        vtrace_output = vtrace_td_error_and_advantage(
            v_tm1, v_t, r_t, discount_t, rho_tm1)
        expected_vs = vtrace_output.errors + v_tm1
        clipped_rho_tm1 = torch.clamp_max(rho_tm1, self._clip_rho_threshold)
        vs_from_q = v_tm1 + clipped_rho_tm1 * (vtrace_output.q_estimate - v_tm1)
        # Test output.
        torch.testing.assert_close(expected_vs, vs_from_q, rtol=1e-3, atol=1e-3)


if __name__ == '__main__':
    absltest.main()
