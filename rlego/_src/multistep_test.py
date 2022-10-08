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
"""Unit tests for `multistep.py`."""
import functools

import functorch
import torch
from absl.testing import parameterized, absltest

from rlego._src import multistep


class LambdaReturnsTest(parameterized.TestCase):

    def setUp(self):
        self.lambda_ = 0.75
        self.r_t = torch.tensor(
            [[1.0, 0.0, -1.0, 0.0, 1.0], [0.5, 0.8, -0.7, 0.0, 2.1]])
        self.discount_t = torch.tensor(
            [[0.5, 0.9, 1.0, 0.5, 0.8], [0.9, 0.5, 0.3, 0.8, 0.7]])
        self.v_t = torch.tensor(
            [[3.0, 1.0, 5.0, -5.0, 3.0], [-1.7, 1.2, 2.3, 2.2, 2.7]])

        self.expected = torch.tensor(
            [[1.6460547, 0.72281253, 0.7375001, 0.6500001, 3.4],
             [0.7866317, 0.9913063, 0.1101501, 2.834, 3.99]],
            dtype=torch.float32)

    def test_lambda_returns_batch(self):
        """Tests for a full batch."""
        lambda_returns = functorch.vmap(multistep.lambda_returns, in_dims=0, out_dims=0)
        # Compute lambda return in batch.
        actual = lambda_returns(self.r_t, self.discount_t, self.v_t, lambda_=self.lambda_)
        # Test return estimate.
        torch.testing.assert_close(self.expected, actual, rtol=1e-5, atol=1e-5)


class DiscountedReturnsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

        self.r_t = torch.tensor(
            [[1.0, 0.0, -1.0, 0.0, 1.0], [0.5, 0.8, -0.7, 0.0, 2.1]])
        self.discount_t = torch.tensor(
            [[0.5, 0.9, 1.0, 0.5, 0.8], [0.9, 0.5, 0.3, 0.8, 0.7]])
        self.v_t = torch.tensor(
            [[3.0, 1.0, 5.0, -5.0, 3.0], [-1.7, 1.2, 2.3, 2.2, 2.7]])
        self.bootstrap_v = torch.tensor([v[-1] for v in self.v_t])

        self.expected = torch.tensor(
            [[1.315, 0.63000005, 0.70000005, 1.7, 3.4],
             [1.33592, 0.9288, 0.2576, 3.192, 3.9899998]],
            dtype=torch.float32)

    def test_discounted_returns_batch(self):
        """Tests for a single element."""
        discounted_returns = functorch.vmap(multistep.discounted_returns)
        # Compute discounted return.
        actual_scalar = discounted_returns(self.r_t, self.discount_t, self.bootstrap_v)
        actual_vector = discounted_returns(self.r_t, self.discount_t, self.v_t)
        # Test output.
        torch.testing.assert_close(self.expected, actual_scalar, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(self.expected, actual_vector, rtol=1e-5, atol=1e-5)


class NStepBootstrappedReturnsTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()
        self.r_t = torch.tensor(
            [[1.0, 0.0, -1.0, 0.0, 1.0], [0.5, 0.8, -0.7, 0.0, 2.1]])
        self.discount_t = torch.tensor(
            [[0.5, 0.9, 1.0, 0.5, 0.8], [0.9, 0.5, 0.3, 0.8, 0.7]])
        self.v_t = torch.tensor(
            [[3.0, 1.0, 5.0, -5.0, 3.0], [-1.7, 1.2, 2.3, 2.2, 2.7]])

        # Different expected results for different values of n.
        self.expected = {}
        self.expected[3] = torch.tensor(
            [[2.8, -3.15, 0.7, 1.7, 3.4], [1.2155, 0.714, 0.2576, 3.192, 3.99]],
            dtype=torch.float32)
        self.expected[5] = torch.tensor(
            [[1.315, 0.63, 0.7, 1.7, 3.4], [1.33592, 0.9288, 0.2576, 3.192, 3.99]],
            dtype=torch.float32)
        self.expected[7] = torch.tensor(
            [[1.315, 0.63, 0.7, 1.7, 3.4], [1.33592, 0.9288, 0.2576, 3.192, 3.99]],
            dtype=torch.float32)

    @parameterized.named_parameters(
        ('smaller_n', 3,), ('equal_n', 5,), ('bigger_n', 7,))
    def test_n_step_sequence_returns_batch(self, n):
        """Tests for a full batch."""
        n_step_returns = functorch.vmap(functools.partial(multistep.n_step_bootstrapped_returns, n=n))
        # Compute n-step return in batch.
        actual = n_step_returns(self.r_t, self.discount_t, self.v_t)
        # Test return estimate.
        torch.testing.assert_close(self.expected[n], actual, rtol=1e-5, atol=1e-5)

    def test_reduces_to_lambda_returns(self):
        """Test function is the same as lambda_returns when n is sequence length."""
        lambda_t = 0.75
        n = len(self.r_t[0])
        expected = multistep.lambda_returns(self.r_t[0], self.discount_t[0],
                                            self.v_t[0], lambda_t)
        actual = multistep.n_step_bootstrapped_returns(self.r_t[0],
                                                       self.discount_t[0],
                                                       self.v_t[0], n, lambda_t)
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-5)


class TDErrorTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

        self.r_t = torch.tensor(
            [[1.0, 0.0, -1.0, 0.0, 1.0], [0.5, 0.8, -0.7, 0.0, 2.1]])
        self.discount_t = torch.tensor(
            [[0.5, 0.9, 1.0, 0.5, 0.8], [0.9, 0.5, 0.3, 0.8, 0.7]])
        self.rho_tm1 = torch.tensor(
            [[0.5, 0.9, 1.3, 0.2, 0.8], [2., 0.1, 1., 0.4, 1.7]])
        self.values = torch.tensor(
            [[3.0, 1.0, 5.0, -5.0, 3.0, 1.], [-1.7, 1.2, 2.3, 2.2, 2.7, 2.]])

    def test_importance_corrected_td_errors_batch(self):
        """Tests equivalence to computing the error from a the lambda-return."""
        # Vmap and optionally compile.
        lambda_returns = functorch.vmap(multistep.lambda_returns)
        td_errors = functorch.vmap(multistep.importance_corrected_td_errors)
        # Compute multistep td-error with recursion on deltas.
        td_direct = td_errors(self.r_t, self.discount_t, self.rho_tm1, self.values,
                              torch.ones_like(self.discount_t))
        # Compute off-policy corrected return, and derive td-error from it.
        ls_ = torch.cat((self.rho_tm1[:, 1:], torch.tensor([[1.], [1.]])), dim=1)
        td_from_returns = self.rho_tm1 * (
                lambda_returns(self.r_t, self.discount_t, self.values[:, 1:], ls_) -
                self.values[:, :-1])
        # Check equivalence.
        torch.testing.assert_close(td_direct, td_from_returns, rtol=1e-5, atol=1e-5)


class TruncatedGeneralizedAdvantageEstimationTest(parameterized.TestCase):

    def setUp(self):
        super().setUp()

        self.r_t = torch.tensor([[0., 0., 1., 0., -0.5],
                                 [0., 0., 0., 0., 1.]])
        self.v_t = torch.tensor([[1., 4., -3., -2., -1., -1.],
                                 [-3., -2., -1., 0.0, 5., -1.]])
        self.discount_t = torch.tensor([[0.99, 0.99, 0.99, 0.99, 0.99],
                                        [0.9, 0.9, 0.9, 0.0, 0.9]])
        self.dummy_rho_tm1 = torch.tensor([[1., 1., 1., 1., 1],
                                           [1., 1., 1., 1., 1.]])
        self.array_lambda = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9],
                                          [0.9, 0.9, 0.9, 0.9, 0.9]])

        # Different expected results for different values of lambda.
        self.expected = {}
        self.expected[1.] = torch.tensor(
            [[-1.45118, -4.4557, 2.5396, 0.5249, -0.49],
             [3., 2., 1., 0., -4.9]],
            dtype=torch.float32)
        self.expected[0.7] = torch.tensor(
            [[-0.676979, -5.248167, 2.4846, 0.6704, -0.49],
             [2.2899, 1.73, 1., 0., -4.9]],
            dtype=torch.float32)
        self.expected[0.4] = torch.tensor(
            [[0.56731, -6.042, 2.3431, 0.815, -0.49],
             [1.725, 1.46, 1., 0., -4.9]],
            dtype=torch.float32)

    @parameterized.named_parameters(
        ('lambda1', 1.0),
        ('lambda0.7', 0.7),
        ('lambda0.4', 0.4))
    def test_truncated_gae(self, lambda_):
        """Tests truncated GAE for a full batch."""
        batched_advantage_fn_variant = functorch.vmap(
            multistep.truncated_generalized_advantage_estimation)
        actual = batched_advantage_fn_variant(
            self.r_t, self.discount_t, self.v_t, lambda_=lambda_)
        torch.testing.assert_close(self.expected[lambda_], actual, atol=1e-3, rtol=1e-5)

    def test_array_lambda(self):
        """Tests that truncated GAE is consistent with scalar or array lambda_."""
        scalar_lambda_fn = functorch.vmap(
            multistep.truncated_generalized_advantage_estimation)
        array_lambda_fn = functorch.vmap(
            multistep.truncated_generalized_advantage_estimation)
        scalar_lambda_result = scalar_lambda_fn(
            self.r_t, self.discount_t, self.v_t, lambda_=0.9)
        array_lambda_result = array_lambda_fn(
            self.r_t, self.discount_t, self.v_t, self.array_lambda)
        torch.testing.assert_close(scalar_lambda_result, array_lambda_result,
                                   atol=1e-3, rtol=1e-5)

    @parameterized.named_parameters(
        ('lambda_1', 1.0),
        ('lambda_0.7', 0.7),
        ('lambda_0.4', 0.4))
    def test_gae_as_special_case_of_importance_corrected_td_errors(self, lambda_):
        """Tests truncated GAE.

        Tests that truncated GAE yields same output as importance corrected
        td errors with dummy ratios.

        Args:
          lambda_: a lambda to use in GAE.
        """
        batched_gae_fn_variant = functorch.vmap(multistep.truncated_generalized_advantage_estimation)
        gae_result = batched_gae_fn_variant(
            self.r_t, self.discount_t, self.v_t, lambda_=lambda_)

        batched_ictd_errors_fn_variant = functorch.vmap(
            multistep.importance_corrected_td_errors)
        ictd_errors_result = batched_ictd_errors_fn_variant(
            self.r_t,
            self.discount_t,
            self.dummy_rho_tm1,
            self.v_t,
            torch.ones_like(self.discount_t) * lambda_,
        )
        torch.testing.assert_close(gae_result, ictd_errors_result, atol=1e-3, rtol=1e-5)


if __name__ == '__main__':
    absltest.main()
