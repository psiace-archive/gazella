# Copyright 2023 Chojan Shang.
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

# Automatic Gradient Descent - https://arxiv.org/pdf/2304.05187.pdf.
# You can find the original implementation by the creators of the paper here - https://github.com/jxbz/agd.

import math
from typing import Callable, Iterable

import torch
from torch.nn.init import orthogonal_
from torch.optim import Optimizer


class AGD(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], gain: float = 1.0):
        defaults = dict(gain=gain)
        super().__init__(params, defaults=defaults)
        self.params = self.param_groups[0]["params"]
        self.gain = gain

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
        if self.state["initialized"]:
            self._init_weights()
            self.state["initialized"] = True

        loss = agd(
            params=self.params, get_largest_singular_value=self._scale, gain=self.gain
        )

        return loss

    @torch.no_grad()
    def _init_weights(self):
        for p in self.params:
            if p.dim() == 1:
                raise Exception("AGD doesn't support biases.")

            # sample a semi-orthogonal matrix
            self._orthogonalize(p, dim=p.dim())
            # rescale its singular values
            self._scale(p)

    @staticmethod
    @torch.no_grad()
    def _orthogonalize(self, weights: torch.Tensor, dim: int):
        if dim == 2:
            orthogonal_(weights)
        if dim == 4:
            [
                orthogonal_(weights[:, :, x, y])
                for x in range(weights.shape[2])
                for y in range(weights.shape[3])
            ]

    @torch.no_grad()
    def _scale(self, weights: torch.Tensor):
        singular_values_approx = math.sqrt(weights.shape[0] / weights.shape[1])

        if weights.dim() == 4:
            singular_values_approx /= math.sqrt(weights.shape[2] * weights.shape[3])

        return singular_values_approx


def agd(params: Iterable[torch.Tensor], get_largest_singular_value: Callable, gain=1.0):
    grad_summary = 0
    num_layers = len(list(params))

    # get gradient summary
    grad_summary = sum(
        p.grad.norm(dim=(0, 1)).sum() * get_largest_singular_value(p) for p in params
    )

    grad_summary_scale = grad_summary / num_layers
    # set automatic learning rate
    learning_rate = math.log((1 + math.sqrt(1 + 4 * grad_summary_scale)) / 2)

    for p in params:
        # update weights
        p -= (
            (gain * learning_rate / num_layers)
            * (p.grad / p.grad.norm(dim=(0, 1), keepdim=True))
            * get_largest_singular_value(p)
        )
