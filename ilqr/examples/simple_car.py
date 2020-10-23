# Copyright (C) 2018, Anass Al
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>
"""Cartpole example."""

import numpy as np
import theano.tensor as T
from ..dynamics import BatchAutoDiffDynamics, tensor_constrain



class SimpleCarDynamics(BatchAutoDiffDynamics):

    """SimpleCar auto-differentiated dynamics model."""

    def __init__(self,
                 dt,
                 constrain=True,
                 min_bounds=-1.0,
                 max_bounds=1.0,
                 **kwargs):
        """Cartpole dynamics.

        Args:
            dt: Time step [s].
            constrain: Whether to constrain the action space or not.
            min_bounds: Minimum bounds for action [N].
            max_bounds: Maximum bounds for action [N].
            mc: Cart mass [kg].
            mp: Pendulum mass [kg].
            l: Pendulum length [m].
            g: Gravity acceleration [m/s^2].
            **kwargs: Additional key-word arguments to pass to the
                AutoDiffDynamics constructor.

        Note:
            state: [x, x', sin(theta), cos(theta), theta']
            action: [F]
            theta: 0 is pointing up and increasing clockwise.
        """
        self.constrained = constrain
        self.min_bounds = min_bounds
        self.max_bounds = max_bounds

        def f(x, u, i):
            # Constrain action space.
            if constrain:
                u = tensor_constrain(u, min_bounds, max_bounds)
                # u = T.set_subtensor(u[...,0], T.tanh(u[...,0]) + 1) # if you want to bound control

            x_ = x[..., 0]
            y = x[..., 1]
            v = x[..., 2]
            sin_theta = x[..., 3]
            cos_theta = x[..., 4]

            F = u[..., 0]
            w = u[..., 1]

            x_dot = v * cos_theta
            y_dot = v * sin_theta
            v_dot = F
            theta_dot = w

            theta = T.arctan2(sin_theta, cos_theta)
            next_theta = theta + theta_dot * dt

            return T.stack([
                x_ + x_dot * dt,
                y + y_dot * dt,
                v + v_dot * dt,
                T.sin(next_theta),
                T.cos(next_theta),
            ]).T

        super(SimpleCarDynamics, self).__init__(f,
                                               state_size=5,
                                               action_size=2,
                                               **kwargs)

    @classmethod
    def augment_state(cls, state):
        """Augments angular state into a non-angular state by replacing theta
        with sin(theta) and cos(theta).

        In this case, it converts:

            [x, x', theta, theta'] -> [x, x', sin(theta), cos(theta), theta']

        Args:
            state: State vector [reducted_state_size].

        Returns:
            Augmented state size [state_size].
        """
        if state.ndim == 1:
            x, y, v, theta = state
        else:
            x = state[..., 0].reshape(-1, 1)
            y = state[..., 1].reshape(-1, 1)
            v = state[..., 2].reshape(-1, 1)
            theta = state[..., 3].reshape(-1, 1)

        return np.hstack([x, y, v, np.sin(theta), np.cos(theta)])

    @classmethod
    def reduce_state(cls, state):
        """Reduces a non-angular state into an angular state by replacing
        sin(theta) and cos(theta) with theta.

        In this case, it converts:

            [x, x', sin(theta), cos(theta), theta'] -> [x, x', theta, theta']

        Args:
            state: Augmented state vector [state_size].

        Returns:
            Reduced state size [reducted_state_size].
        """
        if state.ndim == 1:
            x, y, v, sin_theta, cos_theta = state
        else:
            x = state[..., 0].reshape(-1, 1)
            y = state[..., 1].reshape(-1, 1)
            v = state[..., 2].reshape(-1, 1)
            sin_theta = state[..., 3].reshape(-1, 1)
            cos_theta = state[..., 4].reshape(-1, 1)

        theta = np.arctan2(sin_theta, cos_theta)
        return np.hstack([x, y, v, theta])