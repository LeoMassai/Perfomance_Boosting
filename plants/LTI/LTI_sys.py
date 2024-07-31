import torch
from utils.assistive_functions import to_tensor


# ---------- SYSTEM ----------
class LTISystem:
    def __init__(self, A, B, C, x_init):
        self.A, self.B, self.C = to_tensor(A), to_tensor(B), to_tensor(C)
        self.x_init = to_tensor(x_init)

        # Dimensions
        self.state_dim = self.A.shape[0]
        self.in_dim = self.B.shape[1]
        self.out_dim = self.C.shape[0]
        # Check matrices
        assert self.A.shape == (self.state_dim, self.state_dim)
        assert self.B.shape == (self.state_dim, self.in_dim)
        assert self.C.shape == (self.out_dim, self.state_dim)
        assert self.x_init.shape == (self.state_dim, 1)

    # simulation
    def rollout(self, controller, data: torch.Tensor):
        """
        rollout with state-feedback controller

        Args:
            - controller: state-feedback controller
            - data (torch.Tensor): batch of disturbance samples, with shape (batch_size, T, state_dim)
        """
        xs = (data[:, 0:1, :] + self.x_init)
        us = controller.forward(xs[:, 0:1, :])
        ys = torch.matmul(self.C, xs[:, 0:1, :])
        for t in range(1, data.shape[1]):
            xs = torch.cat(
                (
                    xs,
                    torch.matmul(self.A, xs[:, t - 1:t, :]) + torch.matmul(self.B, us[:, t - 1:t, :]) + data[:, t:t + 1,
                                                                                                        :]),
                1
            )
            ys = torch.cat(
                (ys, torch.matmul(self.C, xs[:, t:t + 1, :])),
                1
            )
            us = torch.cat(
                (us, controller.forward(xs[:, t:t + 1, :])),
                1
            )
        return xs, ys, us
