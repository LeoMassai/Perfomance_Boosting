import torch
import torch.nn.functional as F


class RobotsSystem(torch.nn.Module):
    def __init__(self, xbar: torch.Tensor, linear_plant: bool, x_init=None, u_init=None, k: float = 1.0):
        """

        Args:
            xbar: concatenated nominal initial point of all agents
            linear_plant: if True, a linearized model of the system is used.
                             O.w., the model is non-lineardue to the dependence of friction on the speed.
            x_init: concatenated initial point of all agents. Defaults to xbar when None.
            u_init: initial input to the plant. Defaults to zero when None.
            k (float): gain of the pre-stabilizing controller (acts as a spring constant).
        """
        super().__init__()

        self.linear_plant = linear_plant

        # initial state
        self.register_buffer('xbar', xbar.reshape(1, -1))  # shape = (1, state_dim)
        x_init = self.xbar.detach().clone() if x_init is None else x_init.reshape(1, -1)  # shape = (1, state_dim)
        self.register_buffer('x_init', x_init)
        u_init = torch.zeros(1, int(self.xbar.shape[1] / 2)) if u_init is None else u_init.reshape(1,
                                                                                                   -1)  # shape = (1, in_dim)
        self.register_buffer('u_init', u_init)
        # check dimensions
        self.n_agents = int(self.xbar.shape[1] / 4)
        self.state_dim = 4 * self.n_agents
        self.in_dim = 2 * self.n_agents
        assert self.xbar.shape[1] == self.state_dim and self.x_init.shape[1] == self.state_dim
        assert self.u_init.shape[1] == self.in_dim

        self.h = 0.05
        self.mass = 1.0
        self.k = k
        self.b = 1.0
        self.b2 = None if self.linear_plant else 0.1
        m = self.mass
        B = torch.kron(torch.eye(self.n_agents),
                       torch.tensor([[0, 0],
                                     [0., 0],
                                     [1 / m, 0],
                                     [0, 1 / m]])
                       ) * self.h
        self.register_buffer('B', B)

        _A1 = torch.eye(4 * self.n_agents)
        _A2 = torch.cat((torch.cat((torch.zeros(2, 2),
                                    torch.eye(2)
                                    ), dim=1),
                         torch.cat((torch.diag(torch.tensor([-self.k / self.mass, -self.k / self.mass])),
                                    torch.diag(torch.tensor([-self.b / self.mass, -self.b / self.mass]))
                                    ), dim=1),
                         ), dim=0)
        _A2 = torch.kron(torch.eye(self.n_agents), _A2)
        A_lin = _A1 + self.h * _A2
        self.register_buffer('A_lin', A_lin)

        mask = torch.tensor([[0, 0], [1, 1]]).repeat(self.n_agents, 1)
        self.register_buffer('mask', mask)

    def A_nonlin(self, x):
        assert not self.linear_plant
        A3 = torch.norm(
            x.view(-1, 2 * self.n_agents, 2) * self.mask, dim=-1, keepdim=True
        )  # shape = (batch_size, 2 * n_agents, 1)
        A3 = torch.kron(
            A3, torch.ones(2, 1, device=A3.device)
        )  # shape = (batch_size, 4 * n_agents, 1)
        A3 = -self.b2 / self.mass * torch.diag_embed(
            A3.squeeze(dim=-1), offset=0, dim1=-2, dim2=-1
        )  # shape = (batch_size, 4 * n_agents, 4 * n_agents)
        A = self.A_lin + self.h * A3
        return A  # shape = (batch_size, 4 * n_agents, 4 * n_agents)

    def noiseless_forward(self, t, x: torch.Tensor, u: torch.Tensor):
        """
        forward of the plant without the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)

        Returns:
            next state of the noise-free dynamics.
        """
        x = x.view(-1, 1, self.state_dim)
        u = u.view(-1, 1, self.in_dim)
        if self.linear_plant:
            # x is batched but A is not => can use F.linear to compute xA^T
            f = F.linear(x - self.xbar, self.A_lin) + F.linear(u, self.B) + self.xbar
        else:
            # A depends on x, hence is batched. perform batched matrix multiplication
            f = torch.bmm(x - self.xbar, self.A_nonlin(x).transpose(1, 2)) + F.linear(u, self.B) + self.xbar
        return f  # shape = (batch_size, 1, state_dim)

    def forward(self, t, x, u, w):
        """
        forward of the plant with the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)

        Returns:
            next state.
        """
        return self.noiseless_forward(t, x, u) + w.view(-1, 1, self.state_dim)

    # simulation
    def rollout(self, controller, data, train=False):
        """
        rollout REN for rollouts of the process noise

        Args:
            - data: sequence of disturbance samples of shape
                (batch_size, T, state_dim).

        Rerurn:
            - x_log of shape = (batch_size, T, state_dim)
            - u_log of shape = (batch_size, T, in_dim)
        """

        # init
        controller.reset()
        x = self.x_init.detach().clone().repeat(data.shape[0], 1, 1)
        u = self.u_init.detach().clone().repeat(data.shape[0], 1, 1)

        # Simulate
        for t in range(data.shape[1]):
            x = self.forward(t=t, x=x, u=u, w=data[:, t:t + 1, :])  # shape = (batch_size, 1, state_dim)
            u = controller(x)  # shape = (batch_size, 1, in_dim)

            if t == 0:
                x_log, u_log = x, u
            else:
                x_log = torch.cat((x_log, x), 1)
                u_log = torch.cat((u_log, u), 1)

        controller.reset()
        if not train:
            x_log, u_log = x_log.detach(), u_log.detach()

        return x_log, None, u_log
