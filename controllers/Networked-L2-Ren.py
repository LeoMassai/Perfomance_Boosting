import torch
import torch.nn as nn
from Robust_REN import REN


# Stable networked operator made by RENs with fully trainable l2 gains and interconnection matrices

class NetworkedRENs(nn.Module):
    def __init__(self, N, Muy, Mud, Mey, m, p, n, l, top=True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.device = device
        self.top = top  # If set to True, the topology of M is preserved, otherwise it trains a
        # potentially full matrix Q
        self.p = p  # output dimension for each REN
        self.m = m  # input dimension for each REN
        self.n = n  # state dimension for each REN
        self.l = l  # number of nonlinear layers for each REN
        self.Muy = Muy
        self.Mud = Mud
        self.Mey = Mey
        self.diag_params = nn.Parameter(torch.randn(sum(p)))  # For trainable Mey matrix
        self.N = N
        self.r = nn.ModuleList([REN(self.m[j], self.p[j], self.n[j], self.l[j]) for j in range(N)])
        self.s = nn.Parameter(torch.randn(N, device=device))
        self.gammaw = torch.nn.Parameter(4 * torch.randn(1, device=device))
        if top:
            # Create a mask where M is non-zero
            self.mask = Muy.ge(0.1)
            # Count the number of non-zero elements in M
            num_params = self.mask.sum().item()
            # Initialize the trainable parameters
            self.params = nn.Parameter(0.03 * torch.randn(num_params))
            # Create a clone of M to create Q (the trainable version of M)
            self.Q = Muy.clone()
        else:
            self.Q = nn.Parameter(0.01 * torch.randn((sum(m), sum(p))))

    def forward(self, t, d, x, checkLMI=False):
        # checkLMI if set to True, checks if the dissipativity LMI is satisfied at every step
        Q = self.Q
        if self.top:
            params = self.params
            # Assign the parameters to the corresponding positions in Q
            masked_values = torch.zeros_like(Q, device=self.device)
            masked_values[self.mask] = params
            Q = masked_values

        gammaw = self.gammaw
        #Mey = self.Mey
        tMey = torch.diag(self.diag_params)
        H = torch.matmul(tMey.T, tMey)
        sp = torch.abs(self.s)
        gamma_list = []
        C2s = []
        D22s = []
        row_sum = torch.sum(self.Mud, 1)
        A1t = torch.nonzero(row_sum == 1, as_tuple=False).squeeze(dim=1)
        A0t = torch.nonzero(row_sum == 0, as_tuple=False).squeeze(dim=1)
        uindex = []
        yindex = []
        xindex = []
        startu = 0
        starty = 0
        startx = 0
        pesi = torch.zeros(self.N)
        for j, l in enumerate(self.r):
            # Free parametrization of individual l2 gains ensuring stability of networked REN
            xi = torch.arange(startx, startx + l.n)
            ui = torch.arange(startu, startu + l.m)
            yi = torch.arange(starty, starty + l.p)
            setu = set(ui.numpy())
            A1 = torch.tensor(list(setu.intersection(set(A1t.numpy()))), device=self.device)
            A0 = torch.tensor(list(setu.intersection(set(A0t.numpy()))), device=self.device)
            a = H[j, j] + torch.max(torch.stack([torch.sum(torch.abs(Q[:, j])) for j in yi])) + sp[j]
            #a = H[j, j] + torch.max(torch.stack([torch.sum(torch.abs(Q[:, j])) for j in yi]))
            pesi[j] = a
            if A0.numel() != 0:
                if A1.numel() != 0:
                    gamma = torch.sqrt(
                        1 / a * torch.minimum(gammaw ** 2 / (torch.max(torch.stack([torch.sum(torch.abs(Q[j, :]))
                                                                                    for j in
                                                                                    A1])) * gammaw ** 2 + 1),
                                              1 / (torch.max(torch.stack([torch.sum(torch.abs(Q[j, :]))
                                                                          for j in
                                                                          A0])))))
                else:
                    gamma = torch.sqrt(1 / (a * torch.max(torch.stack([torch.sum(torch.abs(Q[j, :]))
                                                                       for j in
                                                                       A0]))))
            else:
                gamma = torch.sqrt(1 / a * (gammaw ** 2 / (torch.max(torch.stack([torch.sum(torch.abs(Q[j, :]))
                                                                                  for j in
                                                                                  A1])) * gammaw ** 2 + 1)))

            l.set_param(gamma)
            gamma_list.append(gamma)
            C2s.append(l.C2)
            D22s.append(l.D22)
            startu += l.m
            starty += l.p
            startx += l.n
            uindex.append(ui)
            yindex.append(yi)
            xindex.append(xi)
        C2 = torch.block_diag(*C2s)
        D22 = torch.block_diag(*D22s)
        # compute the stacked input for each REN
        u = torch.matmul(torch.inverse(torch.eye(self.Muy.size(0)) - torch.matmul(Q, D22)),
                         (torch.matmul(torch.matmul(Q, C2), x)) + torch.matmul(self.Mud, d))
        y_list = []
        x_list = []
        # update REN dynamics
        for j, l in enumerate(self.r):
            yt, xtemp = l(u[uindex[j]], x[xindex[j]], t)
            y_list.append(yt)
            x_list.append(xtemp)

        y = torch.cat(y_list)
        x_ = torch.cat(x_list)
        e = torch.matmul(tMey, y)
        gammawout = gammaw ** 2

        # check Dissipativity LMI
        if checkLMI:
            with torch.no_grad():
                Nu = torch.block_diag(*[pesi[j] * gamma_list[j] ** 2 * torch.eye(self.m[j]) for j in range(self.N)])
                Ny = torch.block_diag(*[pesi[j] * torch.eye(self.p[j]) for j in range(self.N)])
                Xi = torch.block_diag(Nu, -Ny)
                S = torch.block_diag(gammawout * torch.eye(sum(self.m)), -torch.eye(sum(self.p)))
                XiS = torch.block_diag(Xi, -S)

                M1 = torch.hstack((Q.data, self.Mud))
                M2 = torch.hstack((torch.eye(sum(self.p)), torch.zeros((sum(self.p), sum(self.m)))))
                M3 = torch.hstack((torch.zeros((sum(self.m), sum(self.p))), torch.eye(sum(self.m))))
                M4 = torch.hstack((tMey, torch.zeros((sum(self.p), sum(self.m)))))
                M = torch.vstack((M1, M2, M3, M4))
                lmi = M.T @ XiS @ M
                lmip = torch.linalg.eigvals(lmi)

        return e, x_, gamma_list, gammawout, Q, lmip
