import torch
from torch.func import functional_call, jacrev, vjp, vmap


def compute_trunk_metric(T, weights):
    return torch.einsum("qci,qcj,q->ij", T, T, weights)


def branch_parameter_jacobian(branch, z):
    params = dict(branch.named_parameters())
    buffers = dict(branch.named_buffers())

    def forward(params, x):
        return functional_call(branch, (params, buffers), (x.unsqueeze(0),)).squeeze(0)

    jac = jacrev(forward, argnums=0)(params, z)

    return torch.cat(
        [value.reshape(value.shape[0], -1) for value in jac.values()],
        dim=1,
    )


def metric_factor(M):
    M = 0.5 * (M + M.T)

    eigenvalues, Q = torch.linalg.eigh(M)
    eigenvalues = eigenvalues.clamp_min(0)

    L = torch.diag(torch.sqrt(eigenvalues)) @ Q.T

    return L


@torch.no_grad()
def weighted_feature(J, L):
    return (L @ J).reshape(-1)


def weighted_branch_gram(branch, sdf_batch, L, chunk=16):
    """
    K[i, j] = <L J_i, L J_j>_F, the same as stacking weighted_feature rows into F and
    taking F @ F.T, but without forming J ([n_branch, n_params]) or F. Row r of L J_i is
    the parameter gradient of L[r] . branch(z_i), so it is built from VJPs, chunk rows at
    a time.
    """
    params = {k: v.detach() for k, v in branch.named_parameters()}
    buffers = dict(branch.named_buffers())

    def pullback(z):
        def forward(params):
            return functional_call(branch, (params, buffers), (z.unsqueeze(0),)).squeeze(0)

        return vjp(forward, params)[1]

    pullbacks = [pullback(z) for z in sdf_batch]
    K = torch.zeros(len(sdf_batch), len(sdf_batch), dtype=L.dtype)
    for rows in L.split(chunk):
        # G: [n_samples, chunk, n_params], rows of L J_i for this chunk
        G = torch.stack(
            [
                torch.cat([g.reshape(len(rows), -1) for g in vmap(f)(rows)[0].values()], dim=1)
                for f in pullbacks
            ]
        )
        K += torch.einsum("icp,jcp->ij", G, G)
    return 0.5 * (K + K.T)
