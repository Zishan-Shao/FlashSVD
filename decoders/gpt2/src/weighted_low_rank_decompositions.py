import torch

def get_low_rank(w, k):
    """
    Low-Rank decompostion via SVD.
    Parameters:
        w (torch.tensor, (m, n)) -- matrix to decompose
        k (int)                  -- desired rank
    
    Returns:
        a (torch.tensor, (m, k)) -- first factor
        b (torch.tensor, (k, n)) -- second factor
    """
    u, s, vt = torch.linalg.svd(w, full_matrices=False)
    u = u[:, :k]
    s = s[:k]
    vt = vt[:k]
    return u, torch.diag(s) @ vt


def weighted_svd(w, weight, rank, n_iter=30):
    """
    Weighted Low-Rank decompostion
    as in https://www.aaai.org/Papers/ICML/2003/ICML03-094.pdf.
    Parameters:
        w (torch.tensor, (m, n))      -- matrix to decompose
        weight (torch.tensor, (m, n)) -- weights of elements
        k (int)                       -- desired rank
        n_iter (int)                  -- number of iterations
    
    Returns:
        a (torch.tensor, (m, k)) -- first factor
        b (torch.tensor, (k, n)) -- second factor
    """
    a, b = get_low_rank(w, rank)
    
    if weight is None:
        return a, b

    for _ in range(n_iter):
        a, b = get_low_rank(weight * w + (1 - weight) * (a @ b), rank)
    
    return a, b


def nesterov(w, weight, k, n_iter=30):
    """
    Weighted Low-Rank decompostion with Nesterov acceleration
    as in https://arxiv.org/pdf/2109.11057.pdf, pages 9-10.
    Parameters:
        w (torch.tensor, (m, n))      -- matrix to decompose
        weight (torch.tensor, (m, n)) -- weights of elements
        k (int)                       -- desired rank
        n_iter (int)                  -- number of iterations
    
    Returns:
        a (torch.tensor, (m, k)) -- first factor
        b (torch.tensor, (k, n)) -- second factor
    """
    prev_a, prev_b = get_low_rank(w, k)
    prev_x = prev_a @ prev_b
    
    if weight is None:
        return prev_a, prev_b

    a, b = get_low_rank(weight * w + (1 - weight) * (prev_a @ prev_b), k)
    x = a @ b
    
    for i in range(1, n_iter):
        v = x + (i - 1) / (i + 2) * (x - prev_x)

        a, b = get_low_rank(weight * w + (1 - weight) * v, k)
        
        prev_x = x
        x = a @ b

    return a, b


def anderson(w, weight, k, n_iter=30, buffer_size=10, regularization_coef=0.0):
    """
    Weighted Low-Rank decompostion with Anderson acceleration
    as in https://arxiv.org/pdf/2109.11057.pdf, page 12.
    Parameters:
        w (torch.tensor, (m, n))      -- matrix to decompose
        weight (torch.tensor, (m, n)) -- weights of elements
        k (int)                       -- desired rank
        n_iter (int)                  -- number of iterations
        buffer_size (int)             -- number of residuals to store
        regularization_coef (float)   -- regularization coefficient in least squares problem
    
    Returns:
        a (torch.tensor, (m, k)) -- first factor
        b (torch.tensor, (k, n)) -- second factor
    """
    a, b = get_low_rank(w, k)

    if weight is None:
        return a, b

    residual_buffer = []
    approximations_buffer = []
    
    device = w.device

    y = w.clone()
    x = a @ b

    for i in range(n_iter):
        f = weight * w + (1 - weight) * x
        residual = (f - y).reshape(-1)
        
        residual_buffer.append(residual)
        approximations_buffer.append(f)

        r = len(residual_buffer)

        # shape (r, nm), if shape of w is (n, m)
        R = torch.stack(residual_buffer)
        alpha = torch.linalg.solve(R @ R.T + regularization_coef * torch.eye(r).to(device), torch.ones(r).to(device))
        alpha = alpha / alpha.sum()

        y = torch.sum(torch.stack(approximations_buffer) * alpha[:, None, None], dim=0)
        a, b = get_low_rank(y, k)
        x = a @ b

        if len(residual_buffer) >= buffer_size:
            residual_buffer = residual_buffer[1:]
        if len(approximations_buffer) >= buffer_size:
            approximations_buffer = approximations_buffer[1:]
    
    return a, b