import torch
import torch.nn.functional as F
import numpy as np
import argparse
import torch.nn as nn
import wandb
from pytorch_metric_learning.losses import ContrastiveLoss
from scipy.linalg import eigh
import torch
import numpy as np
from scipy.linalg import eigh

# Argument parser
parser = argparse.ArgumentParser(description="Train (C, r)-generalizing ReLU network")
parser.add_argument("--c", type=int, default=3, help="Number of bits for generalizing solution")
parser.add_argument("--d", type=int, default=30, help="Input dimension")
parser.add_argument("--hidden", type=int, default=120, help="hidden diemension in neural network")
parser.add_argument("--scale", type=float, default=0.01, help="Scale of initial perturbation")
parser.add_argument("--n", type=int, default=100, help="Number of data points")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=0.1, help="Grid of learning rates")
parser.add_argument("--batch_size",  type=int, default=10, help="Grid of batch sizes")
parser.add_argument("--mode", type=str, choices=["SGD", "SAM"], default="SGD", help="Optimization method")
parser.add_argument("--seed", type=int, default=0, help="random seed")
parser.add_argument("--rho", type=float, default=0.1)
parser.add_argument("--experiment", type=str, choices=["purturb", "coherence"], default="coherence", help="Experiment method")
parser.add_argument("--use_metric", type=bool, default=False, help="Using contrastive learning loss or not")
parser.add_argument("--lambda_metric", type=float, default=0.1)
args = parser.parse_args()
wandb.init(project="new_stability-analysis_modify", config=vars(args), name=f"nn_mode_{args.mode}_rho_{args.rho}")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU

    # For reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class TwoLayerReLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(hidden_dim, 1, bias=False)  # bias removed here

    def features(self, x):
        return self.feature_extractor(x)

    def forward(self, x):
        features = self.features(x)
        return self.output_layer(features).squeeze(-1)

def evaluate(model, X_val, y_val):
    model.eval()
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        preds = model(X_val)
        loss = loss_fn(preds, y_val)
        preds_sign = torch.sign(preds)  # since true y ∈ {−1, 1}
        acc = (preds_sign == y_val).float().mean()
    model.train()
    return acc.item(), loss.item()

def generate_data(n, d, device='cuda'):
    X = (torch.randint(0, 2, (n, d), device=device) * 2 - 1).float()  # {-1, 1} values
    y = (X[:, 0] * X[:, 1]).float()                                   # label in {-1, 1}
    return X, y

def initialize_generalizing_solution(model, C):
    with torch.no_grad():
        d = model.net[0].in_features
        num_units = model.net[0].out_features
        r = (d + 1)**0.25

        W1 = torch.zeros((num_units, d), device=model.net[0].weight.device)
        b1 = torch.zeros((num_units,), device=model.net[0].bias.device)
        W2 = torch.zeros((1, num_units), device=model.net[2].weight.device)

        for i in range(2**C):
            a = [int(x) for x in list(np.binary_repr(i, width=C))]
            signs = torch.tensor([(-1)**ai for ai in a], dtype=torch.float32, device=W1.device)
            W1[i, :C] = r * signs
            b1[i] = -r * (C - 1)
            W2[0, i] = (-1)**(sum(a[:2])) / r

        model.net[0].weight.copy_(W1)
        model.net[0].bias.copy_(b1)
        model.net[2].weight.copy_(W2)

def add_perturbation_to_model(model, scale=0.01):
    with torch.no_grad():
        model.net[0].weight.add_(torch.randn_like(model.net[0].weight) * scale)
        model.net[0].bias.add_(torch.randn_like(model.net[0].bias) * scale)
        model.net[2].weight.add_(torch.randn_like(model.net[2].weight) * scale)

def train_sam(model, X, y, X_val, y_val, rho=0.05, lr=0.1, batch_size=32, epochs=50):
    
    model = model.to(X.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    steps = 0

    for epoch in range(epochs):
        idx = torch.randperm(X.size(0))
        X, y = X[idx], y[idx]
            
        acc, loss = evaluate(model, X_val, y_val)
        wandb.log({"val_acc":acc, "val_loss":loss, "epoch":epoch})

        for i in range(0, X.size(0), batch_size):
            xb = X[i:i + batch_size]
            yb = y[i:i + batch_size]

            # First forward-backward pass
            preds = model(xb)
            loss = loss_fn(preds, yb)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            # Compute norm of gradient
            grad_norm = torch.norm(torch.cat([g.view(-1) for g in grads]))
            scale = rho / (grad_norm + 1e-12)
            # Step 1: Save original weights
            orig_params = [p.clone() for p in model.parameters()]

            # Step 2: Perturb
            for p, g in zip(model.parameters(), grads):
                p.data.add_(scale * g)

            # Step 3: Compute loss and gradients at perturbed weights
            preds_perturbed = model(xb)
            loss_perturbed = loss_fn(preds_perturbed, yb)
            grads_perturbed = torch.autograd.grad(loss_perturbed, model.parameters())

            # Step 4: Restore original weights BEFORE optimizer step
            for p, orig in zip(model.parameters(), orig_params):
                p.data.copy_(orig)

            # Step 5: Manually set gradients and call optimizer step
            optimizer.zero_grad()
            for p, g in zip(model.parameters(), grads_perturbed):
                p.grad = g
            optimizer.step()

            preds_sign = torch.sign(preds)  # since true y ∈ {−1, 1}
            acc = (preds_sign == yb).float().mean()
            wandb.log({"acc":acc, "train_loss":loss.item(), "epoch":epoch, "steps":steps})
            steps+=1
    return model

def train_sgd(model, X, y, X_val, y_val, lr=0.1, batch_size=32, epochs=50):
    model = model.to(X.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    steps = 0
    for epoch in range(epochs):
        acc, loss = evaluate(model, X_val, y_val)
        wandb.log({"val_acc":acc, "val_loss":loss, "epoch":epoch})
        perm = torch.randperm(X.size(0), device=X.device)
        for i in range(0, X.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X[idx], y[idx]

            optimizer.zero_grad()
            preds = model(xb)
            loss = F.mse_loss(preds, yb)
            loss.backward()
            optimizer.step()

            preds_sign = torch.sign(preds)  # since true y ∈ {−1, 1}
            acc = (preds_sign == yb).float().mean()
            wandb.log({"acc":acc, "train_loss":loss.item(), "epoch":epoch, "steps":steps})
            steps+=1
    return loss.item()

def compute_full_hessian(loss, model):
    """
    Compute the full Hessian of the loss w.r.t. all parameters in the model.
    Returns a 2D tensor of shape (N, N) where N is the total number of parameters.
    """
    # Flatten all parameters into a single vector
    params = [p for p in model.parameters() if p.requires_grad]
    flat_params = torch.cat([p.contiguous().view(-1) for p in params])
    grad = torch.autograd.grad(loss, params, create_graph=True)
    grad_flat = torch.cat([g.contiguous().view(-1) for g in grad])

    n_params = grad_flat.size(0)
    hessian = torch.zeros(n_params, n_params, device=grad_flat.device)

    for i in range(n_params):
        grad2 = torch.autograd.grad(grad_flat[i], params, retain_graph=True)
        grad2_flat = torch.cat([g.contiguous().view(-1) for g in grad2])
        hessian[i] = grad2_flat.detach()

    return hessian

def largest_eigenvalue(matrix):
    # Make sure the matrix is on the right device and dtype
    assert matrix.size(0) == matrix.size(1), "Matrix must be square."
    eigenvalues = torch.linalg.eigvalsh(matrix)  # For symmetric (Hermitian) matrices
    largest = eigenvalues.max().item()
    return largest

def safe_eigh(A, jitter=1e-8, max_tries=6, return_cpu=False, up='L'):
    """
    Robust eigen-decomposition for (nearly) symmetric matrices.
    Returns eigenvalues (ascending), eigenvectors.
    """
    assert A.dim() >= 2 and A.shape[-1] == A.shape[-2], "A must be square"
    # 0) sanitize
    if torch.isnan(A).any() or torch.isinf(A).any():
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # 1) symmetrize explicitly
    A = 0.5 * (A + A.transpose(-1, -2))

    # 2) scale to unit magnitude to avoid overflow/underflow
    scale = A.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1.0)
    A = A / scale

    # 3) bump precision if possible
    if A.dtype in (torch.float16, torch.bfloat16, torch.float32):
        A = A.to(torch.float64)

    # 4) try with increasing diagonal jitter, then try CPU as fallback
    err = None
    for attempt in range(max_tries):
        eps = jitter * (10.0 ** attempt)
        try:
            A_j = A + eps * torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
            eigenvalues = torch.linalg.eigvalsh(A_j)* scale.squeeze(-1).squeeze(-1)
            largest = eigenvalues.max().item()
            return largest
        except RuntimeError as e:
            err = e

    # 5) CPU fallback (often more forgiving)
    try:
        A_cpu = A.detach().cpu()
        for attempt in range(max_tries):
            eps = jitter * (10.0 ** attempt)
            A_j = A_cpu + eps * torch.eye(A_cpu.shape[-1], dtype=A_cpu.dtype)
            eigenvalues = torch.linalg.eigvalsh(A_j)* scale.squeeze(-1).squeeze(-1)
            largest = eigenvalues.max().item()
            return largest
    except RuntimeError as e2:
        err = e2
    # 6) last resort: PSD-only SVD path (only valid if you *know* A ≽ 0)
    try:
        # if A is PSD, SVD equals eigen-decomp
        eigenvalues = torch.linalg.eigvalsh(A)* scale.squeeze(-1).squeeze(-1)
        largest = eigenvalues.max().item()
        # w = S * scale.squeeze(-1).squeeze(-1)
        # V = U
        return largest
    except RuntimeError as e3:
        raise RuntimeError(
            f"safe_eigh failed after all remedies. Last error: {err}\n"
            "Hints: check symmetry, remove NaNs/Infs, add more jitter, or use CPU."
        ) from e3

def compute_coherence(x, y, model):
    n = x.shape[0]
    device = x.device
    coherence = torch.zeros((n, n), device=device)
    hessians = [None] * n
    largest = torch.tensor(0.0, device=device).item()

    def get_h(i):
        if hessians[i] is None:
            # NOTE: keep grads enabled inside this block
            with torch.enable_grad():
                for p in model.parameters():
                    p.requires_grad_(True)
                loss_i = F.mse_loss(model(x[i:i+1]), y[i:i+1], reduction='mean')
                H = compute_full_hessian(loss_i, model)  # returns symmetric (P×P)
            # numeric hygiene
            H = 0.5 * (H + H.T)
            hessians[i] = H.detach()
        return hessians[i]

    for i in range(n):
        H_i = get_h(i)
        # top eigenvalue once per sample
        lam_i = safe_eigh(H_i)  # or torch.linalg.eigvalsh(H_i)
        largest = max(largest, lam_i)
        coherence[i, i] = torch.sqrt(torch.trace(H_i @ H_i))
        for j in range(i+1, n):
            H_j = get_h(j)
            # (i,j)
            val = torch.sqrt(torch.trace(H_i @ H_j))
            coherence[i, j] = val
            coherence[j, i] = val

    # return spectrum of the coherence matrix and the largest single-sample Hessian eig
    coh_eigs = safe_eigh(coherence)
    return coh_eigs, largest

def effective_rank_pca(features: np.ndarray, threshold: float = 0.9):
    """
    Compute the number of principal components needed to explain a given
    percentage of variance (default 90%) from the feature matrix.

    Args:
        features (np.ndarray): Feature matrix of shape (n_samples, n_features)
        threshold (float): Variance threshold (e.g., 0.9 for 90%)

    Returns:
        int: Effective rank
        np.ndarray: Array of explained variances
    """
    # Center the features
    features_centered = features - np.mean(features, axis=0)

    # Compute covariance matrix
    cov = np.cov(features_centered, rowvar=False)

    # Eigenvalues in descending order
    eigvals = np.linalg.eigvalsh(cov)[::-1]

    # Normalize eigenvalues to get variance ratios
    explained_variance_ratio = eigvals / eigvals.sum()

    # Cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Find number of components to reach threshold
    effective_rank = np.searchsorted(cumulative_variance, threshold) + 1
    explained_ratio_at_rank = float(cumulative_variance[effective_rank - 1])
    
    return effective_rank, explained_ratio_at_rank

def compute_hessian_data(x,y,model):
    # Compute total loss over dataset
    outputs = model(x)
    loss = F.mse_loss(outputs, y)

    # Compute Hessian of total loss
    hessian = compute_full_hessian(loss, model)
    largest = safe_eigh(hessian)
    trace = torch.trace(hessian)
    return largest, trace

def experiment_purturb(args):
    set_seed(args.seed)
    x, y = generate_data(args.n, args.d)
    X_val, y_val = generate_data(args.n, args.d)

    model = TwoLayerReLU(args.d, args.hidden).to(x[0].device)
    initialize_generalizing_solution(model, args.c)
    add_perturbation_to_model(model, scale=args.scale)

    if args.mode == "SGD":
        train_sgd(model, x, y, X_val, y_val, args.lr, args.batch_size, args.epochs)
    elif args.mode == "SAM":
        train_sam(model, x, y, X_val, y_val, args.rho, args.lr, args.batch_size, args.epochs)

def coherence_sgd(model, X, y, X_val, y_val, lr=0.1, batch_size=32, epochs=50):
    model = model.to(X.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    steps = 0
    if args.use_metric:
        contrastive_loss_fn = ContrastiveLoss(pos_margin=0, neg_margin=1)

    for epoch in range(epochs):
        perm = torch.randperm(X.size(0), device=X.device)
        acc, loss = evaluate(model, X_val, y_val)
        if epoch >= 5:
            lambda_coherence, largest = compute_coherence(X,y,model)
        else:
            lambda_coherence, largest = 1,1
        wandb.log({"val_acc":acc, 
                   "val_loss":loss, 
                   "epoch":epoch, 
                   "coherence matrix":lambda_coherence, 
                   "largest":largest, 
                   "coherence": lambda_coherence / largest})

        for i in range(0, X.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X[idx], y[idx]

            optimizer.zero_grad()
            preds = model(xb)
            loss = F.mse_loss(preds, yb)

            if args.use_metric:
                features = model.features(xb)
                contrastive_loss = contrastive_loss_fn(features, (yb + 1) // 2)  # Convert {-1,1} → {0,1}
                loss = loss + args.lambda_metric * contrastive_loss

            loss.backward()
            optimizer.step()

            preds_sign = torch.sign(preds)  # since true y ∈ {−1, 1}
            acc = (preds_sign == yb).float().mean()
            largest_hessian, hessian_trace = compute_hessian_data(X, y, model)
            features = model.features(X).detach().cpu().numpy()
            effective_rank, explained_variance_ratio = effective_rank_pca(features)
            wandb.log({"acc":acc, 
                       "train_loss":loss.item(), 
                       "epoch":epoch, 
                       "steps":steps, 
                       "largest_hessian":largest_hessian,
                       "hessian_trace":hessian_trace, 
                       "effective_rank":effective_rank, 
                       "explained_variance_ratio":explained_variance_ratio})
            steps+=1
    return loss.item()

def coherence_sam(model, X, y, X_val, y_val, rho=0.05, lr=0.1, batch_size=32, epochs=50):
    model = model.to(X.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    steps = 0

    if args.use_metric:
        contrastive_loss_fn = ContrastiveLoss(pos_margin=0, neg_margin=1)

    for epoch in range(epochs):
        # shuffling
        idx = torch.randperm(X.size(0))
        X, y = X[idx], y[idx]

        acc, loss = evaluate(model, X_val, y_val)
        if epoch >= 5:
            lambda_coherence, largest = compute_coherence(X, y, model)
        else:
            lambda_coherence, largest = 1, 1

        wandb.log({
            "val_acc": acc,
            "val_loss": loss,
            "epoch": epoch,
            "coherence matrix": lambda_coherence,
            "largest": largest,
            "coherence": lambda_coherence / largest,
        })

        for i in range(0, X.size(0), batch_size):
            xb = X[i:i + batch_size]
            yb = y[i:i + batch_size]

            # Step 1: Forward and compute gradient at original weights
            preds = model(xb)
            loss = loss_fn(preds, yb)
            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

            # Step 2: Compute perturbation scale and apply perturbation
            grad_norm = torch.norm(torch.cat([g.view(-1) for g in grads]))
            scale = rho / (grad_norm + 1e-12)
            orig_params = [p.clone() for p in model.parameters()]

            for p, g in zip(model.parameters(), grads):
                p.data.add_(scale * g)

            # Step 3: Compute SAM gradient at perturbed weights
            preds_perturbed = model(xb)
            loss_perturbed = loss_fn(preds_perturbed, yb)
            grads_sam = torch.autograd.grad(loss_perturbed, model.parameters(), retain_graph=args.use_metric)

            # Step 4: Restore original weights
            for p, orig in zip(model.parameters(), orig_params):
                p.data.copy_(orig)

            # Step 5: Compute metric learning gradient at original weights
            if args.use_metric:
                features_original = model.features(xb)
                labels = ((yb + 1) // 2).long()
                metric_loss = contrastive_loss_fn(features_original, labels)
                # grads_metric = torch.autograd.grad(metric_loss, model.parameters())
                grads_metric = torch.autograd.grad(metric_loss, model.parameters(), allow_unused=True)
                final_grads = [
                    gp + args.lambda_metric * (gm if gm is not None else torch.zeros_like(gp))
                    for gp, gm in zip(grads_sam, grads_metric)
                ]
            else:
                final_grads = grads_sam

            # Step 6: Apply gradients
            optimizer.zero_grad()
            for p, g in zip(model.parameters(), final_grads):
                p.grad = g
            optimizer.step()

            # Logging
            preds_sign = torch.sign(preds)
            acc = (preds_sign == yb).float().mean()
            largest_hessian, hessian_trace = compute_hessian_data(X, y, model)
            features = model.features(X).detach().cpu().numpy()
            effective_rank, explained_variance_ratio = effective_rank_pca(features)
            wandb.log({
                "acc": acc,
                "train_loss": loss.item(),
                "epoch": epoch,
                "steps": steps,
                "largest_hessian": largest_hessian,
                "hessian_trace": hessian_trace,
                "effective_rank": effective_rank,
                "explained_variance_ratio": explained_variance_ratio
            })

            steps += 1

    return model

def experiment_coherence(args):
    set_seed(args.seed)
    x, y = generate_data(args.n, args.d)
    X_val, y_val = generate_data(args.n, args.d)

    model = TwoLayerReLU(args.d, args.hidden).to(x[0].device)

    if args.mode == "SGD":
        coherence_sgd(model, x, y, X_val, y_val, args.lr, args.batch_size, args.epochs)
    elif args.mode == "SAM":
        coherence_sam(model, x, y, X_val, y_val, args.rho, args.lr, args.batch_size, args.epochs)

if args.experiment == "purturb":
    experiment_purturb(args)
elif args.experiment == "coherence":
    experiment_coherence(args)