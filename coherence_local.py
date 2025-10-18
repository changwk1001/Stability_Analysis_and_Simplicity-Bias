import numpy as np
import argparse
import wandb
import numpy as np
from numpy.linalg import eigvalsh  # symmetric eigvals

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.5, help="Learning rate")
parser.add_argument("--steps", type=int, default=1000, help="Max steps")
parser.add_argument("--repetitions", type=int, default=10, help="Repetitions per experiment")
parser.add_argument("--mode", type=str, choices=["SGD", "SAM", "random"], default="SGD", help="Optimization method")
parser.add_argument("--rho", type=float, default=0.5, help="SAM rho value")
parser.add_argument("--alpha", type=float, default=0.1, help="SAM alpha value")
args = parser.parse_args()
# wandb setup
wandb.init(project="Stability-analysis", config=vars(args), name=f"mode_{args.mode}_rho_{args.rho}_alpha_{args.alpha}")
config = wandb.config

# Number of data
n = 100

def datagenerate(i, sigma, d):
    vec = np.zeros((d, 1))
    if i < sigma:
        vec[0, 0] = 1
    else:
        vec[i - sigma + 1, 0] = 1
    return 2 * n / sigma * (vec @ vec.T)

def datagenerateUnlearning(i, sigma, d):
    vec = np.zeros((d, 1))
    if i < sigma:
        vec[0, 0] = 1
    else:
        vec[i - sigma + 1, 0] = 1
    return 2 * n / sigma * (vec @ vec.T)

def losscalculate(H, w):
    return float(w.T @ H @ w)

def randompick(n, batchsize):
    return np.random.choice(n, batchsize, replace=False)

def onebatch(n, datalist, batchsize):
    index = randompick(n, batchsize)
    return np.sum(datalist[index], axis=0) / batchsize

def randomweight(d):
    return np.random.rand(d, 1)

def unlearning_process(retainlist, forgetlist, batchsize, learning_rate, steps, d, alpha):
    w = randomweight(d)
    init_norm = w.T @ w
    for step in range(steps):
        norm = w.T @ w
        if norm >= 1000 * init_norm:
            return False, step
        w = (np.identity(d) - learning_rate * (1-alpha) * onebatch(n, retainlist, batchsize) + learning_rate * alpha * onebatch(n, forgetlist, batchsize)) @ w
    return True, float((w.T @ w).reshape(-1))

def sgd_process(datalist, batchsize, learning_rate, steps, d):
    w = randomweight(d)
    init_norm = w.T @ w
    for step in range(steps):
        norm = w.T @ w
        if norm >= 1000 * init_norm:
            return False, step
        w = (np.identity(d) - learning_rate * onebatch(n, datalist, batchsize)) @ w
    return True, float((w.T @ w).reshape(-1))

def random_process(datalist, batchsize, learning_rate, steps, d):
    w = randomweight(d)
    init_norm = w.T @ w
    for step in range(steps):
        norm = w.T @ w
        if norm >= 1000 * init_norm:
            return False, step
        noise = np.random.randn(d, 1)  # zero mean, std 1 noise
        w = (np.identity(d) - learning_rate * onebatch(n, datalist, batchsize)) @ (w + noise.reshape(w.shape))
    return True, float((w.T @ w).reshape(-1))

def SAM_process(datalist, batchsize, learning_rate, steps, d, rho, alpha):
    w = randomweight(d)
    init_norm = w.T @ w
    H = np.sum(datalist, axis=0) / n
    for step in range(steps):
        norm = w.T @ w
        if norm >= 1000 * init_norm:
            return False, step
        temp = onebatch(n, datalist, batchsize)
        w = (np.identity(d) - learning_rate * temp - learning_rate * rho / alpha * temp @ H) @ w
    return True, float((w.T @ w).reshape(-1))

def top_eig_S_exact(Hr, Hf, a, b):
    """
    Hr: (m, d, d) array of PSDs H_r
    Hf: (m, d, d) array of PSDs H_f
    returns: lambda_max(S) exactly
    """
    Hr = np.asarray(Hr); Hf = np.asarray(Hf)
    m, d, _ = Hr.shape
    N = m*m
    p = d*d

    # Flatten H_r, H_f for Frobenius inner products
    Hr_flat = Hr.reshape(m, p)         # shape (m, d^2)
    Hf_flat = Hf.reshape(m, p)         # shape (m, d^2)

    # Build all A_{(r,f)} in flattened form: V[i,:] = vec(A_i)
    # Use broadcasting: (m,1,p) + (1,m,p) -> (m,m,p)
    V = (a*Hr_flat[:, None, :] + b*Hf_flat[None, :, :]).reshape(N, p)  # (N, p)

    # G_{ij} = <A_i, A_j>_F = vec(A_i)·vec(A_j) = V V^T
    # This is exact and symmetric PSD
    G = V @ V.T                       # (N, N)

    # S = sqrt(G) elementwise (since S_ij = sqrt(tr(A_i A_j)))
    S = np.sqrt(G, dtype=G.dtype, where=np.ones_like(G, dtype=bool))

    # EXACT top eigenvalue of S (dense): O(N^3); for large N use eigsh(k=1)
    # return float(eigvalsh(S, subset_by_index=[N-1, N-1])[0])  # if your NumPy supports it
    vals = eigvalsh(S)                # full spectrum
    return float(vals[-1])

batchsizes = [i+1 for i in range(49)]
sigmas = [i+1 for i in range(50)]
logging = []
unique = 0
alpha = 0.1
test_table = wandb.Table(columns=["batch_size", "sigma", "success", "diverge step", "norm"])

for sigma in sigmas:
    d = n - sigma + 1
    datalist = np.array([datagenerate(i, sigma, d) for i in range(n)])
    retainlist = datalist = np.array([datagenerate(i, sigma, d) for i in range(n)])
    forgetlist = datalist = np.array([datagenerate(i, sigma, d) for i in range(n)])

    for b in batchsizes:
        success = 0
        unique += 1
        norm = 0
        diverge_step = 0
        for _ in range(config.repetitions):
            if config.mode == "SGD":
                s, state = sgd_process(datalist, b, config.lr, config.steps, d)
                success += s
                if s == True:
                    norm += state
                else:
                    diverge_step += state
            elif config.mode == "random":
                s, state = random_process(datalist, b, config.lr, config.steps, d)
                success += s
                if s == True:
                    norm += state
                else:
                    diverge_step += state
            else:
                s, state = SAM_process(datalist, b, config.lr, config.steps, d, config.rho, config.alpha)
                if s == True:
                    norm += state
                else:
                    diverge_step += state
        result = 1 if success >= (config.repetitions / 2) else 0
        if result:
            norm /= success
        else:
            diverge_step /= (config.repetitions - success)
        
        logging.append((b,sigma,result))
        if result == 1:
            test_table.add_data(b, sigma, result, 0, norm)
        else:
            test_table.add_data(b, sigma, result, diverge_step, 0)
wandb.log({
            "mode":config.mode,
            "rho": config.rho,
            "alpha": config.alpha,
            "lr": config.lr,
            "n":n,
            "table":test_table
        })