import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch_optimizer
import wandb
from sklearn.decomposition import PCA
import numpy as np
from torch.autograd import grad
from numpy.linalg import eigh
from collections import defaultdict
from bypass_bn import enable_running_stats, disable_running_stats
from sam import SAM

def collect_100_indices(dataset, num_classes=10, samples_per_class=100):
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        if len(class_indices[label]) < samples_per_class:
            class_indices[label].append(idx)
        if all(len(v) == samples_per_class for v in class_indices.values()):
            break
    all_indices = []
    for c in range(num_classes):
        all_indices.extend(class_indices[c])
    return all_indices  # total 100

def compute_streamed_gradient_matrix(model, dataset, fixed_indices, device):
    """
    Computes a 100x100 dot product matrix of per-sample gradients,
    but avoids loading all gradients into memory simultaneously.
    Returns the largest eigenvalue of the normalized matrix.
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    gradients = []
    max_norm_sq = 0

    # Compute and store gradients one-by-one (on CPU)
    for idx in fixed_indices:
        x, y = dataset[idx]
        x = x.unsqueeze(0).to(device)
        y = torch.tensor([y]).to(device)

        model.zero_grad()
        output = model(x)
        loss = criterion(output, y)

        g = grad(loss, model.parameters(), retain_graph=False, create_graph=False)
        g_vec = torch.cat([p.detach().cpu().reshape(-1) for p in g])
        max_norm_sq = max(max_norm_sq, g_vec.norm().item() ** 2)
        gradients.append(g_vec)

    n = len(gradients)
    dot_matrix = torch.zeros((n, n))

    # Compute dot products pairwise
    for i in range(n):
        for j in range(i, n):
            dot_ij = torch.dot(gradients[i], gradients[j])
            dot_matrix[i, j] = dot_ij
            dot_matrix[j, i] = dot_ij  # symmetry

    # Convert to NumPy and compute largest eigenvalue
    eigvals = eigh(dot_matrix.numpy())[0] / (max_norm_sq + 1e-8)
    return eigvals[-1], dot_matrix.numpy()

def pca_effective_rank(features, threshold=0.9):
    pca = PCA()
    pca.fit(features)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    r = np.searchsorted(cumulative, threshold) + 1
    return r, cumulative

class ResNetWithFeatures(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # all but fc
        self.classifier = base_model.fc

    def forward(self, x, return_features=False):
        x = self.features(x)   # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        if return_features:
            return self.classifier(x), x
        return self.classifier(x)

def extract_features(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            _, feats = model(x, return_features=True)
            features.append(feats.cpu())
    return torch.cat(features, dim=0).numpy()  # [N, D]


def get_args():
    parser = argparse.ArgumentParser(description="Train ResNet on CIFAR-10 with optional SAM")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rho", type=float, default=0.05, help="SAM rho value (sharpness radius)")
    parser.add_argument("--optimizer", default="sgd", choices=['sgd', 'sam'], help="optimization method")
    return parser.parse_args()

def get_dataloaders(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)
    return (
        DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4),
        DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4),
        trainset
    )

def evaluate(model, dataloader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def main():

    args = get_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    name = "larger scale coherence"
    wandb.init(project=name, config=args, name=args.optimizer + str(args.rho))

    train_loader, test_loader, train_dataset = get_dataloaders(args.batch_size)

    model = ResNetWithFeatures(models.resnet18(num_classes=10).to(device))
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'sam':
        base_optimizer = optim.SGD
        optimizer = SAM(
            model.parameters(), base_optimizer, rho=args.rho, lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay
        )
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)

            if args.optimizer == 'sam':
                enable_running_stats(model)
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                disable_running_stats(model)
                criterion(model(x), y).backward(create_graph=False)
                optimizer.second_step(zero_grad=True)

            else:
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

        acc = evaluate(model, test_loader, device)
        features = extract_features(model, train_loader, device)

        rank_999, _ = pca_effective_rank(features, threshold=0.999)
        rank_99, _ = pca_effective_rank(features, threshold=0.99)
        rank_95, _ = pca_effective_rank(features, threshold=0.95)
        rank_90, _ = pca_effective_rank(features, threshold=0.90)

        print(f"Epoch {epoch+1}: Loss={total_loss:.2f}, Test Accuracy={acc*100:.2f}%")

        fixed_indices = collect_100_indices(train_dataset, 10, 100)  # get 100 per class
        eigval, grad_matrix = compute_streamed_gradient_matrix(model, train_dataset, fixed_indices, device)

        wandb.log({
            "epoch": epoch,
            "train_loss": total_loss,
            "train_acc": acc,
            "val_acc": acc,
            "feature rank": rank_90,
            "rank_999":rank_999,
            "rank_95":rank_95,
            "rank_99":rank_99,
            "grad_matrix/largest_eigval": eigval
        })
        scheduler.step()
    
if __name__ == "__main__":
    main()
