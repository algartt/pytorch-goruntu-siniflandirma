"""PyTorch Fashion-MNIST eğitim komut satırı uygulaması."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import FashionCNN


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_loaders(data_dir: Path, batch_size: int, quick: bool) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_data = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    if quick:
        train_data = Subset(train_data, range(min(2048, len(train_data))))
        test_data = Subset(test_data, range(min(512, len(test_data))))
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0),
        DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0),
    )


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total_items = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_items += labels.size(0)

    return total_loss / total_items, total_correct / total_items


def main() -> None:
    parser = argparse.ArgumentParser(description="Fashion-MNIST üzerinde PyTorch CNN eğitimi")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true", help="Küçük veri alt kümesiyle hızlı kontrol")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/fashion_cnn.pt"))
    args = parser.parse_args()

    seed_everything(args.seed)
    device = choose_device()
    train_loader, test_loader = build_loaders(args.data_dir, args.batch_size, args.quick)
    model = FashionCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    print(f"Cihaz: {device} | Eğitim örneği: {len(train_loader.dataset)}")

    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, device, optimizer)
        test_loss, test_accuracy = run_epoch(model, test_loader, loss_fn, device)
        print(
            f"Epoch {epoch:02d} | eğitim kaybı={train_loss:.4f} doğruluk={train_accuracy:.2%} | "
            f"test kaybı={test_loss:.4f} doğruluk={test_accuracy:.2%}"
        )
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state": model.state_dict(), "accuracy": test_accuracy, "classes": train_loader.dataset.dataset.classes if isinstance(train_loader.dataset, Subset) else train_loader.dataset.classes}, args.output)

    print(f"En iyi doğruluk: {best_accuracy:.2%} | Model: {args.output}")


if __name__ == "__main__":
    main()
