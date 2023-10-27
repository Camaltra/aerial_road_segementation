import torch
import torch.nn as nn
from models.u_net import UNet
from tqdm import tqdm
from config import ModelConfig
from utils import (
    compute_print_val_metrics,
    save_samples_predicted,
    save_checkpoint,
    get_loaders,
    get_train_transform,
    get_valid_transform,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def train_fn(
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str,
) -> float:
    loop = tqdm(loader)
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device)

        predictions = model(data)
        loss = loss_fn(predictions, targets)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)


def main():
    cfg = ModelConfig()
    cfg.print_config()

    writer = SummaryWriter()

    train_transform = get_train_transform(cfg.image_height, cfg.image_width)
    valid_transform = get_valid_transform(cfg.image_height, cfg.image_width)

    model = UNet(in_channel=3, out_channel=1).to(cfg.device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    trn_loader, val_loader = get_loaders(train_transform, valid_transform)

    for epoch in range(cfg.num_epoch):
        train_loss = train_fn(trn_loader, model, optimizer, loss_fn, device=cfg.device)

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)
        valid_loss, valid_accuracy, valid_dice_score = compute_print_val_metrics(
            val_loader, model, loss_fn=loss_fn, device=cfg.device
        )
        save_samples_predicted(val_loader, model, device=cfg.device)

        writer.add_scalars(
            "Loss", {"train_loss": train_loss, "valid_loss": valid_loss}, epoch
        )
        writer.add_scalars("Accuracy", {"valid_accuracy": valid_accuracy}, epoch)
        writer.add_scalars("Dice Score", {"valid_dice_score": valid_dice_score}, epoch)

    torch.save(model, "road_segmentation.pt")


if __name__ == "__main__":
    main()
