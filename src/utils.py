import torch
import torchvision.transforms
from torch.utils.data import DataLoader
from dataset import RoadDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(image_height: int, image_width: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )


def get_valid_transform(image_height: int, image_width: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_height, width=image_width),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )


def save_samples_predicted(
    loader: DataLoader,
    model: torch.nn.Module,
    folder: str = "../saved_images",
    device: str = "mps",
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(
            y.unsqueeze(1).float(), f"{folder}/truth_{idx}.png"
        )


def save_checkpoint(state: dict, filename: str = "my_checkpoint.pth.tar") -> None:
    print("=> Saving checkpoint")
    torch.save(state, filename)


def get_loaders(
    train_transform: A.Compose, valid_transform: A.Compose
) -> tuple[DataLoader, DataLoader]:
    trn_ds = RoadDataset("train", train_transform)
    val_ds = RoadDataset("valid", valid_transform)
    return DataLoader(trn_ds, batch_size=16, shuffle=True), DataLoader(
        val_ds, batch_size=16, shuffle=False
    )


def compute_print_val_metrics(
    loader: DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: str = "mps",
) -> tuple[float, float, float]:
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = model(x)
            activated_preds = torch.sigmoid(preds)
            activated_preds = (activated_preds > 0.5).float()
            num_correct += (activated_preds == y).sum()
            num_pixels += torch.numel(activated_preds)
            dice_score += (2 * (activated_preds * y).sum()) / ((activated_preds + y).sum() + 1e-8)
            total_loss += loss_fn(preds, y.float()).item()
    model.train()
    global_accuracy = num_correct / num_pixels * 100
    global_dice_score = dice_score / len(loader)
    total_loss /= len(loader)
    print(f"Got {num_correct}/{num_pixels} with acc {global_accuracy:.2f}")
    print(f"Dice Score: {global_dice_score:.2f}")

    return total_loss, global_accuracy, global_dice_score
