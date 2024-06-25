import torch
import numpy as np
from omegaconf import OmegaConf

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


def get_config(config_path):
    base_conf = OmegaConf.load(config_path)
    overrides = OmegaConf.from_cli()
    updated_conf = OmegaConf.merge(base_conf, overrides)
    return OmegaConf.to_container(updated_conf)


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# write save_checkpoint and load_checkpoint functions here
def save_checkpoint(model, optimizer, save_dir, epoch):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, f"{save_dir}/model_{epoch}.pth")


def load_checkpoint(model, optimizer, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch


def calculate_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, sensitivity, specificity, f1, accuracy