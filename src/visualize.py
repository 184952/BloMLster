"""Evaluering og figurer for klassifisereren. Figurene lagres i reports/figures/.

    from src.visualize import get_predictions, plot_confusion, ...
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .data import MEAN, STD
from .utils import save_fig


def denorm(img, mean=MEAN, std=STD):
    """Angre Normalize og gjør [3, H, W] om til [H, W, 3] for plt.imshow."""
    img = img.detach().cpu()
    img = img * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    return img.permute(1, 2, 0).clamp(0, 1)


@torch.no_grad()
def get_predictions(model, dataloader, device):
    """Kjør modellen på hele dataloaderen. Returnerer (y_true, y_pred, y_prob) på CPU."""
    model.eval()
    ys, probs = [], []
    for X, y in dataloader:
        probs.append(model(X.to(device)).softmax(1).cpu())
        ys.append(y)
    y_prob = torch.cat(probs)
    return torch.cat(ys), y_prob.argmax(1), y_prob


def summary_metrics(y_true, y_pred, y_prob):
    """Metrikkene fra rapporten: accuracy, macro-F1, top-3 og top-5."""
    topk = y_prob.topk(5, dim=1).indices
    return {
        "accuracy": (y_pred == y_true).float().mean().item(),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "top3": (topk[:, :3] == y_true[:, None]).any(1).float().mean().item(),
        "top5": (topk == y_true[:, None]).any(1).float().mean().item(),
    }


def print_report(y_true, y_pred, classes):
    """Precision, recall og F1 per klasse."""
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))


def plot_class_distribution(labels_csv, classes, name="klassefordeling"):
    """Antall bilder per klasse, fordelt på train/val/test."""
    df = pd.read_csv(labels_csv)
    counts = df.groupby(["label", "split"]).size().unstack()[["train", "val", "test"]]
    counts.index = [classes[i] for i in counts.index]
    counts = counts.loc[counts.sum(1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, 7))
    counts.plot.barh(stacked=True, ax=ax, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set(xlabel="Antall bilder", ylabel="", title="Bilder per art")
    save_fig(fig, name)
    return fig


def plot_history(history, name="treningskurver"):
    """Loss og accuracy per epoke for trening og validering."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set(title="Loss", xlabel="Epoke")
    ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set(title="Accuracy", xlabel="Epoke")
    ax2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax2.legend()
    save_fig(fig, name)
    return fig


def plot_confusion(y_true, y_pred, classes, name="confusion_matrix"):
    """Normalisert confusion matrix: hver rad summerer til 1 (= recall på diagonalen)."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap="Blues", vmin=0, vmax=1,
                annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
    ax.set(xlabel="Predikert", ylabel="Sann", title="Confusion matrix (andel av sann klasse)")
    save_fig(fig, name)
    return fig


def plot_per_class_accuracy(y_true, y_pred, classes, name="accuracy_per_klasse"):
    """Recall per klasse som sortert søylediagram."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    acc = pd.Series(cm.diagonal(), index=classes).sort_values()
    fig, ax = plt.subplots(figsize=(8, 6))
    acc.plot.barh(ax=ax, color="#4C72B0")
    ax.axvline((np.asarray(y_true) == np.asarray(y_pred)).mean(), color="black", ls="--", label="total accuracy")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    ax.set(xlabel="Andel riktig", title="Accuracy per art")
    ax.legend()
    save_fig(fig, name)
    return fig


def top_confusions(y_true, y_pred, classes, n=10):
    """De n vanligste forvekslingene som tabell."""
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)
    pairs = [(classes[i], classes[j], cm[i, j]) for i, j in zip(*np.unravel_index(np.argsort(cm, None)[::-1][:n], cm.shape))]
    return pd.DataFrame(pairs, columns=["sann", "predikert", "antall"])


def plot_wrong(dataset, y_true, y_pred, y_prob, classes, n=16, name="feilklassifisert"):
    """De mest selvsikre feilene: modellen var sikker, men tok feil."""
    wrong = (y_pred != y_true).nonzero().flatten()
    wrong = wrong[y_prob[wrong].max(1).values.argsort(descending=True)][:n]
    cols = 4
    fig, axes = plt.subplots(int(np.ceil(len(wrong) / cols)), cols, figsize=(12, 3.2 * np.ceil(len(wrong) / cols)))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong.tolist()):
        img, _ = dataset[i]
        ax.imshow(denorm(img))
        ax.set_title(f"sann: {classes[y_true[i]]}\npred: {classes[y_pred[i]]} ({y_prob[i].max():.0%})", fontsize=9, color="firebrick")
    fig.suptitle("Mest selvsikre feilklassifiseringer")
    fig.tight_layout()
    save_fig(fig, name)
    return fig


def grad_cam(model, img, device, target=None):
    """Grad-CAM på siste konvolusjonsblokk. Returnerer (heatmap [H, W] i 0–1, predikert klasse)."""
    model.eval()
    acts, grads = {}, {}
    layer = model.features[-1]
    h1 = layer.register_forward_hook(lambda m, i, o: acts.update(a=o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.update(g=go[0]))
    try:
        out = model(img.unsqueeze(0).to(device))
        pred = out.argmax().item()
        model.zero_grad()
        out[0, pred if target is None else target].backward()
    finally:
        h1.remove()
        h2.remove()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=img.shape[1:], mode="bilinear")[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().cpu(), pred


def plot_grad_cam(model, dataset, classes, device, indices, name="grad_cam"):
    """Originalbilde (øverst) og Grad-CAM (nederst) for gitte indekser i datasettet."""
    fig, axes = plt.subplots(2, len(indices), figsize=(3.5 * len(indices), 7))
    for col, i in enumerate(indices):
        img, label = dataset[i]
        cam, pred = grad_cam(model, img, device)
        axes[0, col].imshow(denorm(img))
        axes[0, col].set_title(f"sann: {classes[label]}", fontsize=9)
        axes[1, col].imshow(denorm(img))
        axes[1, col].imshow(cam, cmap="jet", alpha=0.45)
        axes[1, col].set_title(f"pred: {classes[pred]}", fontsize=9, color="green" if pred == label else "firebrick")
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    save_fig(fig, name)
    return fig


def plot_calibration(y_true, y_prob, bins=10, name="kalibrering"):
    """Reliability diagram: stemmer modellens konfidens med hvor ofte den har rett?"""
    conf, pred = y_prob.max(1)
    correct = (pred == y_true).float()
    edges = torch.linspace(0, 1, bins + 1)
    idx = torch.bucketize(conf, edges[1:-1])
    acc = [correct[idx == b].mean().item() if (idx == b).any() else np.nan for b in range(bins)]
    cnt = [(idx == b).sum().item() for b in range(bins)]
    ece = sum(c / len(conf) * abs(a - conf[idx == b].mean().item()) for b, (a, c) in enumerate(zip(acc, cnt)) if c)
    centers = (edges[:-1] + edges[1:]) / 2
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.bar(centers, acc, width=1 / bins, edgecolor="black", alpha=0.8, label="faktisk accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="perfekt kalibrert")
    ax.set(xlabel="Konfidens", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1), title=f"Kalibrering (ECE = {ece:.3f})")
    ax.legend()
    save_fig(fig, name)
    return fig


@torch.no_grad()
def plot_tsne(model, dataloader, classes, device, name="tsne"):
    """t-SNE av de 256 featurene før siste lag: danner klassene egne grupper?"""
    model.eval()
    feats, ys = [], []
    for X, y in dataloader:
        feats.append(model.features(X.to(device)).mean((2, 3)).cpu())
        ys.append(y)
    emb = TSNE(n_components=2, init="pca", random_state=42).fit_transform(torch.cat(feats).numpy())
    ys = torch.cat(ys).numpy()
    fig, ax = plt.subplots(figsize=(10, 8))
    palette = sns.color_palette("tab20", len(classes))
    for c, name_c in enumerate(classes):
        m = ys == c
        ax.scatter(emb[m, 0], emb[m, 1], s=5, color=palette[c], label=name_c)
    ax.legend(markerscale=3, fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.set(title="t-SNE av modellens features (testsett)", xticks=[], yticks=[])
    fig.tight_layout()
    save_fig(fig, name)
    return fig
