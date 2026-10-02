"""Household Animals (Cats vs Dogs) classification - PyTorch.
Stages: (1) VGG-style CNN from scratch, (2) VGG-16 transfer learning (frozen),
(3) VGG-16 fine-tuning (block5 unfrozen). Plus activation maps and Grad-CAM.
Run:  python train.py            (add --subset 4000 for a quick test)
"""
import argparse, copy, json, os, random, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

CLASSES = ["cat", "dog"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HFImages(Dataset):
    def __init__(self, ds, idx, tf): self.ds, self.idx, self.tf = ds, idx, tf
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        r = self.ds[int(self.idx[i])]
        return self.tf(r["image"].convert("RGB")), float(r["labels"])


def make_loaders(ds, split, size, bs):
    aug = T.Compose([T.Resize((size, size)), T.RandomHorizontalFlip(), T.RandomRotation(15),
                     T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])
    plain = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])
    mk = lambda k, tf, sh: DataLoader(HFImages(ds, split[k], tf), batch_size=bs, shuffle=sh, num_workers=2)
    return {"train": mk("train", aug, True), "val": mk("val", plain, False), "test": mk("test", plain, False)}


class SimpleVGG(nn.Module):
    """VGG-style baseline: 4 blocks (32,64,128,128), 3x3 convs + maxpool."""
    def __init__(self):
        super().__init__()
        layers, c = [], 3
        for f in [32, 64, 128, 128]:
            layers += [nn.Conv2d(c, f, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)]; c = f
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(self, x): return self.classifier(self.features(x))


def vgg16_transfer():
    m = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    for p in m.features.parameters(): p.requires_grad = False
    m.classifier = nn.Sequential(nn.Linear(25088, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))
    return m


def run_epoch(model, loader, opt=None, scaler=None):
    train = opt is not None
    model.train(train)
    tot_loss = correct = n = 0
    preds, labels = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev).float()
            with torch.autocast(device_type=dev.type, enabled=dev.type == "cuda"):
                out = model(x).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(out, y)
            if train:
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            p = (out > 0).float()
            tot_loss += loss.item() * len(y); correct += (p == y).sum().item(); n += len(y)
            preds += p.cpu().tolist(); labels += y.cpu().tolist()
    return tot_loss / n, correct / n, preds, labels


def fit(model, loaders, epochs, lr, name):
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=dev.type == "cuda")
    hist = {"loss": [], "val_loss": [], "acc": [], "val_acc": []}
    best, best_state, t0 = 0, None, time.time()
    for e in range(epochs):
        l, a, _, _ = run_epoch(model, loaders["train"], opt, scaler)
        vl, va, _, _ = run_epoch(model, loaders["val"])
        for k, v in zip(hist, [l, vl, a, va]): hist[k].append(v)
        print(f"[{name}] epoch {e+1}/{epochs} loss {l:.4f} acc {a:.4f} | val_loss {vl:.4f} val_acc {va:.4f}", flush=True)
        if va > best: best, best_state = va, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    tl, ta, p, y = run_epoch(model, loaders["test"])
    res = {"trainable_params": sum(p_.numel() for p_ in params), "epochs": epochs, "best_val_acc": best,
           "test_acc": ta, "test_loss": tl, "test_f1": f1_score(y, p), "train_time_s": round(time.time() - t0),
           "cm": confusion_matrix(y, p).tolist()}
    print(f"[{name}] TEST acc {ta:.4f} f1 {res['test_f1']:.4f}", flush=True)
    return res, hist


def plot_curves(hists, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, h in hists.items():
        ax[0].plot(h["acc"], "--", label=f"{name} train"); ax[0].plot(h["val_acc"], label=f"{name} val")
        ax[1].plot(h["loss"], "--", label=f"{name} train"); ax[1].plot(h["val_loss"], label=f"{name} val")
    ax[0].set_title("Accuracy"); ax[1].set_title("Loss")
    for a in ax: a.set_xlabel("epoch"); a.legend(fontsize=6)
    plt.tight_layout(); plt.savefig(f"{out}/curves.png", dpi=150); plt.close()


def plot_cm(cm, out):
    plt.figure(figsize=(3.2, 3)); plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2): plt.text(j, i, cm[i][j], ha="center", va="center")
    plt.xticks([0, 1], CLASSES); plt.yticks([0, 1], CLASSES); plt.xlabel("predicted"); plt.ylabel("true")
    plt.title("VGG-16 fine-tuned (test)"); plt.tight_layout(); plt.savefig(f"{out}/confusion_matrix.png", dpi=150); plt.close()


def denorm(x):
    return (x.cpu() * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]).clamp(0, 1).permute(1, 2, 0).numpy()


def plot_activations(model, ds, split, out):
    """Feature maps of the first 3 conv blocks of the scratch VGG (one test image)."""
    tf = T.Compose([T.Resize((128, 128)), T.ToTensor(), T.Normalize(MEAN, STD)])
    img = ds[int(split["test"][0])]["image"].convert("RGB")
    x = tf(img).unsqueeze(0).to(dev); model.eval()
    fig, ax = plt.subplots(3, 9, figsize=(11, 4.2))
    with torch.no_grad():
        for i, layer in enumerate(model.features):
            x = layer(x)
            if i in (1, 4, 7):
                r = [1, 4, 7].index(i)
                ax[r, 0].imshow(img.resize((128, 128))); ax[r, 0].set_title("input", fontsize=7)
                for c in range(8): ax[r, c + 1].imshow(x[0, c].cpu(), cmap="viridis")
                ax[r, 1].set_title(f"conv block {r+1}", fontsize=7, loc="left")
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/activations.png", dpi=150); plt.close()


def gradcam(model, x):
    model.eval(); x = x.clone().requires_grad_(True)
    a = model.features(x); a.retain_grad()
    out = model.classifier(torch.flatten(model.avgpool(a), 1)).squeeze()
    sign = 1.0 if out.item() > 0 else -1.0
    (sign * out).backward()
    w = a.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    return (cam / (cam.max() + 1e-8)).detach().cpu().numpy(), int(out.item() > 0)


def plot_gradcam(model, ds, split, out, k=6):
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])
    fig, ax = plt.subplots(2, k, figsize=(2 * k, 4.3))
    for j in range(k):
        r = ds[int(split["test"][j])]; x = tf(r["image"].convert("RGB")).unsqueeze(0).to(dev)
        cam, pred = gradcam(model, x)
        ax[0, j].imshow(denorm(x[0])); ax[0, j].set_title(f"true {CLASSES[r['labels']]} / pred {CLASSES[pred]}", fontsize=7)
        ax[1, j].imshow(denorm(x[0])); ax[1, j].imshow(cam, cmap="jet", alpha=0.45)
    for a in ax.ravel(): a.axis("off")
    plt.tight_layout(); plt.savefig(f"{out}/gradcam.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs"); ap.add_argument("--subset", type=int, default=0)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs_scratch", type=int, default=10); ap.add_argument("--epochs_head", type=int, default=5)
    ap.add_argument("--epochs_ft", type=int, default=5)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)

    from datasets import load_dataset  # Kaggle "Dogs vs Cats" mirror on HF Hub (no API key needed)
    ds = load_dataset("microsoft/cats_vs_dogs", split="train")
    idx = np.random.permutation(len(ds))
    if a.subset: idx = idx[:a.subset]
    n = len(idx); nt, nv = int(.8 * n), int(.1 * n)
    split = {"train": idx[:nt], "val": idx[nt:nt + nv], "test": idx[nt + nv:]}
    print({k: len(v) for k, v in split.items()}, "device:", dev)
    results = {"device": str(dev), "dataset": {"name": "Dogs vs Cats (Kaggle)", "total": n, **{k: len(v) for k, v in split.items()}}, "models": {}}
    hists = {}

    L128 = make_loaders(ds, split, 128, a.bs)
    base = SimpleVGG()
    results["models"]["VGG-style CNN (scratch)"], hists["scratch"] = fit(base, L128, a.epochs_scratch, 1e-3, "scratch")
    plot_activations(base, ds, split, a.out)

    L224 = make_loaders(ds, split, 224, a.bs)
    vgg = vgg16_transfer()
    results["models"]["VGG-16 transfer (frozen)"], hists["vgg16 frozen"] = fit(vgg, L224, a.epochs_head, 1e-3, "vgg16-frozen")
    for p in vgg.features[24:].parameters(): p.requires_grad = True  # unfreeze block5
    results["models"]["VGG-16 fine-tuned (block5)"], hists["vgg16 finetune"] = fit(vgg, L224, a.epochs_ft, 1e-5, "vgg16-finetune")

    plot_curves(hists, a.out)
    plot_cm(results["models"]["VGG-16 fine-tuned (block5)"]["cm"], a.out)
    plot_gradcam(vgg, ds, split, a.out)
    torch.save(vgg.state_dict(), f"{a.out}/vgg16_finetuned.pth")
    json.dump(results, open(f"{a.out}/results.json", "w"), indent=2)
    print("Done. Results in", a.out)


if __name__ == "__main__":
    main()
