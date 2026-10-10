"""
Generate the demo images used in the evaluation scenarios.

Run once:
    python tools/make_demo_images.py

Produces data/demo_images/:
    loss_curve.png      — a clean ML training curve (scenario 1 & 3)
    neural_net.png      — a labelled neural-network diagram (scenario 3)
    blurred_scene.png   — a deliberately blurred photo-like image (scenario 4)
    handwritten_math.png— simulated handwriting for OCR (scenario 2)
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent.parent / "data" / "demo_images"
OUT.mkdir(parents=True, exist_ok=True)


def _font(size: int):
    """Best-effort TrueType font, else PIL's bitmap default."""
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def loss_curve() -> None:
    """Training vs validation loss — the classic overfitting picture."""
    w, h = 960, 640
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    left, top, right, bottom = 130, 70, w - 60, h - 90

    # Axes
    d.line([(left, top), (left, bottom)], fill="black", width=3)
    d.line([(left, bottom), (right, bottom)], fill="black", width=3)

    f = _font(24)
    fs = _font(19)
    d.text((left, top - 46), "Model Loss over Training", fill="black", font=f)
    d.text(((left + right) // 2 - 40, bottom + 30), "Epoch", fill="black", font=fs)
    # Y-axis label rotated so it never collides with the tick numbers.
    label = Image.new("RGBA", (70, 30), (255, 255, 255, 0))
    ImageDraw.Draw(label).text((0, 0), "Loss", fill="black", font=fs)
    img.paste(label.rotate(90, expand=True), (12, (top + bottom) // 2 - 20), label.rotate(90, expand=True))

    # Gridlines + y ticks
    for i in range(6):
        value = i * 0.2
        y = bottom - int((value / 1.2) * (bottom - top))
        d.line([(left, y), (right, y)], fill=(220, 220, 220), width=1)
        d.text((left - 58, y - 10), f"{value:.1f}", fill="black", font=fs)

    # x ticks
    for epoch in range(0, 21, 5):
        x = left + int((epoch / 20) * (right - left))
        d.line([(x, bottom), (x, bottom + 8)], fill="black", width=2)
        d.text((x - 12, bottom + 12), str(epoch), fill="black", font=fs)

    def to_xy(epoch, loss):
        x = left + (epoch / 20) * (right - left)
        y = bottom - (loss / 1.2) * (bottom - top)
        return x, y

    # Training loss: keeps falling (0.90 → 0.12)
    train = [(e, 0.90 * math.exp(-0.16 * e) + 0.04) for e in range(21)]
    # Validation loss: falls to ~0.31 at epoch 6, then rises (overfitting)
    val = []
    for e in range(21):
        if e <= 6:
            v = 0.95 * math.exp(-0.19 * e) + 0.06
        else:
            v = 0.31 + 0.031 * (e - 6) ** 1.35
        val.append((e, v))

    d.line([to_xy(*p) for p in train], fill=(31, 90, 200), width=4)
    d.line([to_xy(*p) for p in val], fill=(214, 91, 30), width=4)

    # Mark the divergence point
    dx, dy = to_xy(6, val[6][1])
    d.ellipse([dx - 9, dy - 9, dx + 9, dy + 9], outline=(214, 91, 30), width=4)
    # Annotation placed below-right of the marker so it clears both curves.
    d.text((dx + 20, dy + 18), "validation minimum at epoch 6",
           fill=(150, 60, 20), font=fs)

    # Legend
    lx, ly = right - 240, top + 20
    d.line([(lx, ly), (lx + 40, ly)], fill=(31, 90, 200), width=4)
    d.text((lx + 50, ly - 11), "Training loss", fill="black", font=fs)
    d.line([(lx, ly + 34), (lx + 40, ly + 34)], fill=(214, 91, 30), width=4)
    d.text((lx + 50, ly + 23), "Validation loss", fill="black", font=fs)

    img.save(OUT / "loss_curve.png")
    print("wrote", OUT / "loss_curve.png")


def neural_net() -> None:
    """Labelled diagram: input / hidden (middle) / output layers."""
    w, h = 900, 560
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    f = _font(22)

    layers = [
        ("Input layer", 170, 4),
        ("Hidden (middle) layer", 450, 6),
        ("Output layer", 730, 2),
    ]
    ys = {}
    radius = 26
    for name, x, count in layers:
        total = (count - 1) * 90
        start = (h - 80 - total) / 2 + 40
        ys[x] = []
        for i in range(count):
            y = start + i * 90
            ys[x].append(y)

    # Connections first so nodes draw on top
    for i, y1 in enumerate(ys[170]):
        for j, y2 in enumerate(ys[450]):
            d.line([(170, y1), (450, y2)], fill=(200, 205, 215), width=2)
    for i, y1 in enumerate(ys[450]):
        for j, y2 in enumerate(ys[730]):
            d.line([(450, y1), (730, y2)], fill=(200, 205, 215), width=2)

    colors = [(70, 130, 200), (240, 170, 60), (90, 180, 120)]
    for (name, x, _), color in zip(layers, colors):
        for y in ys[x]:
            d.ellipse([x - radius, y - radius, x + radius, y + radius],
                      fill=color, outline="black", width=2)
        d.text((x - 60, 30), name, fill="black", font=f)

    img.save(OUT / "neural_net.png")
    print("wrote", OUT / "neural_net.png")


def blurred_scene() -> None:
    """
    A blurry, partly-obscured 'photo' of people. Deliberately bad so the
    uncertainty check (V4) has something to bite on.
    """
    w, h = 800, 560
    img = Image.new("RGB", (w, h), (120, 140, 160))
    d = ImageDraw.Draw(img)

    # Background bands
    d.rectangle([0, 380, w, h], fill=(96, 112, 96))
    d.rectangle([0, 0, w, 200], fill=(168, 186, 205))

    # Four blurry figure silhouettes; the fourth is half-hidden by a dark box
    positions = [(180, 300), (330, 320), (470, 305)]
    for cx, cy in positions:
        d.ellipse([cx - 42, cy - 150, cx + 42, cy - 66], fill=(212, 180, 156))
        d.rounded_rectangle([cx - 55, cy - 70, cx + 55, cy + 90], radius=30,
                            fill=(70, 84, 120))

    # Obscured fourth figure (partially behind a dark rectangle)
    cx, cy = 620, 310
    d.ellipse([cx - 42, cy - 150, cx + 42, cy - 66], fill=(212, 180, 156))
    d.rounded_rectangle([cx - 55, cy - 70, cx + 55, cy + 90], radius=30,
                        fill=(70, 84, 120))
    d.rectangle([560, 210, 800, 560], fill=(28, 28, 34))   # occluder

    img = img.filter(ImageFilter.GaussianBlur(radius=9))
    img.save(OUT / "blurred_scene.png")
    print("wrote", OUT / "blurred_scene.png")


def handwritten_math() -> None:
    """Simulated handwriting — a word problem for the OCR path."""
    w, h = 900, 500
    img = Image.new("RGB", (w, h), (250, 248, 240))
    d = ImageDraw.Draw(img)
    f = _font(34)
    fl = _font(30)

    lines = [
        "Question 3:",
        "A model trains for 20 epochs.",
        "Training loss falls from 0.90 to 0.12.",
        "Validation loss falls to 0.31 at epoch 6",
        "and then rises to 0.72 by epoch 20.",
        "Why does this happen?",
    ]
    for i, line in enumerate(lines):
        d.text((60, 50 + i * 62), line, fill=(24, 32, 58),
               font=f if i == 0 else fl)

    # A little slant + blur to look hand-written and slightly imperfect
    img = img.rotate(-0.8, resample=Image.BICUBIC, fillcolor=(250, 248, 240))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
    img.save(OUT / "handwritten_math.png")
    print("wrote", OUT / "handwritten_math.png")


if __name__ == "__main__":
    loss_curve()
    neural_net()
    blurred_scene()
    handwritten_math()
    print("\nDemo images are in", OUT)
