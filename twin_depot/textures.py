"""Procedurally painted textures (bilingual signs, parcel labels) rendered with Pillow."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import layout

_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)


def _find_font() -> tuple[str | None, bool]:
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            return path, "DejaVu" not in path
    return None, False


FONT_PATH, HAS_CJK = _find_font()


def _font(size: int):
    if FONT_PATH is None:
        return ImageFont.load_default()
    return ImageFont.truetype(FONT_PATH, size)


def _rgb(rgba, scale: float = 1.0) -> tuple[int, int, int]:
    r, g, b = (int(max(0, min(255, c * 255 * scale))) for c in rgba[:3])
    return r, g, b


def _center_text(draw: ImageDraw.ImageDraw, box, text: str, font, fill) -> None:
    x0, y0, x1, y1 = box
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(
        ((x0 + x1 - (right - left)) / 2 - left, (y0 + y1 - (bottom - top)) / 2 - top),
        text,
        font=font,
        fill=fill,
    )


def _save_cube(img: Image.Image, path: Path, cell: int = 256) -> None:
    """Squash a sign into a square cell and repeat it on all six faces of a MuJoCo cube texture.

    The geom that wears it has the sign's real aspect ratio, which stretches the text back.
    """
    sq = img.resize((cell, cell), Image.Resampling.LANCZOS)
    cube = Image.new("RGB", (cell * 4, cell * 3), sq.getpixel((2, 2)))
    # Cube cells follow the OpenGL cubemap convention: R/L = +-x, U/D = +-y, F/B = +-z. MuJoCo shows
    # L and F rotated 90 deg CCW and R rotated 90 deg CW, so pre-rotate them to read upright.
    cells = {
        "U": ((1, 0), 0),
        "L": ((0, 1), -90),
        "F": ((1, 1), -90),
        "R": ((2, 1), 90),
        "B": ((3, 1), 0),
        "D": ((1, 2), 0),
    }
    for (cx, cy), rot in cells.values():
        cube.paste(sq.rotate(rot) if rot else sq, (cx * cell, cy * cell))
    cube.save(path)


def _sign(path: Path, title_zh: str, title_en: str, rgba, w: int = 512, h: int = 256) -> None:
    img = Image.new("RGB", (w, h), _rgb(rgba, 0.85))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, w - 9, h - 9], outline=(255, 255, 255), width=8)
    # hazard stripes along the bottom
    for i in range(-h, w, 40):
        d.polygon([(i, h - 40), (i + 20, h - 40), (i + 40, h - 16), (i + 20, h - 16)], fill=(20, 20, 20))
    if HAS_CJK:
        _center_text(d, (0, 14, w, h * 0.58), title_zh, _font(int(h * 0.36)), (255, 255, 255))
        _center_text(d, (0, h * 0.55, w, h - 44), title_en, _font(int(h * 0.15)), (255, 255, 255))
    else:
        _center_text(d, (0, 10, w, h - 44), title_en, _font(int(h * 0.24)), (255, 255, 255))
    _save_cube(img, path)


def _parcel_cube(path: Path, ptype: layout.ParcelType, face: int = 96) -> None:
    """Cube texture laid out as ``.U..LFRB.D..`` on a 4x3 grid (MuJoCo gridlayout)."""
    rng = random.Random(ptype.key)
    base = _rgb(ptype.rgba, 0.95)
    img = Image.new("RGB", (face * 4, face * 3), base)
    d = ImageDraw.Draw(img)
    cells = {"U": (1, 0), "L": (0, 1), "F": (1, 1), "R": (2, 1), "B": (3, 1), "D": (1, 2)}
    glyph = {"express": "急", "standard": "普", "fragile": "易碎", "vip": "VIP", "mystery": "?"}[ptype.key]
    if not HAS_CJK:
        glyph = {"express": "EX", "standard": "ST", "fragile": "FR", "vip": "VIP", "mystery": "?"}[ptype.key]
    for name, (cx, cy) in cells.items():
        x0, y0 = cx * face, cy * face
        d.rectangle([x0, y0, x0 + face - 1, y0 + face - 1], outline=_rgb(ptype.rgba, 0.55), width=4)
        if ptype.key == "standard":  # cardboard tape
            d.rectangle([x0 + face * 0.42, y0, x0 + face * 0.58, y0 + face], fill=_rgb(ptype.rgba, 1.25))
        if name in ("U", "F", "B"):
            d.rectangle([x0 + 10, y0 + 10, x0 + face - 10, y0 + face * 0.62], fill=(250, 250, 245))
            size = int(face * (0.30 if len(glyph) > 1 else 0.42))
            _center_text(d, (x0 + 10, y0 + 10, x0 + face - 10, y0 + face * 0.62), glyph, _font(size), base)
            # barcode
            bx = x0 + 12
            while bx < x0 + face - 14:
                bw = rng.choice((2, 2, 3, 5))
                d.rectangle([bx, y0 + face * 0.70, bx + bw - 1, y0 + face - 12], fill=(25, 25, 25))
                bx += bw + rng.choice((2, 3))
        if ptype.key == "fragile" and name in ("L", "R"):
            _center_text(d, (x0, y0, x0 + face, y0 + face), "▲▲", _font(int(face * 0.4)), (250, 250, 250))
    img.save(path)


def _banner(path: Path, w: int = 1024, h: int = 192) -> None:
    img = Image.new("RGB", (w, h), (18, 24, 38))
    d = ImageDraw.Draw(img)
    d.rectangle([0, h - 18, w, h], fill=(255, 196, 0))
    zh = "孪生仓 · SO-101 智能分拣站" if HAS_CJK else "TWIN DEPOT - SO-101 SORTING CELL"
    _center_text(d, (0, 0, w, h * 0.62), zh, _font(int(h * 0.36)), (240, 244, 255))
    _center_text(
        d,
        (0, h * 0.58, w, h - 22),
        "TWIN DEPOT  //  DIGITAL TWIN SORTING CELL",
        _font(int(h * 0.15)),
        (120, 200, 255),
    )
    _save_cube(img, path, cell=512)


def _scanner(path: Path, w: int = 256, h: int = 96) -> None:
    img = Image.new("RGB", (w, h), (30, 30, 34))
    d = ImageDraw.Draw(img)
    text = "扫描区 SCAN" if HAS_CJK else "SCAN ZONE"
    _center_text(d, (0, 0, w, h), text, _font(int(h * 0.42)), (255, 70, 70))
    _save_cube(img, path)


def _floor(path: Path, n: int = 512) -> None:
    rng = random.Random(7)
    img = Image.new("RGB", (n, n), (92, 94, 98))
    px = img.load()
    assert px is not None
    for _ in range(n * n // 6):  # concrete speckle
        x, y = rng.randrange(n), rng.randrange(n)
        v = rng.randint(-14, 14)
        r, g, b = px[x, y]  # type: ignore[misc]
        px[x, y] = (r + v, g + v, b + v)
    d = ImageDraw.Draw(img)
    d.line([(0, 0), (n, 0)], fill=(70, 72, 76), width=3)
    d.line([(0, 0), (0, n)], fill=(70, 72, 76), width=3)
    img.save(path)


def build_textures(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}
    for b in layout.BINS:
        p = out_dir / f"sign_{b.key}.png"
        _sign(p, b.name_zh, b.name_en, b.rgba)
        files[f"sign_{b.key}"] = p
    for t in layout.PARCEL_TYPES:
        p = out_dir / f"parcel_{t.key}.png"
        _parcel_cube(p, t)
        files[f"parcel_{t.key}"] = p
    for name, fn in (("banner", _banner), ("scanner", _scanner), ("floor", _floor)):
        p = out_dir / f"{name}.png"
        fn(p)
        files[name] = p
    return files
