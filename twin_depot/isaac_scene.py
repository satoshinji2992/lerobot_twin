"""Import an NVIDIA Isaac Sim warehouse (the one Isaac Lab uses) as a static MuJoCo backdrop.

Isaac Lab loads its environments from NVIDIA's public asset server ("Nucleus" on S3), e.g.
``{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/full_warehouse.usd``. This module:

1. ``download``: fetches the USD and everything it references (props, MDL materials, textures)
   into ``assets/isaac/raw/`` - a mirror of the server layout, ~380 MB for the full warehouse.
2. ``convert``: flattens every visible mesh into world space, bakes the Unreal-style MDL material
   parameters (albedo x tint, masked colour, crate channels, decal alpha) into small PNGs and
   merges all meshes that share a look into one MuJoCo ``.msh`` (positions, normals, UVs).
   ~3500 meshes / 680k triangles become ~60 draw calls. Output: ``assets/isaac/<scene>/``.
3. ``mjcf``: the ``<asset>`` and ``<worldbody>`` XML for ``scene.py``.

Only ``download``/``convert`` need OpenUSD (``pip install usd-core``); building the MuJoCo scene
from an already converted warehouse needs nothing extra. The warehouse is purely visual - the
work cell (table, conveyor, bins) keeps its own colliders and a keep-out box clears the props
that would stand inside it.

The assets are NVIDIA's (Isaac Sim asset license): they stay in ``assets/`` (git-ignored) and
are not redistributed.

    python import_isaac_scene.py            # download + convert (once)
    python play.py --hall isaac              # use it
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os
import struct
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import layout

NUCLEUS_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/"
SCENES = {  # Isaac Sim 5.1 / Isaac Lab 2.3 environments
    "full_warehouse": "Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "warehouse": "Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "warehouse_multiple_shelves": "Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    "warehouse_with_forklifts": "Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd",
}
ISAAC_DIR = Path(__file__).resolve().parents[1] / "assets" / "isaac"
RAW_DIR = ISAAC_DIR / "raw"
FORMAT = 2  # bump when the converted output changes
# Props that become vehicles (mocap bodies the game drives) instead of static backdrop.
ACTORS = {"forklift": "/Root/forklift"}


@dataclass(frozen=True)
class Placement:
    """Where the work cell sits in the warehouse: twin = Rz(yaw) @ (usd - anchor), floor at FLOOR_Z."""

    anchor: tuple[float, float]
    yaw_deg: float
    keepout: tuple[float, float, float, float]  # x0, x1, y0, y1 in twin coordinates: props removed

    def to_twin(self, pts: np.ndarray) -> np.ndarray:
        a = math.radians(self.yaw_deg)
        c, s = math.cos(a), math.sin(a)
        x, y = pts[:, 0] - self.anchor[0], pts[:, 1] - self.anchor[1]
        return np.stack([c * x - s * y, s * x + c * y, pts[:, 2] + layout.FLOOR_Z], axis=1)

    def rotate(self, vec: np.ndarray) -> np.ndarray:
        a = math.radians(self.yaw_deg)
        c, s = math.cos(a), math.sin(a)
        return np.stack([c * vec[:, 0] - s * vec[:, 1], s * vec[:, 0] + c * vec[:, 1], vec[:, 2]], axis=1)


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(f"[isaac] {msg}", flush=True)


def _require_usd():
    try:
        from pxr import Sdf, Usd, UsdGeom, UsdShade  # noqa: F401
    except ImportError:
        raise SystemExit(
            "OpenUSD is needed to import the Isaac warehouse: pip install usd-core\n"
            "(only for the one-time import; playing needs nothing extra)"
        ) from None


# ------------------------------------------------------------------------------------------
# 1. download
# ------------------------------------------------------------------------------------------
def _layer_deps(url: str, path: Path) -> list[str]:
    from pxr import Sdf

    if path.suffix not in (".usd", ".usda", ".usdc"):
        return []
    lyr = Sdf.Layer.FindOrOpen(str(path))
    if lyr is None:
        return []
    found = set(lyr.subLayerPaths)

    def visit(spec_path):
        spec = lyr.GetObjectAtPath(spec_path)
        if isinstance(spec, Sdf.PrimSpec):
            for ref in [
                *spec.referenceList.GetAddedOrExplicitItems(),
                *spec.payloadList.GetAddedOrExplicitItems(),
            ]:
                if ref.assetPath:
                    found.add(ref.assetPath)
        elif isinstance(spec, Sdf.AttributeSpec) and spec.typeName == Sdf.ValueTypeNames.Asset:
            v = spec.default
            if v is not None and v.path:
                found.add(v.path)

    lyr.Traverse(Sdf.Path.absoluteRootPath, visit)
    out = []
    for a in found:
        a = a.replace("\\", "/")
        if "://" in a or a.startswith("/") or "<UDIM>" in a or ("/" not in a and a.endswith(".mdl")):
            continue  # absolute / built-in (OmniPBR.mdl) / UDIM sets
        out.append(urllib.parse.urljoin(url, a))
    return out


def download(scene: str = "full_warehouse", verbose: bool = True) -> Path:
    """Mirror the scene USD and all of its dependencies under ``assets/isaac/raw``."""
    _require_usd()
    start = NUCLEUS_ROOT + SCENES[scene]

    def local(url: str) -> Path:
        return RAW_DIR / urllib.parse.unquote(url[len(NUCLEUS_ROOT) :])

    def fetch(url: str):
        p = local(url)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_name(p.name + ".part")
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                        f.write(r.read())
                    os.replace(tmp, p)
                    break
                except Exception as e:  # flaky TLS on long crawls: retry
                    if attempt == 3:
                        return url, None, str(e)
                    time.sleep(1.5 * (attempt + 1))
        return url, p, None

    seen, frontier, errors, t0 = {start}, [start], [], time.time()
    _log(verbose, f"downloading {SCENES[scene]} and its dependencies from NVIDIA's asset server ...")
    with cf.ThreadPoolExecutor(16) as ex:
        while frontier:
            results, frontier = list(ex.map(fetch, frontier)), []
            for url, p, err in results:
                if err:
                    errors.append((url, err))
                    continue
                for dep in _layer_deps(url, p):
                    if dep.startswith(NUCLEUS_ROOT) and dep not in seen:
                        seen.add(dep)
                        frontier.append(dep)
            _log(verbose, f"  {len(seen)} files ({time.time() - t0:.0f} s)")
    if errors:
        _log(verbose, f"  {len(errors)} files failed (re-run to retry), e.g. {errors[0][0]}")
    return local(start)


# ------------------------------------------------------------------------------------------
# 2. convert
# ------------------------------------------------------------------------------------------
@dataclass
class _Look:
    key: str
    rgba: tuple[float, float, float, float]
    texture: str | None  # file name inside the output textures/ dir
    emission: float = 0.0
    alpha: bool = False  # decal: texture alpha cuts it out


def _asset(v) -> str | None:
    p = getattr(v, "resolvedPath", "") or ""
    return p if p and os.path.exists(p) else None


def _vec(v, n=4, default=1.0):
    if v is None:
        return (default,) * n
    return tuple(float(x) for x in list(v)[:n])


COLOR_STEP = 0.06  # props carry per-instance random tints: snap them so looks can be shared


def _quantize(v):
    """Snap colour-like (vector) shader inputs; leave textures and scalars alone."""
    if hasattr(v, "resolvedPath") or isinstance(v, str | bool | int | float) or v is None:
        return v
    try:
        return tuple(round(float(x) / COLOR_STEP) * COLOR_STEP for x in v)
    except TypeError:
        return v


_LOOK_INPUTS = {
    "AlbedoTexture", "BaseColor_Texture", "diffuse_texture", "BaseColor_Box", "Text", "TextureSelection",
    "BaseColor_Tint", "BaseColorBox_Tint", "ColorAlbedo", "diffuse_color_constant", "MaskSelection",
    "AlphaSelection", "Body", "Cap", "Handle",
}  # fmt: skip


class _Baker:
    """Turns MDL shader inputs into one albedo PNG + an rgba tint for MuJoCo."""

    def __init__(self, out_dir: Path, tex_size: int):
        from PIL import Image

        self.Image = Image
        self.dir = out_dir / "textures"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.size = tex_size
        self.cache: dict[tuple, _Look] = {}
        self._img: dict[tuple, np.ndarray] = {}

    def _load(self, path: str, size: int | None = None, mode: str = "RGB") -> np.ndarray:
        s = size or self.size
        key = (path, s, mode)
        if key not in self._img:
            im = self.Image.open(path).convert(mode)
            if max(im.size) > s:
                im = im.resize((s, s), self.Image.Resampling.LANCZOS)
            self._img[key] = np.asarray(im).astype(np.float32) / 255.0
        return self._img[key]

    def _plain(self, path: str, size: int) -> str:
        fn = f"{Path(path).stem}_{size}.png"
        if not (self.dir / fn).exists():
            self._save(self._load(path, size), fn[:-4])
        return fn

    def _save(self, arr: np.ndarray, name: str) -> str:
        mode = "RGBA" if arr.shape[2] == 4 else "RGB"
        fn = f"{name}.png"
        self.Image.fromarray(np.clip(arr * 255 + 0.5, 0, 255).astype(np.uint8), mode).save(self.dir / fn)
        return fn

    @staticmethod
    def _kind(mat_name: str) -> str:
        """The only things the material *name* decides (everything else comes from its inputs)."""
        low = mat_name.lower().replace("_", "")
        for k in ("glow", "cone", "wetfloor", "sign", "stripe", "decal", "barcode", "board"):
            if k in low:
                return k
        return ""

    def look(self, mat_name: str, ins: dict) -> _Look:
        # every prop file carries its own copy of its material: key on what it resolves to
        # (normal/ORM/roughness maps are irrelevant for MuJoCo, UV tiling is baked into the UVs)
        ins = {k: _quantize(v) for k, v in ins.items() if k in _LOOK_INPUTS}
        norm = tuple(
            sorted((k, _asset(v) or "" if hasattr(v, "resolvedPath") else str(v)) for k, v in ins.items())
        )
        sig = (self._kind(mat_name), norm)
        if sig in self.cache:
            return self.cache[sig]
        look = self._bake(sig[0], ins, f"t{len(self.cache):03d}")
        self.cache[sig] = look
        return look

    def _bake(self, kind: str, ins: dict, name: str) -> _Look:
        if kind == "glow":
            return _Look(name, (1.0, 0.98, 0.92, 1.0), None, emission=1.0)
        alb = next(
            (
                _asset(ins[k])
                for k in (
                    "AlbedoTexture",
                    "BaseColor_Texture",
                    "diffuse_texture",
                    "BaseColor_Box",
                    "Text",
                    "TextureSelection",
                )
                if k in ins and _asset(ins[k])
            ),
            None,
        )
        big = kind in ("sign", "stripe", "decal", "barcode", "board")
        size = self.size * 2 if big else self.size
        tint = _vec(ins.get("BaseColor_Tint") or ins.get("BaseColorBox_Tint"))
        if alb is None:
            col = ins.get("ColorAlbedo") or ins.get("diffuse_color_constant") or ins.get("Body")
            if col is not None:
                return _Look(name, (*_vec(col, 3), 1.0), None)
            fallback = {"cone": (1.0, 0.35, 0.05), "wetfloor": (1.0, 0.85, 0.1), "sign": (0.9, 0.9, 0.85)}
            rgb = fallback.get(kind, (0.6, 0.6, 0.62))
            return _Look(name, (*rgb, 1.0), None)
        mask_path = _asset(ins.get("MaskSelection")) if "MaskSelection" in ins else None
        alpha_path = _asset(ins.get("AlphaSelection")) if "AlphaSelection" in ins else None
        painted = mask_path and ("ColorAlbedo" in ins or all(k in ins for k in ("Body", "Cap", "Handle")))
        if not painted and not alpha_path:  # plain albedo x tint: share one texture file per albedo
            return _Look(name, (*_vec(tint, 3), 1.0), self._plain(alb, size))
        a = self._load(alb, size if alpha_path else self.size // 2).copy()
        if mask_path and all(k in ins for k in ("Body", "Cap", "Handle")):  # plastic crates/carts
            m = self._load(mask_path, a.shape[0])
            lum = a.mean(axis=2, keepdims=True) / max(1e-3, float(a.mean()))
            cols = [np.array(_vec(ins[k], 3)) * 2.2 for k in ("Body", "Cap", "Handle")]
            painted = sum(m[..., i : i + 1] * cols[i] for i in range(3))
            w = np.clip(m.sum(axis=2, keepdims=True), 0, 1)
            a = a * (1 - w) + np.clip(lum * painted, 0, 1) * w
        elif mask_path and "ColorAlbedo" in ins:  # masked paint (rack frames, ceiling, floor)
            m = self._load(mask_path, a.shape[0])[..., :1]
            col = np.array(_vec(ins["ColorAlbedo"], 3))
            lum = a.mean(axis=2, keepdims=True) / max(1e-3, float(a.mean()))
            a = a * (1 - m) + np.clip(lum * col * 1.6, 0, 1) * m
        alpha = None
        if alpha_path:
            alpha = self._load(alpha_path, a.shape[0])[..., :1]
            a = np.concatenate([a, alpha], axis=2)
        fn = self._save(a, name)
        return _Look(name, (*_vec(tint, 3), 1.0), fn, alpha=alpha is not None)


def _material_inputs(prim) -> tuple[str, dict]:
    from pxr import UsdShade

    mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not mat:
        return "none", {}
    ins = {}
    for child in mat.GetPrim().GetChildren():
        if child.IsA(UsdShade.Shader):
            for i in UsdShade.Shader(child).GetInputs():
                ins[i.GetBaseName()] = i.Get()
    return mat.GetPrim().GetName(), ins


def _tiling(ins: dict) -> tuple[float, float]:
    if "U_Tiling" in ins or "V_Tiling" in ins:
        return float(ins.get("U_Tiling") or 1.0), float(ins.get("V_Tiling") or 1.0)
    if ins.get("MainTiling") is not None:
        t = list(ins["MainTiling"])
        return float(t[0] or 1.0), float(t[1] or 1.0)
    return 1.0, 1.0


def _primvar(mesh, names, n_points, n_corners, counts):
    """Per-corner values of the first matching primvar (flattened), or None."""
    from pxr import UsdGeom

    api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    for nm in names:
        pv = api.GetPrimvar(nm)
        if not pv or not pv.HasValue():
            continue
        vals = pv.ComputeFlattened()
        if vals is None or len(vals) == 0:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        interp = pv.GetInterpolation()
        return arr, interp
    return None, None


def _expand(arr, interp, idx, counts):
    """Map a primvar to face-vertex corners."""
    if arr is None:
        return None
    n_corners = len(idx)
    if interp == "faceVarying" and len(arr) == n_corners:
        return arr
    if interp in ("vertex", "varying") and len(arr) > idx.max():
        return arr[idx]
    if interp == "uniform" and len(arr) == len(counts):
        return np.repeat(arr, counts, axis=0)
    if interp == "constant" or len(arr) == 1:
        return np.repeat(arr[:1], n_corners, axis=0)
    return None


def _triangulate(counts: np.ndarray) -> np.ndarray:
    """Corner indices (into the face-vertex list) of a fan triangulation."""
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    tris = []
    for k in range(1, int(counts.max()) - 1 if len(counts) else 0):
        sel = counts > k + 1
        s = starts[sel]
        tris.append(np.stack([s, s + k, s + k + 1], axis=1))
    return np.concatenate(tris) if tris else np.zeros((0, 3), dtype=np.int64)


def _write_msh(path: Path, v: np.ndarray, n: np.ndarray, uv: np.ndarray, f: np.ndarray) -> None:
    with open(path, "wb") as fh:
        fh.write(struct.pack("4i", len(v), len(n), len(uv), len(f)))
        fh.write(v.astype("<f4").tobytes())
        fh.write(n.astype("<f4").tobytes())
        fh.write(uv.astype("<f4").tobytes())
        fh.write(f.astype("<i4").tobytes())


def default_placement() -> Placement:
    p = layout.ISAAC_PLACEMENT
    return Placement(tuple(p["anchor"]), float(p["yaw_deg"]), tuple(p["keepout"]))


def convert(
    scene: str = "full_warehouse",
    placement: Placement | None = None,
    tex_size: int = 512,
    verbose: bool = True,
) -> Path:
    """Bake the downloaded USD into ``assets/isaac/<scene>/`` (meshes, textures, manifest)."""
    _require_usd()
    from pxr import Usd, UsdGeom

    placement = placement or default_placement()
    src = RAW_DIR / SCENES[scene]
    if not src.exists():
        src = download(scene, verbose)
    out = ISAAC_DIR / scene
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    for old in (out / "meshes").glob("*.msh"):
        old.unlink()
    t0 = time.time()
    stage = Usd.Stage.Open(str(src))
    up = UsdGeom.GetStageUpAxis(stage)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    if up != "Z":
        raise SystemExit(f"{src.name}: up axis {up} not supported")
    baker = _Baker(out, tex_size)
    groups: dict[str, dict] = {}
    looks: dict[str, _Look] = {}
    kx0, kx1, ky0, ky1 = placement.keepout
    stats = {"meshes": 0, "skipped_keepout": 0, "tris": 0}
    actor_groups: dict[str, dict[str, dict]] = {a: {} for a in ACTORS}
    fork_pts: dict[str, list[np.ndarray]] = {a: [] for a in ACTORS}
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in stage.Traverse(Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == "invisible" or img.ComputePurpose() not in ("default", "render"):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        idx = mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts or not idx:
            continue
        pts = np.asarray(pts, dtype=np.float64) * mpu
        counts = np.asarray(counts, dtype=np.int64)
        idx = np.asarray(idx, dtype=np.int64)
        if counts.sum() != len(idx) or idx.max() >= len(pts):
            continue
        m = np.array(xf_cache.GetLocalToWorldTransform(prim), dtype=np.float64)  # row-vector convention
        m[3, :3] *= mpu
        world = pts @ m[:3, :3] + m[3, :3]
        tw = placement.to_twin(world)
        lo, hi = tw.min(axis=0), tw.max(axis=0)
        path = str(prim.GetPath())
        actor = next((a for a, root in ACTORS.items() if path.startswith(root + "/")), None)
        flat = hi[2] - lo[2] < 0.05 and hi[2] < layout.FLOOR_Z + 0.06  # floor tiles and decals stay
        if (
            actor is None
            and not flat
            and lo[0] < kx1
            and hi[0] > kx0
            and lo[1] < ky1
            and hi[1] > ky0
            and lo[2] < layout.FLOOR_Z + 3.0
        ):
            stats["skipped_keepout"] += 1
            continue
        mat_name, ins = _material_inputs(prim)
        look = baker.look(mat_name, ins)
        looks[look.key] = look
        # per-corner attributes
        n_arr, n_int = _primvar(mesh, ["normals"], len(pts), len(idx), counts)
        if n_arr is None:
            nv = mesh.GetNormalsAttr().Get()
            if nv:
                n_arr, n_int = np.asarray(nv, dtype=np.float64), mesh.GetNormalsInterpolation()
        uv_arr, uv_int = _primvar(mesh, ["st", "st0", "UVMap", "uv", "map1"], len(pts), len(idx), counts)
        normals = _expand(n_arr, n_int, idx, counts)
        uvs = _expand(uv_arr, uv_int, idx, counts)
        corners = tw[idx]
        tri = _triangulate(counts)
        if mesh.GetOrientationAttr().Get() == "leftHanded":
            tri = tri[:, [0, 2, 1]]
        det = np.linalg.det(m[:3, :3])
        if det < 0:  # mirrored instance
            tri = tri[:, [0, 2, 1]]
        if normals is not None:
            nrm = normals @ np.linalg.inv(m[:3, :3]).T  # row vectors: n' = n M^-T
            nrm = placement.rotate(nrm)
            nrm /= np.maximum(1e-12, np.linalg.norm(nrm, axis=1, keepdims=True))
        else:
            nrm = None
        if uvs is not None:
            su, sv = _tiling(ins)
            uvs = uvs[:, :2] * (su, sv)
            uvs[:, 1] = 1.0 - uvs[:, 1]  # USD st: origin bottom-left; MuJoCo textures: top row first
        bucket = groups if actor is None else actor_groups[actor]
        if actor is not None and "fork" in prim.GetName().lower():
            fork_pts[actor].append(corners)
        g = bucket.setdefault(look.key, {"v": [], "n": [], "uv": [], "f": [], "count": 0})
        g["v"].append(corners)
        if nrm is None:  # flat normals per triangle, per corner
            a, b, c = corners[tri[:, 0]], corners[tri[:, 1]], corners[tri[:, 2]]
            fn = np.cross(b - a, c - a)
            fn /= np.maximum(1e-12, np.linalg.norm(fn, axis=1, keepdims=True))
            nrm = np.zeros_like(corners)
            for k in range(3):
                nrm[tri[:, k]] = fn
        g["n"].append(nrm)
        g["uv"].append(uvs if uvs is not None else np.zeros((len(corners), 2)))
        g["f"].append(tri + g["count"])
        g["count"] += len(corners)
        stats["meshes"] += 1
        stats["tris"] += len(tri)
    _log(
        verbose,
        f"read {stats['meshes']} meshes, {stats['tris']} triangles ({time.time() - t0:.0f} s); merging ...",
    )
    manifest = {"format": FORMAT, "scene": scene, "placement": placement.__dict__, "groups": []}
    manifest["groups"] = _write_groups(groups, looks, out / "meshes", "")
    manifest["actors"] = {}
    for name, agroups in actor_groups.items():
        if not agroups:
            continue
        pose, half_len = _actor_frame(agroups, fork_pts[name])
        manifest["actors"][name] = {
            "pose": pose,
            "half_len": half_len,
            "groups": _write_groups(agroups, looks, out / "meshes", f"actor_{name}_"),
        }
    manifest["stats"] = stats
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    _log(
        verbose,
        f"wrote {len(manifest['groups'])} merged meshes, {len(baker.cache)} looks -> {out} "
        f"({time.time() - t0:.0f} s, {stats['skipped_keepout']} props cleared for the work cell)",
    )
    return out / "manifest.json"


def _write_groups(groups: dict, looks: dict, mesh_dir: Path, prefix: str) -> list[dict]:
    out = []
    for key, g in groups.items():
        v = np.concatenate(g["v"])
        n = np.concatenate(g["n"])
        uv = np.concatenate(g["uv"])
        f = np.concatenate(g["f"])
        # share identical corners (position + normal + uv) to shrink the files
        rows = np.round(np.concatenate([v, n * 1e-3, uv * 1e-2], axis=1), 5)
        uniq, inv = np.unique(rows, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        first = np.zeros(len(uniq), dtype=np.int64)
        first[inv[::-1]] = np.arange(len(inv))[::-1]
        v, n, uv, f = v[first], n[first], uv[first], inv[f]
        keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
        f = f[keep]
        if len(f) == 0:
            continue
        look = looks[key]
        fn = f"{prefix}{key}.msh"
        _write_msh(mesh_dir / fn, v, n, uv if look.texture else np.zeros((0, 2)), f)
        out.append(
            {
                "mesh": fn,
                "texture": look.texture,
                "rgba": look.rgba,
                "emission": look.emission,
                "alpha": look.alpha,
                "tris": int(len(f)),
                "bounds": [v.min(axis=0).tolist(), v.max(axis=0).tolist()],
            }
        )
    return out


def _actor_frame(groups: dict, fork: list[np.ndarray]) -> tuple[list[float], float]:
    """Move an actor's corners (twin frame, in place) into a body frame: origin at the footprint
    centre on the floor, +x pointing towards the forks. Returns (x, y, yaw_deg) and half length."""
    allv = np.concatenate([v for g in groups.values() for v in g["v"]])
    xy = allv[:, :2]
    c0 = xy.mean(axis=0)
    w, vecs = np.linalg.eigh(np.cov((xy - c0).T))
    axis = vecs[:, np.argmax(w)]
    if fork:
        fc = np.concatenate(fork)[:, :2].mean(axis=0)
        if (fc - c0) @ axis < 0:
            axis = -axis
    yaw = math.atan2(axis[1], axis[0])
    c, s = math.cos(-yaw), math.sin(-yaw)
    rot = np.array([[c, -s], [s, c]])
    loc = (xy - c0) @ rot.T
    off = (loc.min(axis=0) + loc.max(axis=0)) / 2
    origin = c0 + np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ off
    half_len = float((loc.max(axis=0) - loc.min(axis=0))[0] / 2)
    for g in groups.values():
        for k, v in enumerate(g["v"]):
            lv = (v[:, :2] - origin) @ rot.T
            g["v"][k] = np.concatenate([lv, v[:, 2:3] - layout.FLOOR_Z], axis=1)
        for k, n in enumerate(g["n"]):
            g["n"][k] = np.concatenate([n[:, :2] @ rot.T, n[:, 2:3]], axis=1)
    return [float(origin[0]), float(origin[1]), math.degrees(yaw)], half_len


# ------------------------------------------------------------------------------------------
# 3. use
# ------------------------------------------------------------------------------------------
def load_manifest(scene: str = "full_warehouse") -> dict | None:
    path = ISAAC_DIR / scene / "manifest.json"
    if not path.exists():
        return None
    man = json.loads(path.read_text(encoding="utf-8"))
    want = json.loads(json.dumps(default_placement().__dict__))  # tuples -> lists, like the file
    if man.get("format") != FORMAT or man.get("placement") != want:
        return None  # stale: converted by another version or for another placement
    man["dir"] = str(path.parent)
    return man


def available(scene: str = "full_warehouse") -> bool:
    return load_manifest(scene) is not None


def mjcf(man: dict, groups: list[dict] | None = None, prefix: str = "isaac") -> tuple[list[str], list[str]]:
    """``<asset>`` and geom XML lines for a converted warehouse (or one of its actors)."""
    base = Path(man["dir"])
    assets, world = [], []
    for k, g in enumerate(man["groups"] if groups is None else groups):
        name = f"{prefix}_{k:03d}"
        assets.append(
            f'<mesh name="{name}" file="{(base / "meshes" / g["mesh"]).as_posix()}" inertia="shell"/>'
        )
        r, gg, b, a = g["rgba"]
        mat = (
            f'<material name="{name}" rgba="{r:.3f} {gg:.3f} {b:.3f} {a:.3f}" specular="0.15" shininess="0.3"'
        )
        if g["emission"]:
            mat += f' emission="{g["emission"]:.2f}"'
        if g["texture"]:
            assets.append(
                f'<texture name="{name}" type="2d" file="{(base / "textures" / g["texture"]).as_posix()}"/>'
            )
            mat += f' texture="{name}"'
        assets.append(mat + "/>")
        world.append(f'<geom name="{name}" class="deco" type="mesh" mesh="{name}" material="{name}"/>')
    return assets, world


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Import an Isaac Sim warehouse for Twin Depot (needs usd-core).")
    p.add_argument("--scene", choices=list(SCENES), default="full_warehouse")
    p.add_argument("--tex-size", type=int, default=512, help="max albedo texture size (signs/decals get 2x)")
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    download(args.scene)
    if not args.download_only:
        convert(args.scene, tex_size=args.tex_size)
