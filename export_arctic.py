import argparse
import os
import sys
from contextlib import contextmanager
from typing import Optional

import torch
from torch import Tensor

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_HOLD_ROOT = os.path.join(_REPO_ROOT, "submodules", "hold")
_HOLD_CODE_ROOT = os.path.join(_HOLD_ROOT, "code")
for _p in [_HOLD_ROOT, _HOLD_CODE_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from submodules.hold.common.xdict import xdict


J3D_MAPPING_OURS_TO_HOLD = [
    0,
    5,
    6,
    7,
    9,
    10,
    11,
    17,
    18,
    19,
    13,
    14,
    15,
    1,
    2,
    3,
    4,
    8,
    12,
    16,
    20,
]


def parse_args():
    parser = argparse.ArgumentParser(description="Postprocess a single ARCTIC sequence for export/visualization.")
    parser.add_argument(
        "--hash",
        type=str,
        default=None,
        help="HOLD experiment hashcode (uses logs_dir/hash/checkpoints/ckpt_name).",
    )
    parser.add_argument(
        "--sd_p",
        type=str,
        default=None,
        help="Explicit path to HOLD checkpoint (overrides --hash).",
    )
    parser.add_argument(
        "--logs_dir",
        type=str,
        default=os.path.join("submodules", "hold", "code", "logs"),
        help="Root logs directory containing experiment hashcode folders.",
    )
    parser.add_argument(
        "--ckpt_name",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename inside logs_dir/hash/checkpoints.",
    )
    parser.add_argument(
        "--override_pt",
        type=str,
        default=None,
        help="Override tensor path. Default: data/{seq_name}/output/combined/{seq_name}.pt",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for postprocessed .pt file. Default: ./arctic_preds",
    )
    parser.add_argument(
        "--vis_dir",
        type=str,
        default=None,
        help="Output directory for visualization exports. Default: arctic_preds_vis",
    )
    parser.add_argument("--max_obj_points", type=int, default=10000, help="Max object points kept for '*.object' tensors.")
    parser.add_argument("--frame_stride", type=int, default=50, help="Stride for exporting visualization frames.")
    parser.add_argument(
        "--to_fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Convert floating tensors to float16 before saving.",
    )
    parser.add_argument(
        "--export_vis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export visualization meshes/pointclouds.",
    )
    return parser.parse_args()


def _add_hold_to_syspath():
    for p in [_HOLD_ROOT, _HOLD_CODE_ROOT]:
        if p not in sys.path:
            sys.path.insert(0, p)


def farthest_point_sampling(xyz: Tensor, n_samples: int) -> Tensor:
    N = xyz.shape[0]
    centroids = torch.zeros(n_samples, dtype=torch.long)
    distance = torch.ones(N) * 1e10
    farthest = torch.randint(0, N, (1,), dtype=torch.long)
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = xyz[farthest].view(1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=0)[1]
    return centroids


@contextmanager
def _pushd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _maybe_reorder_j3d(key: str, val: Tensor) -> Tensor:
    if not key.startswith("j3d"):
        return val
    if val.ndim != 3 or val.shape[1] != len(J3D_MAPPING_OURS_TO_HOLD):
        return val
    return val[:, J3D_MAPPING_OURS_TO_HOLD, :]


def _make_dummy_faces(num_verts: int) -> Tensor:
    if num_verts < 3:
        return torch.empty((0, 3), dtype=torch.long)
    tri_count = num_verts // 3
    faces = torch.arange(tri_count * 3, dtype=torch.long).view(-1, 3)
    return faces


def _infer_node_id_from_key(key: str) -> Optional[str]:
    if "object" in key:
        return "object"
    if key.endswith(".left") or key.startswith("v_posed.left") or key.startswith("verts.left"):
        return "left"
    if key.endswith(".right") or key.startswith("v_posed.right") or key.startswith("verts.right"):
        return "right"
    return None


def _ensure_faces(out: xdict):
    if "faces" not in out or not isinstance(out["faces"], dict):
        out["faces"] = {}
    for node_id in ["left", "right", "object"]:
        if node_id not in out["faces"]:
            out["faces"][node_id] = torch.empty((0, 3), dtype=torch.long)


def _fix_object_faces_if_invalid(out: xdict):
    if "faces" not in out or not isinstance(out["faces"], dict):
        return
    if "object" not in out["faces"]:
        return
    face = out["faces"]["object"]
    if not isinstance(face, torch.Tensor) or face.numel() == 0:
        return
    obj_key = None
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and k.endswith(".object") and v.ndim == 3 and v.shape[-1] == 3:
            obj_key = k
            break
    if obj_key is None:
        return
    num_verts = int(out[obj_key].shape[1])
    if int(face.max()) >= num_verts:
        out["faces"]["object"] = _make_dummy_faces(num_verts)


def _downsample_object_tensors(out: xdict, max_obj_points: int):
    object_keys: list[str] = []
    ref_key = None
    for k, v in out.items():
        if isinstance(v, torch.Tensor) and k.endswith(".object") and v.ndim == 3 and v.shape[-1] == 3:
            object_keys.append(k)
            if ref_key is None:
                ref_key = k
    if ref_key is None:
        return

    ref_tensor = out[ref_key]
    T, N, _ = ref_tensor.shape
    if N <= max_obj_points:
        return

    idx = farthest_point_sampling(ref_tensor[0], max_obj_points)
    for k in object_keys:
        v = out[k]
        if not (isinstance(v, torch.Tensor) and v.ndim == 3 and v.shape[1] == N):
            continue
        out.overwrite(k, v[:, idx, :])

    _ensure_faces(out)
    out["faces"]["object"] = _make_dummy_faces(max_obj_points)


def _convert_fp16(out: xdict):
    for k, v in list(out.items()):
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            out.overwrite(k, v.to(torch.float16))
        elif k == "faces" and isinstance(v, dict):
            # faces should stay integer; just ensure type is LongTensor
            for face_key, face_value in v.items():
                if isinstance(face_value, torch.Tensor) and face_value.dtype != torch.long:
                    v[face_key] = face_value.to(torch.long)
            out.overwrite(k, v)


def _resolve_hold_ckpt(args) -> str:
    if args.sd_p:
        return args.sd_p
    if not args.hash:
        raise ValueError("Provide either --hash or --sd_p.")
    return os.path.join(args.logs_dir, args.hash, "checkpoints", args.ckpt_name)


def _ckpt_path_for_hold_loader(ckpt_path: str) -> str:
    path = os.path.normpath(ckpt_path)
    parts = path.split(os.sep)
    if "logs" in parts:
        idx = parts.index("logs")
        return os.path.join(*parts[idx:])
    return ckpt_path


def load_hold_predictions(hold_sd_p: str) -> xdict:
    _add_hold_to_syspath()
    from src.utils.io import ours as hold_ours  # type: ignore
    from src.arctic.extraction.keys import keys as hold_keys  # type: ignore

    hold_sd_p = _ckpt_path_for_hold_loader(hold_sd_p)
    with _pushd(_HOLD_CODE_ROOT):
        data_pred = hold_ours.load_data(hold_sd_p)
    data_pred = data_pred.to_16_bits().detach().to("cpu")
    out = xdict()
    for key in hold_keys:
        out[key] = data_pred[key]
    return out


def merge_hold_with_override(hold_out: xdict, override: xdict) -> xdict:
    out = hold_out
    _ensure_faces(out)

    sampled_idx = None
    for key, val in override.items():
        if isinstance(val, torch.Tensor) and key in out and isinstance(out[key], torch.Tensor):
            val = val.detach().to("cpu")
            val = _maybe_reorder_j3d(key, val)

            if key.endswith(".object") and val.ndim == 3 and out[key].ndim == 3:
                T_out, N_out, _ = out[key].shape
                T_ovr, N_ovr, _ = val.shape
                if T_out == T_ovr and N_ovr > N_out:
                    if sampled_idx is None:
                        sampled_idx = farthest_point_sampling(val[0], N_out)
                    val = val[:, sampled_idx, :]

            if val.shape == out[key].shape:
                out.overwrite(key, val)
            else:
                print(f"[SKIP] shape mismatch for {key}: override {tuple(val.shape)} vs base {tuple(out[key].shape)}")

        elif key == "faces" and isinstance(val, dict):
            for node_id, faces in val.items():
                if isinstance(faces, torch.Tensor):
                    out["faces"][node_id] = faces.detach().to("cpu").to(torch.long)
        else:
            continue

    _fix_object_faces_if_invalid(out)
    return out


def export_visualizations(out: xdict, export_dir: str, frame_stride: int):
    try:
        import trimesh  # type: ignore
    except Exception as e:
        raise RuntimeError("trimesh is required for --export_vis") from e

    os.makedirs(export_dir, exist_ok=True)
    _ensure_faces(out)

    for key, verts in out.items():
        if not (isinstance(verts, torch.Tensor) and verts.ndim == 3 and verts.shape[-1] == 3):
            continue
        if not (key.startswith("v") or key.startswith("j3d")):
            continue

        verts_f = verts.detach().cpu().float()
        T = verts_f.shape[0]

        if key.startswith("j3d"):
            for frame_idx in range(0, T, frame_stride):
                v_np = verts_f[frame_idx].numpy()
                pcd = trimesh.points.PointCloud(vertices=v_np)
                fname = os.path.join(export_dir, f"{key.replace('.', '_')}_{frame_idx:04d}.ply")
                pcd.export(fname)
            continue

        node_id = _infer_node_id_from_key(key)
        faces = out.get("faces", {}).get(node_id, None) if node_id else None
        if not (isinstance(faces, torch.Tensor) and faces.ndim == 2 and faces.shape[-1] == 3 and faces.numel() > 0):
            for frame_idx in range(0, T, frame_stride):
                v_np = verts_f[frame_idx].numpy()
                pcd = trimesh.points.PointCloud(vertices=v_np)
                fname = os.path.join(export_dir, f"{key.replace('.', '_')}_{frame_idx:04d}.ply")
                pcd.export(fname)
            continue

        faces_np = faces.detach().cpu().to(torch.long).numpy()
        num_verts = int(verts_f.shape[1])
        if faces_np.max(initial=-1) >= num_verts:
            for frame_idx in range(0, T, frame_stride):
                v_np = verts_f[frame_idx].numpy()
                pcd = trimesh.points.PointCloud(vertices=v_np)
                fname = os.path.join(export_dir, f"{key.replace('.', '_')}_{frame_idx:04d}.ply")
                pcd.export(fname)
            continue

        for frame_idx in range(0, T, frame_stride):
            v_np = verts_f[frame_idx].numpy()
            mesh = trimesh.Trimesh(vertices=v_np, faces=faces_np, process=False)
            fname = os.path.join(export_dir, f"{key.replace('.', '_')}_{frame_idx:04d}.obj")
            mesh.export(fname)


def main():
    args = parse_args()
    hold_ckpt = _resolve_hold_ckpt(args)
    if not os.path.exists(hold_ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {hold_ckpt}")

    print(f"[HOLD] Loading: {hold_ckpt}")
    hold_out = load_hold_predictions(hold_ckpt)
    seq_name = hold_out["full_seq_name"]

    override_pt = args.override_pt or os.path.join("data", seq_name, "output", "combined", f"{seq_name}.pt")
    if os.path.exists(override_pt):
        override = xdict(torch.load(override_pt, map_location="cpu"))
        out = merge_hold_with_override(hold_out, override)
    else:
        print(f"[OVERRIDE] Missing override tensor: {override_pt}")
        out = hold_out
        _ensure_faces(out)

    _downsample_object_tensors(out, max_obj_points=args.max_obj_points)
    _fix_object_faces_if_invalid(out)

    if args.to_fp16:
        _convert_fp16(out)

    out_dir = args.out_dir or "arctic_preds"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{seq_name}.pt")
    out.save(out_path)
    print(f"[SAVED] {out_path}")

    if args.export_vis:
        vis_root = args.vis_dir or "arctic_preds_vis"
        seq_vis_dir = os.path.join(vis_root, f"gs_{seq_name}_meshes")
        export_visualizations(out, export_dir=seq_vis_dir, frame_stride=args.frame_stride)


if __name__ == "__main__":
    main()
