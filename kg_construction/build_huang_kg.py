# ISBI 2027 public release.
# See the repository README for prerequisites, execution order, dataset citations, and model sources.
# Replace only PATH_TO_... entries with local paths; keep the reported experimental protocol unchanged.

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi


def ffloat(x):
    return float(x)


def build_case(pred_path, comp_dir, kg_dir, return_graph=False, save_json=True):
    _dump_json = json.dump if save_json else (lambda *args, **kwargs: None)
    case_id = pred_path.stem

    mask = tifffile.imread(pred_path)
    mask = np.asarray(mask) > 0

    if mask.ndim != 3:
        raise RuntimeError(
            f"{case_id}: expected 3D mask, got {mask.shape}"
        )

    Z, H, W = mask.shape
    total_volume_voxels = int(Z * H * W)
    total_foreground = int(mask.sum())

    if total_foreground == 0:
        raise RuntimeError(
            f"{case_id}: empty prediction mask"
        )

    # ------------------------------------------------------------
    # 26-connected components
    # ------------------------------------------------------------
    structure = np.ones((3, 3, 3), dtype=np.uint8)

    labels, n_components = ndi.label(
        mask,
        structure=structure,
    )

    counts = np.bincount(labels.ravel())

    components = []

    for component_id in range(1, n_components + 1):
        nvox = int(counts[component_id])

        if nvox == 0:
            continue

        components.append(
            {
                "source_component_id": component_id,
                "voxel_count": nvox,
            }
        )

    # Largest components first.
    # Component id is deterministic tie-breaker.
    components.sort(
        key=lambda d: (
            -d["voxel_count"],
            d["source_component_id"],
        )
    )

    # ------------------------------------------------------------
    # Minimum set covering >=99% predicted foreground mass
    # ------------------------------------------------------------
    retained = []
    cumulative = 0.0

    for comp in components:
        mass_fraction = (
            comp["voxel_count"]
            / total_foreground
        )

        cumulative += mass_fraction

        retained.append(
            {
                **comp,
                "mass_fraction": float(mass_fraction),
                "cumulative_mass_fraction": float(cumulative),
            }
        )

        if cumulative >= 0.99:
            break

    objects = ndi.find_objects(labels)

    lesion_records = []
    internal = []

    # ------------------------------------------------------------
    # Full component geometry for retained components
    # ------------------------------------------------------------
    for rank, comp in enumerate(retained, start=1):
        cid = comp["source_component_id"]

        obj = objects[cid - 1]

        if obj is None:
            raise RuntimeError(
                f"{case_id}: missing object for component {cid}"
            )

        z0, z1 = obj[0].start, obj[0].stop - 1
        y0, y1 = obj[1].start, obj[1].stop - 1
        x0, x1 = obj[2].start, obj[2].stop - 1

        centroid = ndi.center_of_mass(
            mask,
            labels,
            cid,
        )

        cz, cy, cx = [
            float(v)
            for v in centroid
        ]

        # Determine exact visible slices and lesion area on each slice.
        local = (
            labels[
                z0:z1 + 1,
                y0:y1 + 1,
                x0:x1 + 1,
            ]
            == cid
        )

        areas = local.sum(axis=(1, 2))

        visible = []

        for local_z, area in enumerate(areas):
            area = int(area)

            if area > 0:
                visible.append(
                    (
                        z0 + local_z,
                        area,
                    )
                )

        visible_slices = [
            int(z)
            for z, _ in visible
        ]

        lesion_id = (
            f"{case_id}_Lesion_{rank}"
        )

        record = {
            "lesion_id": lesion_id,
            "source_component_id": int(cid),
            "voxel_count": int(comp["voxel_count"]),
            "centroid": {
                "z": cz,
                "y": cy,
                "x": cx,
            },
            "bbox": {
                "z_min": int(z0),
                "z_max": int(z1),
                "y_min": int(y0),
                "y_max": int(y1),
                "x_min": int(x0),
                "x_max": int(x1),
            },
            "slice_count": int(len(visible_slices)),
            "first_slice": int(min(visible_slices)),
            "last_slice": int(max(visible_slices)),
            "mass_fraction": float(
                comp["mass_fraction"]
            ),
            "cumulative_mass_fraction": float(
                comp["cumulative_mass_fraction"]
            ),
        }

        lesion_records.append(record)

        internal.append(
            {
                "lesion_id": lesion_id,
                "source_component_id": int(cid),
                "voxel_count": int(comp["voxel_count"]),
                "centroid": (cz, cy, cx),
                "bbox": (
                    z0, z1,
                    y0, y1,
                    x0, x1,
                ),
                "visible": visible,
                "visible_set": set(visible_slices),
                "mass_fraction": float(
                    comp["mass_fraction"]
                ),
            }
        )

    retained_voxels = sum(
        x["voxel_count"]
        for x in retained
    )

    retained_mass = (
        retained_voxels
        / total_foreground
    )

    # ------------------------------------------------------------
    # Save mass99 lesion artifact
    # ------------------------------------------------------------
    comp_json = {
        "case_id": case_id,
        "selection_rule":
            "cumulative_foreground_mass_0.99",
        "original_components": int(n_components),
        "retained_components": int(len(retained)),
        "retained_mass_fraction": float(retained_mass),
        "lesions": lesion_records,
    }

    comp_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        comp_dir / f"{case_id}_lesions.json",
        "w",
    ) as f:
        _dump_json(
            comp_json,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Slice nodes
    # ------------------------------------------------------------
    slice_nodes = []

    for z in range(Z):
        slice_nodes.append(
            {
                "id": f"Slice_{z}",
                "type": "slice",
                "slice_index": int(z),
                "z_norm": (
                    float(z / (Z - 1))
                    if Z > 1
                    else 0.0
                ),
            }
        )

    # ------------------------------------------------------------
    # Lesion nodes
    # ------------------------------------------------------------
    lesion_nodes = []

    for lesion in internal:
        cz, cy, cx = lesion["centroid"]

        (
            z0, z1,
            y0, y1,
            x0, x1,
        ) = lesion["bbox"]

        lesion_nodes.append(
            {
                "id": lesion["lesion_id"],
                "type": "lesion",
                "source_component_id":
                    lesion["source_component_id"],
                "voxel_count":
                    lesion["voxel_count"],
                "volume_fraction": float(
                    lesion["voxel_count"]
                    / total_volume_voxels
                ),
                "foreground_mass_fraction":
                    float(
                        lesion["mass_fraction"]
                    ),
                "centroid_norm": [
                    float(cz / (Z - 1))
                    if Z > 1 else 0.0,
                    float(cy / (H - 1))
                    if H > 1 else 0.0,
                    float(cx / (W - 1))
                    if W > 1 else 0.0,
                ],
                "bbox_norm": [
                    float(z0 / (Z - 1))
                    if Z > 1 else 0.0,
                    float(z1 / (Z - 1))
                    if Z > 1 else 0.0,
                    float(y0 / (H - 1))
                    if H > 1 else 0.0,
                    float(y1 / (H - 1))
                    if H > 1 else 0.0,
                    float(x0 / (W - 1))
                    if W > 1 else 0.0,
                    float(x1 / (W - 1))
                    if W > 1 else 0.0,
                ],
                "slice_count":
                    int(len(lesion["visible"])),
                "slice_span_fraction":
                    float(
                        len(lesion["visible"])
                        / Z
                    ),
            }
        )

    # ------------------------------------------------------------
    # Lesion -> slice edges
    # ------------------------------------------------------------
    lesion_slice_edges = []

    slice_pixels = H * W

    for lesion in internal:
        for z, area in lesion["visible"]:
            lesion_slice_edges.append(
                {
                    "source":
                        lesion["lesion_id"],
                    "target":
                        f"Slice_{z}",
                    "relation":
                        "visible_on",
                    "lesion_area_pixels":
                        int(area),
                    "slice_area_fraction":
                        float(
                            area
                            / slice_pixels
                        ),
                    "lesion_area_fraction":
                        float(
                            area
                            / lesion["voxel_count"]
                        ),
                }
            )

    # ------------------------------------------------------------
    # Lesion <-> lesion co-visible edges
    # ------------------------------------------------------------
    lesion_lesion_edges = []

    def norm_centroid(lesion):
        z, y, x = lesion["centroid"]

        return np.asarray(
            [
                z / (Z - 1)
                if Z > 1 else 0.0,
                y / (H - 1)
                if H > 1 else 0.0,
                x / (W - 1)
                if W > 1 else 0.0,
            ],
            dtype=float,
        )

    for i in range(len(internal)):
        for j in range(i + 1, len(internal)):
            a = internal[i]
            b = internal[j]

            shared = (
                a["visible_set"]
                & b["visible_set"]
            )

            if not shared:
                continue

            union = (
                a["visible_set"]
                | b["visible_set"]
            )

            ca = norm_centroid(a)
            cb = norm_centroid(b)

            delta = cb - ca

            lesion_lesion_edges.append(
                {
                    "source":
                        a["lesion_id"],
                    "target":
                        b["lesion_id"],
                    "relation":
                        "co_visible",
                    "shared_slice_count":
                        int(len(shared)),
                    "slice_jaccard":
                        float(
                            len(shared)
                            / len(union)
                        ),
                    "centroid_delta_norm": [
                        float(v)
                        for v in delta
                    ],
                    "centroid_distance_norm":
                        float(
                            np.linalg.norm(delta)
                        ),
                }
            )

    # ------------------------------------------------------------
    # KG JSON
    # ------------------------------------------------------------
    graph = {
        "case_id": case_id,
        "graph_schema": {
            "node_types": [
                "slice",
                "lesion",
            ],
            "edge_types": [
                "visible_on",
                "co_visible",
            ],
            "component_selection":
                "minimum set of 26-connected components "
                "covering at least 99% of predicted foreground mass",
        },
        "volume_shape": [
            int(Z),
            int(H),
            int(W),
        ],
        "nodes": {
            "slices": slice_nodes,
            "lesions": lesion_nodes,
        },
        "edges": {
            "lesion_to_slice":
                lesion_slice_edges,
            "lesion_to_lesion":
                lesion_lesion_edges,
        },
    }

    kg_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        kg_dir / f"{case_id}_kg.json",
        "w",
    ) as f:
        _dump_json(
            graph,
            f,
            indent=2,
        )

    summary = {
        "case_id": case_id,
        "original_components":
            int(n_components),
        "retained_components":
            int(len(retained)),
        "total_foreground_voxels":
            int(total_foreground),
        "retained_foreground_voxels":
            int(retained_voxels),
        "retained_mass_fraction":
            float(retained_mass),
        "slices": int(Z),
        "lesions": int(len(lesion_nodes)),
        "lesion_slice_edges":
            int(len(lesion_slice_edges)),
        "lesion_lesion_edges":
            int(len(lesion_lesion_edges)),
    }

    if return_graph:
        return summary, graph

    return summary


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--pred-dir",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--components-dir",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--kg-dir",
        required=True,
        type=Path,
    )

    ap.add_argument(
        "--case",
        default=None,
    )

    args = ap.parse_args()

    if args.case:
        files = [
            args.pred_dir
            / f"{args.case}.tif"
        ]
    else:
        files = sorted(
            args.pred_dir.glob("*.tif")
        )

    if not files:
        raise RuntimeError(
            "No prediction TIFF files found."
        )

    rows = []

    for i, p in enumerate(files, start=1):
        if not p.exists():
            raise FileNotFoundError(p)

        row = build_case(
            p,
            args.components_dir,
            args.kg_dir,
        )

        rows.append(row)

        print(
            f"[{i:03d}/{len(files):03d}] "
            f"{row['case_id']} | "
            f"components "
            f"{row['original_components']} -> "
            f"{row['retained_components']} | "
            f"mass={row['retained_mass_fraction']:.6f} | "
            f"LS={row['lesion_slice_edges']} | "
            f"LL={row['lesion_lesion_edges']}"
        )

    df = pd.DataFrame(rows)

    df[
        [
            "case_id",
            "original_components",
            "retained_components",
            "total_foreground_voxels",
            "retained_foreground_voxels",
            "retained_mass_fraction",
        ]
    ].to_csv(
        args.components_dir
        / "case_mass99_summary.csv",
        index=False,
    )

    df[
        [
            "case_id",
            "slices",
            "lesions",
            "lesion_slice_edges",
            "lesion_lesion_edges",
        ]
    ].to_csv(
        args.kg_dir
        / "graph_summary.csv",
        index=False,
    )

    print("\nFinished:", len(df), "case(s)")


if __name__ == "__main__":
    main()
