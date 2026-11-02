#!/usr/bin/env python3
"""Normalize explicit data partitions."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import tempfile
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tefed.data import load_manifest, oriented_to_horizontal, validate_manifest


def read_class_map(path: Path):
    mapping = json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(mapping, dict) or not mapping
            or any(not isinstance(name, str) or not name.strip() for name in mapping)
            or any(type(value) is not int for value in mapping.values())
            or sorted(mapping.values()) != list(range(len(mapping)))):
        raise ValueError("class-map must map category names to contiguous integer class IDs starting at zero")
    return mapping


def read_class_aliases(path: Path | None, class_map):
    if path is None:
        return {}
    aliases = json.loads(path.read_text(encoding="utf-8-sig"))
    if (not isinstance(aliases, dict)
            or any(not isinstance(name, str) or not name.strip() for name in aliases)
            or any(not isinstance(name, str) or name not in class_map for name in aliases.values())
            or any(name in class_map and target != name for name, target in aliases.items())):
        raise ValueError("class-aliases must map source names to existing canonical categories")
    return aliases


def read_ir(path: Path, image_size: tuple[int, int]):
    """Read the single-channel IR formats accepted by the data loader."""
    import numpy as np
    from PIL import Image

    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            array = np.array(image)
    if array.ndim != 2:
        raise ValueError(f"IR must be single-channel: {path}")
    if (array.shape[1], array.shape[0]) != image_size:
        raise ValueError(f"RGB/IR dimensions differ for {path.name}; prepare registered inputs first")
    if array.dtype != np.uint8:
        if (not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all()
                or np.any(array < 0) or np.any(array > 1)):
            raise ValueError(f"IR must be uint8 or finite floating-point values in [0,1]: {path}")
    return array


def _stem_index(directory: Path, suffix: str, name: str):
    if not directory.is_dir():
        raise FileNotFoundError(f"{name} directory not found: {directory}")
    if (not isinstance(suffix, str) or not suffix.startswith(".") or len(suffix) < 2
            or any(value in suffix for value in ("/", "\\", "*", "?"))):
        raise ValueError(f"{name} suffix must be an explicit extension, such as .jpg")
    indexed = {}
    resolved = set()
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.casefold() != suffix.casefold():
            continue
        key = path.stem.casefold()
        if key in indexed or path.resolve() in resolved:
            raise ValueError(f"Duplicate {name} sample ID or asset: {path.stem}")
        indexed[key] = path
        resolved.add(path.resolve())
    return indexed


def _annotation_index(directory):
    result = {}
    for suffix in (".txt", ".xml"):
        for key, file in _stem_index(directory, suffix, "annotation").items():
            if key in result:
                raise ValueError(f"Ambiguous annotation ID {file.stem}: both TXT and XML are present")
            result[key] = file
    return dict(sorted(result.items()))


def read_dronevehicle_labels(file: Path, image_size, class_map, difficult_policy="error", class_aliases=None):
    if difficult_policy not in {"error", "include", "drop"}:
        raise ValueError("difficult-policy must be error, include or drop")
    aliases = {} if class_aliases is None else class_aliases
    entries = []
    if file.suffix.casefold() == ".xml":
        try:
            tree = ElementTree.parse(file).getroot()
        except ElementTree.ParseError as error:
            raise ValueError(f"{file}: invalid XML annotation") from error
        try:
            declared = (int(tree.findtext("size/width")), int(tree.findtext("size/height")))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{file}: XML needs integer size/width and size/height") from error
        if declared != tuple(image_size):
            raise ValueError(f"{file}: XML size {declared} differs from RGB size {tuple(image_size)}")
        for index, item in enumerate(tree.findall("object"), 1):
            try:
                coordinates = [float(item.findtext(f"polygon/{axis}{corner}"))
                               for corner in range(1, 5) for axis in ("x", "y")]
                difficulty = int(item.findtext("difficult", "0"))
            except (TypeError, ValueError) as error:
                raise ValueError(f"{file}: object {index} needs eight polygon coordinates and integer difficult") from error
            entries.append((index, (item.findtext("name") or "").strip(), coordinates, difficulty))
    elif file.suffix.casefold() == ".txt":
        for line_no, line in enumerate(file.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip() or line.strip().lower().startswith(("imagesource:", "gsd:")):
                continue
            fields = line.split()
            if len(fields) < 9:
                raise ValueError(f"{file}:{line_no}: expected 8 corner coordinates, class and optional difficulty")
            name, difficulty = " ".join(fields[8:]), 0
            if name not in class_map and name not in aliases and len(fields) >= 10:
                try:
                    difficulty = int(fields[-1])
                except ValueError as error:
                    raise ValueError(f"{file}:{line_no}: unknown category or invalid difficulty") from error
                name = " ".join(fields[8:-1])
            entries.append((line_no, name, [float(x) for x in fields[:8]], difficulty))
    else:
        raise ValueError("DroneVehicle annotations must be TXT or XML")
    labels = []
    for line_no, name, coordinates, difficult in entries:
        name = aliases.get(name, name)
        if name not in class_map:
            raise ValueError(f"{file}:{line_no}: unknown category {name!r}")
        if difficult < 0:
            raise ValueError(f"{file}:{line_no}: difficulty must be nonnegative")
        if difficult and difficult_policy == "error":
            raise ValueError(f"{file}:{line_no}: choose an explicit difficult-policy")
        if difficult and difficult_policy == "drop":
            continue
        box = oriented_to_horizontal(coordinates, image_size=image_size, normalize=True)
        labels.append({"box": box, "class_id": class_map[name]})
    return labels


def coco_records(path: Path, image_root: Path, client_id: str, split: str):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not all(isinstance(data.get(k), list) for k in ("images", "annotations", "categories")):
        raise ValueError("COCO input requires images, annotations and categories lists")
    categories = sorted(data["categories"], key=lambda x: int(x["id"]))
    category_map = {int(c["id"]): i for i, c in enumerate(categories)}
    if len(category_map) != len(categories):
        raise ValueError("COCO categories contain duplicate IDs")
    images = {i["id"]: i for i in data["images"]}
    if len(images) != len(data["images"]):
        raise ValueError("COCO images contain duplicate IDs")
    annotations = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0) or ann.get("ignore", 0):
            raise ValueError("Crowd/ignore annotations need an explicit supported evaluation protocol; not silently discarded")
        if ann.get("image_id") not in images or ann.get("category_id") not in category_map:
            raise ValueError("COCO annotation refers to an unknown image/category")
        image = images[ann["image_id"]]
        width, height = image["width"], image["height"]
        if width <= 0 or height <= 0:
            raise ValueError("COCO image width/height must be positive")
        if "quadrilateral" in ann:
            box = oriented_to_horizontal(ann["quadrilateral"], image_size=(width, height), normalize=True)
        else:
            bbox = ann.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError("COCO bbox must be pixel xywh; use quadrilateral for four explicit corners")
            x, y, w, h = map(float, bbox)
            box = [x / width, y / height, (x + w) / width, (y + h) / height]
        annotations[ann["image_id"]].append({"box": box, "class_id": category_map[ann["category_id"]]})
    records = [{"id": f"{client_id}:{image['id']}", "client_id": client_id, "split": split,
                "rgb": str((image_root / image["file_name"]).resolve()),
                "labels": annotations[image["id"]]} for image in data["images"]]
    return records, {"class_names": [str(c["name"]) for c in categories],
                     "source_category_id_to_class_id": {str(k): v for k, v in category_map.items()}}


def dronevehicle_records(annotation_dir: Path, image_root: Path, client_id: str,
                         split: str, class_map_path: Path, image_suffix: str,
                         difficult_policy: str, *, ir_root: Path | None = None,
                         ir_suffix: str | None = None, class_aliases_path: Path | None = None):
    from PIL import Image
    class_map = read_class_map(class_map_path)
    aliases = read_class_aliases(class_aliases_path, class_map)
    files = _annotation_index(annotation_dir)
    images = _stem_index(image_root, image_suffix, "RGB")
    infrared = _stem_index(ir_root, ir_suffix or image_suffix, "IR") if ir_root is not None else None
    if not files:
        raise ValueError("No TXT or XML DroneVehicle annotation files found")
    records = []
    for key, file in files.items():
        if key not in images:
            raise FileNotFoundError(f"Missing RGB pair for annotation {file.stem}")
        image_path = images[key]
        with Image.open(image_path) as image:
            size = image.size
        labels = read_dronevehicle_labels(file, size, class_map, difficult_policy, aliases)
        row = {"id": f"{client_id}:{file.stem}", "client_id": client_id,
               "split": split, "rgb": str(image_path.resolve()), "labels": labels}
        if infrared is not None:
            if key not in infrared:
                raise FileNotFoundError(f"Missing IR pair for annotation {file.stem}")
            ir_path = infrared[key]
            read_ir(ir_path, size)
            row["nir"] = str(ir_path.resolve())
        records.append(row)
    metadata = {"class_names": [name for name, _ in sorted(class_map.items(), key=lambda x: x[1])],
                "oriented_conversion": "minimum enclosing horizontal box",
                "source_class_aliases": aliases,
                "difficult_policy": difficult_policy}
    if infrared is not None:
        metadata.update(modalities=["RGB", "IR"],
                        pairing_rule="annotation, RGB and IR filenames share a case-insensitive stem",
                        spatial_preprocessing="unchanged; matching image dimensions required")
    return records, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-format", choices=("auto", "json", "jsonl", "csv", "coco", "dronevehicle"), default="auto")
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--client-id")
    parser.add_argument("--split", choices=("labeled", "unlabeled", "validation", "test", "hard"))
    parser.add_argument("--class-map", type=Path, help="DroneVehicle category-name to zero-based integer mapping JSON")
    parser.add_argument("--class-aliases", type=Path, help="Source category aliases mapped to canonical class-map names")
    parser.add_argument("--image-suffix", default=".jpg")
    parser.add_argument("--ir-root", type=Path, help="DroneVehicle IR directory; writes paired IR paths to nir")
    parser.add_argument("--ir-suffix", help="IR extension; defaults to --image-suffix")
    parser.add_argument("--difficult-policy", choices=("error", "include", "drop"), default="error")
    parser.add_argument("--calibration-ids", type=Path, help="JSON list of existing validation IDs")
    parser.add_argument("--views", type=Path, help="JSON object mapping view names to held-out ID lists")
    parser.add_argument("--require-equal-standard-views", action="store_true")
    args = parser.parse_args(argv)
    if args.ir_suffix and args.ir_root is None:
        parser.error("--ir-suffix requires --ir-root")
    path = args.input.resolve()
    fmt = args.input_format
    if fmt == "auto":
        fmt = path.suffix.lstrip(".").lower()
    if args.ir_root is not None and fmt != "dronevehicle":
        parser.error("--ir-root is supported with --input-format dronevehicle")
    if args.class_aliases is not None and fmt != "dronevehicle":
        parser.error("--class-aliases is supported with --input-format dronevehicle")
    metadata = {}
    calibration_ids = []
    if fmt in ("json", "jsonl"):
        manifest = load_manifest(path, strict_files=args.check_files)
        records = manifest.records
        metadata = dict(manifest.metadata)
        calibration_ids = list(manifest.calibration_ids)
    elif fmt == "csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            records = []
            for row in csv.DictReader(stream):
                row = {key: value for key, value in row.items() if value != ""}
                if "labels" in row:
                    row["labels"] = json.loads(row["labels"])
                records.append(row)
    elif fmt in ("coco", "dronevehicle"):
        if args.image_root is None or args.client_id is None or args.split is None:
            parser.error("COCO/DroneVehicle conversion requires --image-root, --client-id and --split")
        if fmt == "coco":
            records, metadata = coco_records(path, args.image_root.resolve(), args.client_id, args.split)
        else:
            if args.class_map is None:
                parser.error("DroneVehicle conversion requires --class-map")
            records, metadata = dronevehicle_records(path, args.image_root.resolve(), args.client_id,
                args.split, args.class_map.resolve(), args.image_suffix, args.difficult_policy,
                ir_root=args.ir_root.resolve() if args.ir_root else None, ir_suffix=args.ir_suffix,
                class_aliases_path=args.class_aliases.resolve() if args.class_aliases else None)
    else:
        parser.error("Cannot infer input format; provide --input-format")
    if args.calibration_ids:
        calibration_ids = json.loads(args.calibration_ids.read_text(encoding="utf-8-sig"))
        if not isinstance(calibration_ids, list):
            raise ValueError("calibration-ids file must contain a JSON list")
    if args.views:
        metadata["views"] = json.loads(args.views.read_text(encoding="utf-8-sig"))
    if args.require_equal_standard_views:
        metadata["require_equal_standard_views"] = True
    manifest = validate_manifest(records, calibration_ids=calibration_ids, strict_files=args.check_files,
                                 base_dir=path.parent, metadata=metadata)
    out = args.output.resolve()
    sources = {path}
    sources.update(Path(value).resolve() for value in
                   (args.class_map, args.class_aliases, args.calibration_ids, args.views) if value is not None)
    sources.update(asset for row in manifest.records for asset in (row.rgb, row.depth, row.nir) if asset is not None)
    if fmt == "dronevehicle":
        sources.update(file.resolve() for file in path.iterdir()
                       if file.is_file() and file.suffix.casefold() in {".txt", ".xml"})
    if out in sources:
        raise ValueError("Output must not replace an input or sensor file")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Validate before replacing the output atomically.
    handle, temp_name = tempfile.mkstemp(prefix=out.name + ".", suffix=".tmp", dir=out.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest.as_dict(), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temp_name, out)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    counts = defaultdict(int)
    for row in manifest.records:
        counts[f"{row.client_id}/{row.split}"] += 1
    print(json.dumps({"output": str(out), "records": len(manifest.records),
                      "calibration_id_count": len(manifest.calibration_ids), "counts": dict(counts)}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, TypeError) as exc:
        raise SystemExit(f"Manifest preparation failed: {exc}")
