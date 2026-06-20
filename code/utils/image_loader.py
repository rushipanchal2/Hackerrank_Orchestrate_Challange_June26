"""Image loading and base64 encoding for vision calls."""

import base64
import io
import pathlib

from PIL import Image

# 768px long-edge is sufficient for damage detection; produces ~75% smaller
# base64 payloads than the original 1568px limit, saving tokens and latency.
_MAX_EDGE = 768
_JPEG_QUALITY = 72   # visually fine for damage classification; further reduces size


def encode_images(
    csv_paths: list[str],
    images_base_dir: str | pathlib.Path,
) -> list[dict]:
    """Resolve and base64-encode images with compression.

    Returns list of {image_id, base64_str, path, exists, size_kb}.
    """
    base = pathlib.Path(images_base_dir)
    result = []

    for csv_path in csv_paths:
        csv_path = csv_path.strip()
        if not csv_path:
            continue

        image_id = pathlib.Path(csv_path).stem
        resolved = (base / csv_path).resolve()

        if not resolved.exists():
            result.append({
                "image_id": image_id,
                "base64_str": "",
                "path": str(resolved),
                "exists": False,
            })
            continue

        try:
            img = Image.open(resolved).convert("RGB")
            if max(img.size) > _MAX_EDGE:
                img.thumbnail((_MAX_EDGE, _MAX_EDGE), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
            raw = buf.getvalue()
            b64 = base64.b64encode(raw).decode("utf-8")
            result.append({
                "image_id": image_id,
                "base64_str": b64,
                "path": str(resolved),
                "exists": True,
                "size_kb": round(len(raw) / 1024, 1),
            })
        except Exception as exc:
            result.append({
                "image_id": image_id,
                "base64_str": "",
                "path": str(resolved),
                "exists": False,
                "error": str(exc),
            })

    return result
