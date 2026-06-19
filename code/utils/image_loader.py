"""Image loading and base64 encoding for Claude vision calls."""

import base64
import io
import pathlib

from PIL import Image

# Anthropic resizes any image to a long edge of ~1568px / ~1.15MP server-side before the
# model sees it, so downsizing to this here is lossless in terms of model input — it only
# avoids the API's 5MB-per-image 400 error on very large source files.
_MAX_EDGE = 1568


def encode_images(
    csv_paths: list[str],
    images_base_dir: str | pathlib.Path,
) -> list[dict]:
    """
    Resolve and base64-encode images.

    csv_paths  : path strings exactly as they appear in the CSV
                 (e.g. 'images/test/case_001/img_1.jpg')
    images_base_dir : root directory to prepend
                 (e.g. 'dataset') so the resolved path becomes
                 'dataset/images/test/case_001/img_1.jpg'

    Returns a list of dicts:
      {image_id, base64_str, path (absolute str), exists (bool)}
    """
    base = pathlib.Path(images_base_dir)
    result = []

    for csv_path in csv_paths:
        csv_path = csv_path.strip()
        if not csv_path:
            continue

        image_id = pathlib.Path(csv_path).stem  # e.g. 'img_1'
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
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            result.append({
                "image_id": image_id,
                "base64_str": b64,
                "path": str(resolved),
                "exists": True,
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
