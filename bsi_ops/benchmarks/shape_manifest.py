FIXED76_SHAPE_MANIFEST = [
    (512, 768, 768),
    (512, 3072, 768),
    (512, 768, 3072),
    (512, 2048, 2048),
    (512, 8192, 2048),
    (512, 2048, 8192),
    (512, 4096, 4096),
    (512, 16384, 4096),
    (512, 4096, 16384),
    (512, 32768, 4096),
    (512, 8192, 8192),
]


def resolve_shape_manifest(name: str):
    key = (name or "").strip().lower()
    if key in ("", "none"):
        return None
    if key == "fixed76":
        return FIXED76_SHAPE_MANIFEST
    raise ValueError(f"Unsupported shape manifest: {name}")
