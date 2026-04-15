#!/usr/bin/env python3
"""Convert a PDF stamp design to a mirrored 3D-printable STL for FDM printing."""

import sys
import os
import numpy as np
from PIL import Image

try:
    import pymupdf
except ImportError:
    pymupdf = None

from stl import mesh as stlmesh


def pdf_to_png(pdf_path, dpi=600):
    if pymupdf is None:
        raise ImportError("pymupdf required for PDF input: pip install pymupdf")
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    tmp = "/tmp/_stamp_render.png"
    pix.save(tmp)
    return tmp


def load_and_prepare(image_path, target_width_mm, pixel_size_mm, threshold=128):
    img = Image.open(image_path).convert("L")
    arr = np.array(img)

    dark = arr < threshold
    rows = np.any(dark, axis=1)
    cols = np.any(dark, axis=0)
    if not rows.any():
        raise ValueError("No dark content found in image")
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    pad = 10
    cropped = arr[max(0, rmin - pad):rmax + pad + 1, max(0, cmin - pad):cmax + pad + 1]

    target_width_px = int(target_width_mm / pixel_size_mm)
    aspect = cropped.shape[0] / cropped.shape[1]
    target_height_px = int(target_width_px * aspect)

    img_resized = Image.fromarray(cropped).resize(
        (target_width_px, target_height_px), Image.LANCZOS
    )
    binary = (np.array(img_resized) < threshold)
    binary = np.fliplr(binary)
    return binary


def generate_stamp_stl(binary, pixel_size_mm, margin_mm, base_height_mm, text_height_mm):
    margin_px = int(margin_mm / pixel_size_mm)
    total_rows = binary.shape[0] + 2 * margin_px
    total_cols = binary.shape[1] + 2 * margin_px

    cells = np.zeros((total_rows, total_cols), dtype=bool)
    cells[margin_px:margin_px + binary.shape[0],
          margin_px:margin_px + binary.shape[1]] = binary

    total_h = np.float32(base_height_mm + text_height_mm)
    base_h = np.float32(base_height_mm)
    ps = pixel_size_mm

    cell_r, cell_c = np.mgrid[0:total_rows, 0:total_cols]
    x0 = (cell_c * ps).ravel().astype(np.float32)
    x1 = ((cell_c + 1) * ps).ravel().astype(np.float32)
    y0 = ((total_rows - 1 - cell_r) * ps).ravel().astype(np.float32)
    y1 = ((total_rows - cell_r) * ps).ravel().astype(np.float32)
    z = np.where(cells.ravel(), total_h, base_h).astype(np.float32)
    z0 = np.zeros_like(z)

    all_tris = []

    # Top faces
    all_tris.append(np.stack([
        np.stack([x0, y0, z], axis=1),
        np.stack([x1, y0, z], axis=1),
        np.stack([x1, y1, z], axis=1),
    ], axis=1))
    all_tris.append(np.stack([
        np.stack([x0, y0, z], axis=1),
        np.stack([x1, y1, z], axis=1),
        np.stack([x0, y1, z], axis=1),
    ], axis=1))

    # Bottom faces
    all_tris.append(np.stack([
        np.stack([x0, y0, z0], axis=1),
        np.stack([x0, y1, z0], axis=1),
        np.stack([x1, y1, z0], axis=1),
    ], axis=1))
    all_tris.append(np.stack([
        np.stack([x0, y0, z0], axis=1),
        np.stack([x1, y1, z0], axis=1),
        np.stack([x1, y0, z0], axis=1),
    ], axis=1))

    def add_horizontal_walls(r):
        diff = cells[r, :] != cells[r + 1, :]
        if not diff.any():
            return
        ci = np.where(diff)[0]
        cx0 = (ci * ps).astype(np.float32)
        cx1 = ((ci + 1) * ps).astype(np.float32)
        cy = np.full(len(ci), (total_rows - 1 - r) * ps, dtype=np.float32)
        ht = np.where(cells[r, ci], total_h, base_h).astype(np.float32)
        hb = np.where(cells[r + 1, ci], total_h, base_h).astype(np.float32)
        zlo, zhi = np.minimum(ht, hb), np.maximum(ht, hb)

        top_hi = ht > hb
        if top_hi.any():
            i = top_hi
            all_tris.append(np.stack([np.stack([cx0[i], cy[i], zlo[i]], 1), np.stack([cx1[i], cy[i], zlo[i]], 1), np.stack([cx1[i], cy[i], zhi[i]], 1)], 1))
            all_tris.append(np.stack([np.stack([cx0[i], cy[i], zlo[i]], 1), np.stack([cx1[i], cy[i], zhi[i]], 1), np.stack([cx0[i], cy[i], zhi[i]], 1)], 1))
        bot_hi = ~top_hi
        if bot_hi.any():
            i = bot_hi
            all_tris.append(np.stack([np.stack([cx0[i], cy[i], zlo[i]], 1), np.stack([cx0[i], cy[i], zhi[i]], 1), np.stack([cx1[i], cy[i], zhi[i]], 1)], 1))
            all_tris.append(np.stack([np.stack([cx0[i], cy[i], zlo[i]], 1), np.stack([cx1[i], cy[i], zhi[i]], 1), np.stack([cx1[i], cy[i], zlo[i]], 1)], 1))

    def add_vertical_walls(c):
        diff = cells[:, c] != cells[:, c + 1]
        if not diff.any():
            return
        ri = np.where(diff)[0]
        cx = np.full(len(ri), (c + 1) * ps, dtype=np.float32)
        ry0 = ((total_rows - 1 - ri) * ps).astype(np.float32)
        ry1 = ((total_rows - ri) * ps).astype(np.float32)
        hl = np.where(cells[ri, c], total_h, base_h).astype(np.float32)
        hr = np.where(cells[ri, c + 1], total_h, base_h).astype(np.float32)
        zlo, zhi = np.minimum(hl, hr), np.maximum(hl, hr)

        left_hi = hl > hr
        if left_hi.any():
            i = left_hi
            all_tris.append(np.stack([np.stack([cx[i], ry0[i], zlo[i]], 1), np.stack([cx[i], ry0[i], zhi[i]], 1), np.stack([cx[i], ry1[i], zhi[i]], 1)], 1))
            all_tris.append(np.stack([np.stack([cx[i], ry0[i], zlo[i]], 1), np.stack([cx[i], ry1[i], zhi[i]], 1), np.stack([cx[i], ry1[i], zlo[i]], 1)], 1))
        right_hi = ~left_hi
        if right_hi.any():
            i = right_hi
            all_tris.append(np.stack([np.stack([cx[i], ry0[i], zlo[i]], 1), np.stack([cx[i], ry1[i], zlo[i]], 1), np.stack([cx[i], ry1[i], zhi[i]], 1)], 1))
            all_tris.append(np.stack([np.stack([cx[i], ry0[i], zlo[i]], 1), np.stack([cx[i], ry1[i], zhi[i]], 1), np.stack([cx[i], ry0[i], zhi[i]], 1)], 1))

    for r in range(total_rows - 1):
        add_horizontal_walls(r)
    for c in range(total_cols - 1):
        add_vertical_walls(c)

    max_x = np.float32(total_cols * ps)
    max_y = np.float32(total_rows * ps)

    # Outer walls (vectorized per edge)
    def outer_wall_x(row_indices, y_val, flip):
        zz = np.where(cells[row_indices, :].ravel(), total_h, base_h) if len(row_indices) > 1 else np.where(cells[row_indices[0], :], total_h, base_h)
        # For single-row edges
        r_idx = row_indices[0] if len(row_indices) == 1 else None
        if r_idx is not None:
            zz = np.where(cells[r_idx, :], total_h, base_h).astype(np.float32)
            xx0 = (np.arange(total_cols) * ps).astype(np.float32)
            xx1 = ((np.arange(total_cols) + 1) * ps).astype(np.float32)
            yy = np.full(total_cols, y_val, dtype=np.float32)
            z_zero = np.zeros(total_cols, dtype=np.float32)
            if flip:
                all_tris.append(np.stack([np.stack([xx0, yy, z_zero], 1), np.stack([xx0, yy, zz], 1), np.stack([xx1, yy, zz], 1)], 1))
                all_tris.append(np.stack([np.stack([xx0, yy, z_zero], 1), np.stack([xx1, yy, zz], 1), np.stack([xx1, yy, z_zero], 1)], 1))
            else:
                all_tris.append(np.stack([np.stack([xx0, yy, z_zero], 1), np.stack([xx1, yy, z_zero], 1), np.stack([xx1, yy, zz], 1)], 1))
                all_tris.append(np.stack([np.stack([xx0, yy, z_zero], 1), np.stack([xx1, yy, zz], 1), np.stack([xx0, yy, zz], 1)], 1))

    # Front (y=0)
    outer_wall_x([total_rows - 1], np.float32(0), False)
    # Back (y=max_y)
    outer_wall_x([0], max_y, True)

    # Left (x=0)
    zz = np.where(cells[:, 0], total_h, base_h).astype(np.float32)
    yy0 = ((total_rows - 1 - np.arange(total_rows)) * ps).astype(np.float32)
    yy1 = ((total_rows - np.arange(total_rows)) * ps).astype(np.float32)
    xx = np.zeros(total_rows, dtype=np.float32)
    z_zero = np.zeros(total_rows, dtype=np.float32)
    all_tris.append(np.stack([np.stack([xx, yy0, z_zero], 1), np.stack([xx, yy0, zz], 1), np.stack([xx, yy1, zz], 1)], 1))
    all_tris.append(np.stack([np.stack([xx, yy0, z_zero], 1), np.stack([xx, yy1, zz], 1), np.stack([xx, yy1, z_zero], 1)], 1))

    # Right (x=max_x)
    zz = np.where(cells[:, total_cols - 1], total_h, base_h).astype(np.float32)
    xx = np.full(total_rows, max_x, dtype=np.float32)
    all_tris.append(np.stack([np.stack([xx, yy0, z_zero], 1), np.stack([xx, yy1, z_zero], 1), np.stack([xx, yy1, zz], 1)], 1))
    all_tris.append(np.stack([np.stack([xx, yy0, z_zero], 1), np.stack([xx, yy1, zz], 1), np.stack([xx, yy0, zz], 1)], 1))

    combined = np.concatenate(all_tris, axis=0)
    m = stlmesh.Mesh(np.zeros(len(combined), dtype=stlmesh.Mesh.dtype))
    m.vectors = combined
    return m, total_cols * ps, total_rows * ps


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.pdf|png> [output.stl] [width_mm]")
        sys.exit(1)

    input_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"{base_name}_stamp.stl"
    target_width_mm = float(sys.argv[3]) if len(sys.argv) > 3 else 43.5

    pixel_size_mm = 0.05
    margin_mm = 2.0
    base_height_mm = 3.0
    text_height_mm = 2.0

    if input_path.lower().endswith(".pdf"):
        image_path = pdf_to_png(input_path)
    else:
        image_path = input_path

    print(f"Target printable width: {target_width_mm} mm")
    print(f"Resolution: {pixel_size_mm} mm/pixel")

    binary = load_and_prepare(image_path, target_width_mm, pixel_size_mm)
    print(f"Text grid: {binary.shape[1]} x {binary.shape[0]} pixels")

    m, w, h = generate_stamp_stl(binary, pixel_size_mm, margin_mm, base_height_mm, text_height_mm)
    m.save(output_path)

    dims = m.vectors.reshape(-1, 3).max(axis=0) - m.vectors.reshape(-1, 3).min(axis=0)
    print(f"STL dimensions: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm")
    print(f"Saved: {output_path} ({os.path.getsize(output_path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
