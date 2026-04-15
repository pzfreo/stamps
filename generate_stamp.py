#!/usr/bin/env python3
"""Convert a PDF stamp design to a mirrored 3D-printable STL for FDM printing.

Generates a stamp with pyramid/chamfer support around raised text for
better printability on FDM printers. The slope is quantized to the
printer's layer height for clean stepping.
"""

import sys
import os
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

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


def build_heightmap(cells, base_h, text_h, pixel_size_mm, slope_angle_deg, layer_height):
    total_rows, total_cols = cells.shape
    total_h = base_h + text_h

    dist = distance_transform_edt(~cells) * pixel_size_mm
    slope_distance = text_h / np.tan(np.radians(slope_angle_deg))

    heightmap = np.full((total_rows, total_cols), base_h, dtype=np.float32)
    heightmap[cells] = total_h

    slope_mask = (~cells) & (dist < slope_distance)
    heightmap[slope_mask] = (base_h + text_h * (1.0 - dist[slope_mask] / slope_distance)).astype(np.float32)

    heightmap = (np.round(heightmap / layer_height) * layer_height).astype(np.float32)
    return heightmap


def generate_stamp_stl(binary, pixel_size_mm, margin_mm, base_height_mm,
                       text_height_mm, slope_angle_deg=45, layer_height=0.08):
    margin_px = int(margin_mm / pixel_size_mm)
    total_rows = binary.shape[0] + 2 * margin_px
    total_cols = binary.shape[1] + 2 * margin_px

    cells = np.zeros((total_rows, total_cols), dtype=bool)
    cells[margin_px:margin_px + binary.shape[0],
          margin_px:margin_px + binary.shape[1]] = binary

    heightmap = build_heightmap(cells, base_height_mm, text_height_mm,
                                pixel_size_mm, slope_angle_deg, layer_height)

    ps = pixel_size_mm
    cell_r, cell_c = np.mgrid[0:total_rows, 0:total_cols]
    x0 = (cell_c * ps).ravel().astype(np.float32)
    x1 = ((cell_c + 1) * ps).ravel().astype(np.float32)
    y0 = ((total_rows - 1 - cell_r) * ps).ravel().astype(np.float32)
    y1 = ((total_rows - cell_r) * ps).ravel().astype(np.float32)
    z = heightmap.ravel()
    z0 = np.zeros_like(z)

    all_tris = []

    all_tris.append(np.stack([np.stack([x0,y0,z],1), np.stack([x1,y0,z],1), np.stack([x1,y1,z],1)], 1))
    all_tris.append(np.stack([np.stack([x0,y0,z],1), np.stack([x1,y1,z],1), np.stack([x0,y1,z],1)], 1))
    all_tris.append(np.stack([np.stack([x0,y0,z0],1), np.stack([x0,y1,z0],1), np.stack([x1,y1,z0],1)], 1))
    all_tris.append(np.stack([np.stack([x0,y0,z0],1), np.stack([x1,y1,z0],1), np.stack([x1,y0,z0],1)], 1))

    for r in range(total_rows - 1):
        diff = heightmap[r, :] != heightmap[r+1, :]
        if not diff.any():
            continue
        ci = np.where(diff)[0]
        cx0 = (ci * ps).astype(np.float32)
        cx1 = ((ci + 1) * ps).astype(np.float32)
        cy = np.full(len(ci), (total_rows - 1 - r) * ps, dtype=np.float32)
        ht, hb = heightmap[r, ci], heightmap[r+1, ci]
        zlo, zhi = np.minimum(ht, hb), np.maximum(ht, hb)
        top_hi = ht > hb
        if top_hi.any():
            i = top_hi
            all_tris.append(np.stack([np.stack([cx0[i],cy[i],zlo[i]],1), np.stack([cx1[i],cy[i],zlo[i]],1), np.stack([cx1[i],cy[i],zhi[i]],1)],1))
            all_tris.append(np.stack([np.stack([cx0[i],cy[i],zlo[i]],1), np.stack([cx1[i],cy[i],zhi[i]],1), np.stack([cx0[i],cy[i],zhi[i]],1)],1))
        bot_hi = ~top_hi
        if bot_hi.any():
            i = bot_hi
            all_tris.append(np.stack([np.stack([cx0[i],cy[i],zlo[i]],1), np.stack([cx0[i],cy[i],zhi[i]],1), np.stack([cx1[i],cy[i],zhi[i]],1)],1))
            all_tris.append(np.stack([np.stack([cx0[i],cy[i],zlo[i]],1), np.stack([cx1[i],cy[i],zhi[i]],1), np.stack([cx1[i],cy[i],zlo[i]],1)],1))

    for c in range(total_cols - 1):
        diff = heightmap[:, c] != heightmap[:, c+1]
        if not diff.any():
            continue
        ri = np.where(diff)[0]
        cx = np.full(len(ri), (c + 1) * ps, dtype=np.float32)
        ry0 = ((total_rows - 1 - ri) * ps).astype(np.float32)
        ry1 = ((total_rows - ri) * ps).astype(np.float32)
        hl, hr = heightmap[ri, c], heightmap[ri, c+1]
        zlo, zhi = np.minimum(hl, hr), np.maximum(hl, hr)
        left_hi = hl > hr
        if left_hi.any():
            i = left_hi
            all_tris.append(np.stack([np.stack([cx[i],ry0[i],zlo[i]],1), np.stack([cx[i],ry0[i],zhi[i]],1), np.stack([cx[i],ry1[i],zhi[i]],1)],1))
            all_tris.append(np.stack([np.stack([cx[i],ry0[i],zlo[i]],1), np.stack([cx[i],ry1[i],zhi[i]],1), np.stack([cx[i],ry1[i],zlo[i]],1)],1))
        right_hi = ~left_hi
        if right_hi.any():
            i = right_hi
            all_tris.append(np.stack([np.stack([cx[i],ry0[i],zlo[i]],1), np.stack([cx[i],ry1[i],zlo[i]],1), np.stack([cx[i],ry1[i],zhi[i]],1)],1))
            all_tris.append(np.stack([np.stack([cx[i],ry0[i],zlo[i]],1), np.stack([cx[i],ry1[i],zhi[i]],1), np.stack([cx[i],ry0[i],zhi[i]],1)],1))

    max_x = np.float32(total_cols * ps)
    max_y = np.float32(total_rows * ps)

    zz = heightmap[total_rows-1, :].astype(np.float32)
    xx0 = (np.arange(total_cols) * ps).astype(np.float32)
    xx1 = ((np.arange(total_cols) + 1) * ps).astype(np.float32)
    yy = np.zeros(total_cols, dtype=np.float32)
    zz0 = np.zeros(total_cols, dtype=np.float32)
    all_tris.append(np.stack([np.stack([xx0,yy,zz0],1), np.stack([xx1,yy,zz0],1), np.stack([xx1,yy,zz],1)],1))
    all_tris.append(np.stack([np.stack([xx0,yy,zz0],1), np.stack([xx1,yy,zz],1), np.stack([xx0,yy,zz],1)],1))

    zz = heightmap[0, :].astype(np.float32)
    yy = np.full(total_cols, max_y, dtype=np.float32)
    all_tris.append(np.stack([np.stack([xx0,yy,zz0],1), np.stack([xx0,yy,zz],1), np.stack([xx1,yy,zz],1)],1))
    all_tris.append(np.stack([np.stack([xx0,yy,zz0],1), np.stack([xx1,yy,zz],1), np.stack([xx1,yy,zz0],1)],1))

    zz = heightmap[:, 0].astype(np.float32)
    yy0 = ((total_rows - 1 - np.arange(total_rows)) * ps).astype(np.float32)
    yy1 = ((total_rows - np.arange(total_rows)) * ps).astype(np.float32)
    xx = np.zeros(total_rows, dtype=np.float32)
    zz0r = np.zeros(total_rows, dtype=np.float32)
    all_tris.append(np.stack([np.stack([xx,yy0,zz0r],1), np.stack([xx,yy0,zz],1), np.stack([xx,yy1,zz],1)],1))
    all_tris.append(np.stack([np.stack([xx,yy0,zz0r],1), np.stack([xx,yy1,zz],1), np.stack([xx,yy1,zz0r],1)],1))

    zz = heightmap[:, total_cols-1].astype(np.float32)
    xx = np.full(total_rows, max_x, dtype=np.float32)
    all_tris.append(np.stack([np.stack([xx,yy0,zz0r],1), np.stack([xx,yy1,zz0r],1), np.stack([xx,yy1,zz],1)],1))
    all_tris.append(np.stack([np.stack([xx,yy0,zz0r],1), np.stack([xx,yy1,zz],1), np.stack([xx,yy0,zz],1)],1))

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
    slope_angle_deg = 45
    layer_height = 0.08

    if input_path.lower().endswith(".pdf"):
        image_path = pdf_to_png(input_path)
    else:
        image_path = input_path

    print(f"Target printable width: {target_width_mm} mm")
    print(f"Resolution: {pixel_size_mm} mm/pixel")
    print(f"Pyramid slope: {slope_angle_deg}°, layer height: {layer_height} mm")

    binary = load_and_prepare(image_path, target_width_mm, pixel_size_mm)
    print(f"Text grid: {binary.shape[1]} x {binary.shape[0]} pixels")

    m, w, h = generate_stamp_stl(binary, pixel_size_mm, margin_mm, base_height_mm,
                                  text_height_mm, slope_angle_deg, layer_height)
    m.save(output_path)

    dims = m.vectors.reshape(-1, 3).max(axis=0) - m.vectors.reshape(-1, 3).min(axis=0)
    print(f"STL dimensions: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm")
    print(f"Saved: {output_path} ({os.path.getsize(output_path) / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
