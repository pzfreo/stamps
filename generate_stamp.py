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
    binary = np.fliplr(binary)  # mirror for stamp printing
    return binary


def generate_stamp_stl(binary, pixel_size_mm, margin_mm, base_height_mm, text_height_mm):
    margin_px = int(margin_mm / pixel_size_mm)
    total_rows = binary.shape[0] + 2 * margin_px
    total_cols = binary.shape[1] + 2 * margin_px

    cells = np.zeros((total_rows, total_cols), dtype=bool)
    cells[margin_px:margin_px + binary.shape[0],
          margin_px:margin_px + binary.shape[1]] = binary

    total_h = base_height_mm + text_height_mm
    ps = pixel_size_mm
    tris = []

    def add_quad(v0, v1, v2, v3):
        tris.append([v0, v1, v2])
        tris.append([v0, v2, v3])

    for r in range(total_rows):
        for c in range(total_cols):
            x0, x1 = c * ps, (c + 1) * ps
            y0 = (total_rows - 1 - r) * ps
            y1 = (total_rows - r) * ps
            z = total_h if cells[r, c] else base_height_mm
            add_quad([x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z])
            add_quad([x0, y0, 0], [x0, y1, 0], [x1, y1, 0], [x1, y0, 0])

    max_x = total_cols * ps
    max_y = total_rows * ps

    for c in range(total_cols):
        x0, x1 = c * ps, (c + 1) * ps
        z = total_h if cells[total_rows - 1, c] else base_height_mm
        add_quad([x0, 0, 0], [x1, 0, 0], [x1, 0, z], [x0, 0, z])

    for c in range(total_cols):
        x0, x1 = c * ps, (c + 1) * ps
        z = total_h if cells[0, c] else base_height_mm
        add_quad([x0, max_y, 0], [x0, max_y, z], [x1, max_y, z], [x1, max_y, 0])

    for r in range(total_rows):
        y0 = (total_rows - 1 - r) * ps
        y1 = (total_rows - r) * ps
        z = total_h if cells[r, 0] else base_height_mm
        add_quad([0, y0, 0], [0, y0, z], [0, y1, z], [0, y1, 0])

    for r in range(total_rows):
        y0 = (total_rows - 1 - r) * ps
        y1 = (total_rows - r) * ps
        z = total_h if cells[r, total_cols - 1] else base_height_mm
        add_quad([max_x, y0, 0], [max_x, y1, 0], [max_x, y1, z], [max_x, y0, z])

    for r in range(total_rows - 1):
        for c in range(total_cols):
            h_top = total_h if cells[r, c] else base_height_mm
            h_bot = total_h if cells[r + 1, c] else base_height_mm
            if h_top != h_bot:
                x0, x1 = c * ps, (c + 1) * ps
                y = (total_rows - 1 - r) * ps
                z_lo, z_hi = min(h_top, h_bot), max(h_top, h_bot)
                if h_top > h_bot:
                    add_quad([x0, y, z_lo], [x1, y, z_lo], [x1, y, z_hi], [x0, y, z_hi])
                else:
                    add_quad([x0, y, z_lo], [x0, y, z_hi], [x1, y, z_hi], [x1, y, z_lo])

    for r in range(total_rows):
        for c in range(total_cols - 1):
            h_left = total_h if cells[r, c] else base_height_mm
            h_right = total_h if cells[r, c + 1] else base_height_mm
            if h_left != h_right:
                x = (c + 1) * ps
                y0 = (total_rows - 1 - r) * ps
                y1 = (total_rows - r) * ps
                z_lo, z_hi = min(h_left, h_right), max(h_left, h_right)
                if h_left > h_right:
                    add_quad([x, y0, z_lo], [x, y0, z_hi], [x, y1, z_hi], [x, y1, z_lo])
                else:
                    add_quad([x, y0, z_lo], [x, y1, z_lo], [x, y1, z_hi], [x, y0, z_hi])

    tris_arr = np.array(tris, dtype=np.float32)
    m = stlmesh.Mesh(np.zeros(len(tris_arr), dtype=stlmesh.Mesh.dtype))
    m.vectors = tris_arr
    return m, total_cols * ps, total_rows * ps


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.pdf|png> [output.stl] [width_mm]")
        sys.exit(1)

    input_path = sys.argv[1]
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = sys.argv[2] if len(sys.argv) > 2 else f"{base_name}_stamp.stl"
    target_width_mm = float(sys.argv[3]) if len(sys.argv) > 3 else 43.5

    pixel_size_mm = 0.2
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
