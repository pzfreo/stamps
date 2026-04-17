#!/usr/bin/env python3
"""Convert a PDF, PNG, or SVG stamp design to a mirrored 3D-printable STL for FDM printing.

Generates a stamp with pyramid/chamfer support around raised text for
better printability on FDM printers. The slope is quantized to the
printer's layer height for clean stepping.

SVG input uses a vector pipeline (no rasterization) for smooth curves
and smaller file sizes. PDF/PNG input uses the raster pipeline.
"""

import argparse
import sys
import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt

try:
    import pymupdf
except ImportError:
    pymupdf = None

try:
    import cairosvg
except ImportError:
    cairosvg = None

from stl import mesh as stlmesh


# ---------------------------------------------------------------------------
# Raster pipeline helpers (PDF / PNG)
# ---------------------------------------------------------------------------

def pdf_to_png(pdf_path, dpi=600):
    if pymupdf is None:
        raise ImportError("pymupdf required for PDF input: pip install pymupdf")
    doc = pymupdf.open(pdf_path)
    page = doc[0]
    pix = page.get_pixmap(dpi=dpi)
    tmp = "/tmp/_stamp_render.png"
    pix.save(tmp)
    return tmp


def load_and_prepare(image_path, target_width_mm, pixel_size_mm, threshold=128,
                     erode_mm=0.0):
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

    if erode_mm > 0:
        erode_px = erode_mm / pixel_size_mm
        dist = distance_transform_edt(binary)
        binary = dist > erode_px

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

    unique_z = np.sort(np.unique(heightmap))
    for k in range(len(unique_z) - 1):
        z_bot = np.float32(unique_z[k])
        z_top = np.float32(unique_z[k + 1])
        filled = heightmap > z_bot

        row_diff = filled[:-1, :] != filled[1:, :]
        if row_diff.any():
            ri, ci = np.where(row_diff)
            cx0 = (ci * ps).astype(np.float32)
            cx1 = ((ci + 1) * ps).astype(np.float32)
            cy = ((total_rows - 1 - ri) * ps).astype(np.float32)
            z_lo = np.full(len(ri), z_bot, dtype=np.float32)
            z_hi = np.full(len(ri), z_top, dtype=np.float32)
            top_filled = filled[ri, ci]
            if top_filled.any():
                i = top_filled
                all_tris.append(np.stack([np.stack([cx0[i],cy[i],z_lo[i]],1), np.stack([cx1[i],cy[i],z_lo[i]],1), np.stack([cx1[i],cy[i],z_hi[i]],1)],1))
                all_tris.append(np.stack([np.stack([cx0[i],cy[i],z_lo[i]],1), np.stack([cx1[i],cy[i],z_hi[i]],1), np.stack([cx0[i],cy[i],z_hi[i]],1)],1))
            bot_filled = ~top_filled
            if bot_filled.any():
                i = bot_filled
                all_tris.append(np.stack([np.stack([cx0[i],cy[i],z_lo[i]],1), np.stack([cx0[i],cy[i],z_hi[i]],1), np.stack([cx1[i],cy[i],z_hi[i]],1)],1))
                all_tris.append(np.stack([np.stack([cx0[i],cy[i],z_lo[i]],1), np.stack([cx1[i],cy[i],z_hi[i]],1), np.stack([cx1[i],cy[i],z_lo[i]],1)],1))

        col_diff = filled[:, :-1] != filled[:, 1:]
        if col_diff.any():
            ri, ci = np.where(col_diff)
            cx = ((ci + 1) * ps).astype(np.float32)
            ry0 = ((total_rows - 1 - ri) * ps).astype(np.float32)
            ry1 = ((total_rows - ri) * ps).astype(np.float32)
            z_lo = np.full(len(ri), z_bot, dtype=np.float32)
            z_hi = np.full(len(ri), z_top, dtype=np.float32)
            left_filled = filled[ri, ci]
            if left_filled.any():
                i = left_filled
                all_tris.append(np.stack([np.stack([cx[i],ry0[i],z_lo[i]],1), np.stack([cx[i],ry0[i],z_hi[i]],1), np.stack([cx[i],ry1[i],z_hi[i]],1)],1))
                all_tris.append(np.stack([np.stack([cx[i],ry0[i],z_lo[i]],1), np.stack([cx[i],ry1[i],z_hi[i]],1), np.stack([cx[i],ry1[i],z_lo[i]],1)],1))
            right_filled = ~left_filled
            if right_filled.any():
                i = right_filled
                all_tris.append(np.stack([np.stack([cx[i],ry0[i],z_lo[i]],1), np.stack([cx[i],ry1[i],z_lo[i]],1), np.stack([cx[i],ry1[i],z_hi[i]],1)],1))
                all_tris.append(np.stack([np.stack([cx[i],ry0[i],z_lo[i]],1), np.stack([cx[i],ry1[i],z_hi[i]],1), np.stack([cx[i],ry0[i],z_hi[i]],1)],1))

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


# ---------------------------------------------------------------------------
# Vector pipeline helpers (SVG)
# ---------------------------------------------------------------------------

def svg_to_polygons(svg_path, target_width_mm):
    """Parse SVG vector paths into scaled, mirrored shapely polygons."""
    from svgpathtools import parse_path
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    from shapely import affinity

    tree = ET.parse(svg_path)
    root = tree.getroot()
    ns = '{http://www.w3.org/2000/svg}'

    def parse_matrix(s):
        if not s or 'matrix(' not in s:
            return None
        inner = s[s.index('matrix(') + 7:s.index(')')]
        return [float(v) for v in inner.replace(',', ' ').split()]

    def apply_matrix(x, y, m):
        return m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]

    paths_data = []

    def collect(el, matrices):
        t = el.get('transform')
        m = parse_matrix(t)
        if m:
            matrices = matrices + [m]
        tag = el.tag.split('}')[-1] if '}' in el.tag else el.tag
        if tag == 'path':
            d = el.get('d')
            style = el.get('style', '')
            if d and 'fill:none' not in style:
                paths_data.append((d, list(matrices)))
        for child in el:
            collect(child, matrices)

    collect(root, [])

    if not paths_data:
        raise ValueError("No filled paths found in SVG")

    polygons = []
    for d_str, matrices in paths_data:
        path = parse_path(d_str)
        n_pts = max(200, int(path.length() * 5))
        points = []
        for i in range(n_pts):
            pt = path.point(i / n_pts)
            x, y = pt.real, pt.imag
            for m in matrices:
                x, y = apply_matrix(x, y, m)
            points.append((x, y))
        if len(points) < 3:
            continue
        try:
            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty and poly.area > 0.01:
                polygons.append(poly)
        except Exception:
            continue

    if not polygons:
        raise ValueError("No valid vector paths found in SVG")

    combined = unary_union(polygons)

    bx0, by0, bx1, by1 = combined.bounds
    combined = affinity.translate(combined, -bx0, -by0)
    scale = target_width_mm / (bx1 - bx0)
    combined = affinity.scale(combined, xfact=scale, yfact=scale, origin=(0, 0))

    w = combined.bounds[2]
    combined = affinity.scale(combined, xfact=-1, yfact=1, origin=(w / 2, 0))

    return combined


def _to_polygon_list(geom):
    """Normalize shapely geometry to a list of Polygons."""
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    if geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]
    return []


def _ensure_min_width(poly, min_width):
    """Fatten features thinner than min_width to be exactly min_width wide."""
    half_w = min_width / 2
    opened = poly.buffer(-half_w).buffer(half_w)
    thin_features = poly.difference(opened)
    if thin_features.is_empty:
        return poly
    thickened = thin_features.buffer(half_w)
    result = poly.union(thickened)
    if not result.is_valid:
        result = result.buffer(0)
    return result


def generate_stamp_stl_vector(text_poly, margin_mm, base_height_mm,
                               text_height_mm, slope_angle_deg, layer_height,
                               erode_mm=0.0, min_width_mm=0.0):
    """Build a watertight STL mesh directly from vector text polygons."""
    import trimesh
    from shapely.geometry import box, Polygon

    if min_width_mm > 0:
        text_poly = _ensure_min_width(text_poly, min_width_mm)

    if erode_mm > 0:
        text_poly = text_poly.buffer(-erode_mm)
        if text_poly.is_empty:
            raise ValueError("Erosion removed all text content")
        if min_width_mm > 0:
            text_poly = _ensure_min_width(text_poly, min_width_mm)

    slope_dist = text_height_mm / np.tan(np.radians(slope_angle_deg))
    total_h = base_height_mm + text_height_mm

    bx0, by0, bx1, by1 = text_poly.bounds
    rect = box(bx0 - margin_mm, by0 - margin_mm,
               bx1 + margin_mm, by1 + margin_mm)

    n_steps = max(1, round(text_height_mm / layer_height))
    heights = [base_height_mm + k * layer_height for k in range(n_steps + 1)]

    # contours[k] = region where height >= heights[k]
    # buffer distance decreases as height increases
    # Simplify polygon detail to ~half the layer height for efficiency
    simplify_tol = layer_height / 2

    contours = []
    for k in range(n_steps + 1):
        buf = slope_dist * (1.0 - k / n_steps)
        if buf > 1e-6:
            c = text_poly.buffer(buf, join_style=2).intersection(rect)
        else:
            c = text_poly
        c = c.simplify(simplify_tol, preserve_topology=True)
        contours.append(c)

    all_tris = []

    def triangulate_face(poly, z, flip=False):
        for p in _to_polygon_list(poly):
            try:
                v2, f = trimesh.creation.triangulate_polygon(p)
            except Exception:
                continue
            if len(f) == 0:
                continue
            if flip:
                f = f[:, ::-1]
            v3 = np.column_stack([v2, np.full(len(v2), z)])
            all_tris.append(v3[f].astype(np.float32))

    def add_walls(poly, z_bot, z_top):
        for p in _to_polygon_list(poly):
            for ring, exterior in [(p.exterior, True)] + \
                                  [(h, False) for h in p.interiors]:
                coords = list(ring.coords[:-1])
                n = len(coords)
                for i in range(n):
                    j = (i + 1) % n
                    x0, y0 = coords[i]
                    x1, y1 = coords[j]
                    a = [x0, y0, z_bot]
                    b = [x1, y1, z_bot]
                    c = [x1, y1, z_top]
                    d = [x0, y0, z_top]
                    if exterior:
                        all_tris.append(np.array([[a, b, c], [a, c, d]],
                                                 dtype=np.float32))
                    else:
                        all_tris.append(np.array([[a, c, b], [a, d, c]],
                                                 dtype=np.float32))

    # Bottom face (full rectangle at z=0, normals pointing down)
    triangulate_face(rect, 0.0, flip=True)

    # Top face at base height (outside the first slope contour)
    if n_steps > 0 and not contours[1].is_empty:
        base_top = rect.difference(contours[1])
    else:
        base_top = rect
    triangulate_face(base_top, base_height_mm)

    # Terrace ring top faces
    for k in range(1, n_steps):
        ring = contours[k].difference(contours[k + 1])
        triangulate_face(ring, heights[k])

    # Text face at full height
    triangulate_face(contours[n_steps], total_h)

    # Perimeter walls (z=0 to z=base_h)
    add_walls(rect, 0.0, base_height_mm)

    # Slope step walls (one layer_height each)
    for k in range(1, n_steps + 1):
        add_walls(contours[k], heights[k - 1], heights[k])

    combined = np.concatenate(all_tris, axis=0)
    m = stlmesh.Mesh(np.zeros(len(combined), dtype=stlmesh.Mesh.dtype))
    m.vectors = combined

    w = rect.bounds[2] - rect.bounds[0]
    h = rect.bounds[3] - rect.bounds[1]
    return m, w, h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Convert a PDF, PNG, or SVG stamp design to a mirrored "
                    "3D-printable STL for FDM printing."
    )
    parser.add_argument("input", help="input PDF, PNG, or SVG file")
    parser.add_argument("-o", "--output",
                        help="output STL file (default: <input>_stamp.stl)")
    parser.add_argument("-w", "--width", type=float, default=43.5,
                        help="target printable width in mm (default: 43.5)")
    parser.add_argument("--resolution", type=float, default=0.05,
                        help="pixel size in mm for raster mode (default: 0.05)")
    parser.add_argument("--margin", type=float, default=2.0,
                        help="border margin around text in mm (default: 2.0)")
    parser.add_argument("--base-height", type=float, default=3.0,
                        help="base/handle height in mm (default: 3.0)")
    parser.add_argument("--text-height", type=float, default=2.0,
                        help="raised text height in mm (default: 2.0)")
    parser.add_argument("--slope-angle", type=float, default=60,
                        help="pyramid slope angle in degrees (default: 60)")
    parser.add_argument("--layer-height", type=float, default=0.08,
                        help="layer height for slope quantization in mm "
                             "(default: 0.08)")
    parser.add_argument("--threshold", type=int, default=128,
                        help="grayscale threshold for raster mode, 0-255 "
                             "(default: 128)")
    parser.add_argument("--erode", type=float, default=0.0,
                        help="shrink text outlines by this many mm to "
                             "compensate for nozzle spread (default: 0, "
                             "try 0.2 for 0.4mm nozzle)")
    parser.add_argument("--dpi", type=int, default=600,
                        help="rasterization DPI for PDF input (default: 600)")
    parser.add_argument("--min-width", type=float, default=0.0,
                        help="fatten text features thinner than this to "
                             "ensure printability in mm (default: 0, "
                             "try 0.4 for 0.4mm nozzle)")
    parser.add_argument("--decimate", type=float, default=0,
                        help="reduce triangle count by this fraction (0-0.95),"
                             " e.g. 0.8 removes 80%% of triangles "
                             "(default: 0, off)")
    args = parser.parse_args()

    input_path = args.input
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = args.output or f"{base_name}_stamp.stl"
    target_width_mm = args.width
    margin_mm = args.margin
    base_height_mm = args.base_height
    text_height_mm = args.text_height
    slope_angle_deg = args.slope_angle
    layer_height = args.layer_height

    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".svg":
        # ---- Vector pipeline ----
        print(f"Vector pipeline: {input_path}")
        text_poly = svg_to_polygons(input_path, target_width_mm)
        bx0, by0, bx1, by1 = text_poly.bounds
        print(f"Content: {bx1-bx0:.1f} x {by1-by0:.1f} mm "
              f"({text_poly.geom_type})")
        print(f"Slope: {slope_angle_deg}°, layer height: {layer_height} mm")

        m, w, h = generate_stamp_stl_vector(
            text_poly, margin_mm, base_height_mm, text_height_mm,
            slope_angle_deg, layer_height, erode_mm=args.erode,
            min_width_mm=args.min_width)
    else:
        # ---- Raster pipeline ----
        pixel_size_mm = args.resolution
        if ext == ".pdf":
            image_path = pdf_to_png(input_path, dpi=args.dpi)
        else:
            image_path = input_path

        print(f"Target printable width: {target_width_mm} mm")
        print(f"Resolution: {pixel_size_mm} mm/pixel")
        print(f"Pyramid slope: {slope_angle_deg}°, layer height: "
              f"{layer_height} mm")

        binary = load_and_prepare(image_path, target_width_mm, pixel_size_mm,
                                  threshold=args.threshold,
                                  erode_mm=args.erode)
        print(f"Text grid: {binary.shape[1]} x {binary.shape[0]} pixels")

        m, w, h = generate_stamp_stl(binary, pixel_size_mm, margin_mm,
                                      base_height_mm, text_height_mm,
                                      slope_angle_deg, layer_height)

    dims = m.vectors.reshape(-1, 3).max(axis=0) - m.vectors.reshape(-1, 3).min(axis=0)
    print(f"STL dimensions: {dims[0]:.1f} x {dims[1]:.1f} x {dims[2]:.1f} mm")
    print(f"Triangles: {len(m.vectors)}")

    try:
        import trimesh
        import pymeshfix
    except ImportError:
        m.save(output_path)
        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"Saved: {output_path} ({size_mb:.1f} MB)")
        print("Warning: trimesh/pymeshfix not installed, skipping mesh repair")
        return

    tm = trimesh.Trimesh(vertices=m.vectors.reshape(-1, 3),
                         faces=np.arange(len(m.vectors) * 3).reshape(-1, 3))
    tm.merge_vertices()
    print(f"Triangles after merge: {len(tm.faces)}")

    if tm.is_watertight:
        repaired = tm
    else:
        meshfix = pymeshfix.MeshFix(tm.vertices, tm.faces)
        meshfix.repair()
        verts = meshfix.mesh.points
        faces = meshfix.mesh.faces.reshape(-1, 4)[:, 1:]
        repaired = trimesh.Trimesh(vertices=verts, faces=faces)

    if args.decimate > 0:
        target_faces = int(len(repaired.faces) * (1.0 - args.decimate))
        target_faces = max(target_faces, 100)
        repaired = repaired.simplify_quadric_decimation(
            face_count=target_faces)
        print(f"Decimated to {len(repaired.faces)} triangles "
              f"({args.decimate * 100:.0f}% reduction)")

    repaired.export(output_path)
    print(f"Mesh repaired: watertight={repaired.is_watertight}")

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"Saved: {output_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
