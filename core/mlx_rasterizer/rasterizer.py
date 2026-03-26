import mlx.core as mx
import mlx.nn as nn
import numpy as np # <--- NEU
from dataclasses import dataclass
from typing import Optional

SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = (
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
)
SH_C3 = (
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
)

@dataclass
class TriangleRasterizationSettings:
    """
    Speichert alle Kamera- und Rendering-Einstellungen.
    (Wir passen die genauen Felder später noch an das Original an).
    """
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    bg: mx.array
    scale_modifier: float
    viewmatrix: mx.array
    projmatrix: mx.array
    sh_degree: int
    campos: mx.array
    prefiltered: bool
    debug: bool

class TriangleRasterizer(nn.Module):
    """
    Unser Apple Silicon (MLX) Rasterizer!
    Ersetzt die komplette C++/CUDA Pipeline des Originals.
    """
    TILE_SIZE = 16

    def __init__(self, raster_settings: TriangleRasterizationSettings):
        super().__init__()
        self.raster_settings = raster_settings
        self.last_tiling_state = {}
        self.last_num_visible = 0
        self.last_render_state = {}

    def project_points(self, points_3d: mx.array, proj_matrix: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """
        Der MLX-Ersatz für preprocessCUDA! 
        Projiziert 3D-Punkte in den 2D-Bildschirmraum.
        """
        # 1. Wir brauchen die Anzahl der Punkte (N)
        N = points_3d.shape[0]
        
        # 2. Homogene Koordinaten: Wir hängen eine '1' an jedes (x, y, z) an -> (x, y, z, 1)
        ones = mx.ones((N, 1))
        points_hom = mx.concatenate([points_3d, ones], axis=-1)
        
        # 3. Matrizenmultiplikation mit der Projektionsmatrix
        # In MLX (wie in PyTorch) nutzen wir das @ Symbol für Matrix-Multiplikation
        p_proj = points_hom @ proj_matrix
        
        # 4. Perspektivische Division (Teilen durch w)
        # Wir extrahieren w (die letzte Spalte) und die x,y,z Koordinaten
        w = p_proj[:, 3:4]
        
        # Kleine Sicherheitsmaßnahme (+ 1e-7), damit wir nie durch Null teilen!
        p_ndc = p_proj[:, 0:3] / (w + 1e-7) 
        
        # Wir geben die normalisierten 2D-Koordinaten (NDC), das projizierte Z und W zurück
        return p_ndc, p_proj[:, 2:3], w

    def flatten_triangle_points(self, triangles_points: mx.array) -> mx.array:
        """
        Akzeptiert sowohl flache Punktlisten `[N, 3]` als auch Triangle-Batches
        `[T, K, 3]` und normalisiert intern auf `[N, 3]`.
        """
        if len(triangles_points.shape) == 2:
            return triangles_points
        if len(triangles_points.shape) == 3:
            return mx.reshape(triangles_points, (-1, triangles_points.shape[-1]))
        raise ValueError(
            f"Expected `triangles_points` with shape [N, 3] or [T, K, 3], got {triangles_points.shape}."
        )

    def ndc_to_screen(self, points_ndc: mx.array) -> mx.array:
        """
        Entspricht ndc2Pix aus dem CUDA-Referenzcode.
        """
        W = self.raster_settings.image_width
        H = self.raster_settings.image_height

        screen_x = ((points_ndc[:, 0] + 1.0) * W - 1.0) * 0.5
        screen_y = ((points_ndc[:, 1] + 1.0) * H - 1.0) * 0.5

        return mx.stack([screen_x, screen_y], axis=1)

    def compute_triangle_centers(
        self,
        triangles_points: mx.array,
        flat_triangle_points: mx.array,
        num_points_per_triangle: mx.array,
        cumsum_of_points_per_triangle: mx.array,
        triangle_count: int,
    ) -> mx.array:
        """
        Berechnet Dreiecks-Schwerpunkte. Im Standardfall `[T, K, 3]`
        bleibt das vollständig in MLX und damit differenzierbar.
        """
        if triangle_count == 0:
            return mx.zeros((0, 3), dtype=mx.float32)

        if len(triangles_points.shape) == 3:
            return mx.mean(triangles_points[:triangle_count], axis=1)

        counts_np = np.array(num_points_per_triangle).astype(np.int32).reshape(-1)[:triangle_count]
        offsets_np = np.array(cumsum_of_points_per_triangle).astype(np.int32).reshape(-1)[:triangle_count]

        if counts_np.size > 0 and np.all(counts_np == counts_np[0]):
            points_per_triangle = int(counts_np[0])
            expected_points = triangle_count * points_per_triangle
            if flat_triangle_points.shape[0] >= expected_points:
                reshaped = mx.reshape(
                    flat_triangle_points[:expected_points],
                    (triangle_count, points_per_triangle, flat_triangle_points.shape[-1]),
                )
                return mx.mean(reshaped, axis=1)

        centers_np = np.zeros((triangle_count, 3), dtype=np.float32)
        flat_points_np = np.array(flat_triangle_points)
        for triangle_idx in range(triangle_count):
            count = int(counts_np[triangle_idx])
            start = int(offsets_np[triangle_idx])
            end = start + count
            if count > 0 and end <= flat_points_np.shape[0]:
                centers_np[triangle_idx] = flat_points_np[start:end].mean(axis=0)
        return mx.array(centers_np)

    def compute_color_from_sh(self, triangle_centers: mx.array, shs: mx.array) -> tuple[mx.array, mx.array]:
        """
        MLX-Port von `computeColorFromSH` aus dem CUDA-Referenzcode.
        """
        triangle_count = triangle_centers.shape[0]
        if triangle_count == 0:
            empty_colors = mx.zeros((0, 3), dtype=mx.float32)
            return empty_colors, empty_colors < 0.0

        coeff_count = shs.shape[1] if len(shs.shape) > 1 else 0
        effective_degree = min(self.raster_settings.sh_degree, 3, int(np.sqrt(coeff_count)) - 1)

        dir_vec = triangle_centers - mx.expand_dims(self.raster_settings.campos, axis=0)
        dir_norm = mx.sqrt(mx.sum(dir_vec * dir_vec, axis=1, keepdims=True))
        dir_norm = mx.maximum(dir_norm, 1e-8)
        direction = dir_vec / dir_norm

        x = direction[:, 0:1]
        y = direction[:, 1:2]
        z = direction[:, 2:3]

        result = SH_C0 * shs[:, 0, :]

        if effective_degree > 0:
            result = result - SH_C1 * y * shs[:, 1, :] + SH_C1 * z * shs[:, 2, :] - SH_C1 * x * shs[:, 3, :]

            if effective_degree > 1:
                xx = x * x
                yy = y * y
                zz = z * z
                xy = x * y
                yz = y * z
                xz = x * z

                result = (
                    result
                    + SH_C2[0] * xy * shs[:, 4, :]
                    + SH_C2[1] * yz * shs[:, 5, :]
                    + SH_C2[2] * (2.0 * zz - xx - yy) * shs[:, 6, :]
                    + SH_C2[3] * xz * shs[:, 7, :]
                    + SH_C2[4] * (xx - yy) * shs[:, 8, :]
                )

                if effective_degree > 2:
                    result = (
                        result
                        + SH_C3[0] * y * (3.0 * xx - yy) * shs[:, 9, :]
                        + SH_C3[1] * xy * z * shs[:, 10, :]
                        + SH_C3[2] * y * (4.0 * zz - xx - yy) * shs[:, 11, :]
                        + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shs[:, 12, :]
                        + SH_C3[4] * x * (4.0 * zz - xx - yy) * shs[:, 13, :]
                        + SH_C3[5] * z * (xx - yy) * shs[:, 14, :]
                        + SH_C3[6] * x * (xx - 3.0 * yy) * shs[:, 15, :]
                    )

        result = result + 0.5
        clamped = result < 0.0
        return mx.maximum(result, 0.0), clamped

    def group_points_by_triangle(
        self,
        triangles_points: mx.array,
        flat_points: mx.array,
        num_points_per_triangle: mx.array,
        cumsum_of_points_per_triangle: mx.array,
        triangle_count: int,
    ) -> mx.array:
        """
        Gruppiert flache Punktdaten zurück zu `[T, K, C]`.
        """
        if triangle_count == 0:
            return mx.zeros((0, 0, flat_points.shape[-1]), dtype=flat_points.dtype)

        if len(triangles_points.shape) == 3:
            points_per_triangle = triangles_points.shape[1]
            return mx.reshape(
                flat_points[: triangle_count * points_per_triangle],
                (triangle_count, points_per_triangle, flat_points.shape[-1]),
            )

        counts_np = np.array(num_points_per_triangle).astype(np.int32).reshape(-1)[:triangle_count]
        offsets_np = np.array(cumsum_of_points_per_triangle).astype(np.int32).reshape(-1)[:triangle_count]

        if counts_np.size > 0 and np.all(counts_np == counts_np[0]):
            points_per_triangle = int(counts_np[0])
            return mx.reshape(
                flat_points[: triangle_count * points_per_triangle],
                (triangle_count, points_per_triangle, flat_points.shape[-1]),
            )

        flat_points_np = np.array(flat_points)
        grouped_np = []
        for triangle_idx in range(triangle_count):
            count = int(counts_np[triangle_idx])
            start = int(offsets_np[triangle_idx])
            end = start + count
            grouped_np.append(flat_points_np[start:end])
        return mx.array(np.stack(grouped_np))

    def compute_expanded_tile_coverage(
        self,
        normals_np: np.ndarray,
        expanded_offsets_np: np.ndarray,
        tiles_x: int,
        tiles_y: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
        """
        CUDA-nahe Bestimmung der erweiterten Screen-/Tile-Abdeckung aus den
        Schnittpunkten der verschobenen Kanten.
        """
        intersections = []
        for edge_idx in range(3):
            normal = normals_np[edge_idx]
            offset = expanded_offsets_np[edge_idx]
            previous_normal = normals_np[(edge_idx - 1) % 3]
            previous_offset = expanded_offsets_np[(edge_idx - 1) % 3]

            det = normal[0] * previous_normal[1] - normal[1] * previous_normal[0]
            if abs(det) < 1.0e-3:
                continue

            intersect_x = -(offset * previous_normal[1] - previous_offset * normal[1]) / det
            intersect_y = -(previous_offset * normal[0] - offset * previous_normal[0]) / det
            intersections.append([intersect_x, intersect_y])

        if not intersections:
            return None, None, None

        intersections_np = np.asarray(intersections, dtype=np.float32)
        min_xy = intersections_np.min(axis=0)
        max_xy = intersections_np.max(axis=0)

        tile_min = np.floor(min_xy / self.TILE_SIZE).astype(np.int32)
        tile_max = np.floor((max_xy + self.TILE_SIZE - 1) / self.TILE_SIZE).astype(np.int32)

        tile_min = np.clip(tile_min, [0, 0], [tiles_x, tiles_y])
        tile_max = np.clip(tile_max, [0, 0], [tiles_x, tiles_y])

        if tile_max[0] <= tile_min[0] or tile_max[1] <= tile_min[1]:
            return None, None, None

        return np.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=np.float32), tile_min, tile_max

    def render_tiles(
        self,
        W: int,
        H: int,
        tiles_x: int,
        tiles_y: int,
        tile_ranges_np: dict[int, list[int]],
        visible_local_index: dict[int, int],
        preprocess_normals: mx.array,
        preprocess_offsets: mx.array,
        preprocess_inv_phi: mx.array,
        visible_triangle_sigma: mx.array,
        visible_triangle_opacities: mx.array,
        visible_triangle_colors: mx.array,
        visible_triangle_depths: mx.array,
        visible_triangle_view_normals: mx.array,
    ) -> dict:
        """
        Tile-weiser Painter's-Algorithm-Renderpass.
        """
        bg = mx.reshape(self.raster_settings.bg, (1, 1, 3))
        image_tile_rows = []
        alpha_tile_rows = []
        depth_tile_rows = []
        normal_tile_rows = []

        for tile_y in range(tiles_y):
            image_row_tiles = []
            alpha_row_tiles = []
            depth_row_tiles = []
            normal_row_tiles = []
            for tile_x in range(tiles_x):
                pixel_x_start = tile_x * self.TILE_SIZE
                pixel_x_end = min(pixel_x_start + self.TILE_SIZE, W)
                pixel_y_start = tile_y * self.TILE_SIZE
                pixel_y_end = min(pixel_y_start + self.TILE_SIZE, H)

                tile_h = pixel_y_end - pixel_y_start
                tile_w = pixel_x_end - pixel_x_start

                tile_id = tile_y * tiles_x + tile_x
                xs = mx.arange(pixel_x_start, pixel_x_end, dtype=mx.float32)
                ys = mx.arange(pixel_y_start, pixel_y_end, dtype=mx.float32)
                grid_x = mx.broadcast_to(mx.expand_dims(xs, axis=0), (tile_h, tile_w))
                grid_y = mx.broadcast_to(mx.expand_dims(ys, axis=1), (tile_h, tile_w))

                transmittance = mx.ones((tile_h, tile_w), dtype=mx.float32)
                accum = mx.zeros((tile_h, tile_w, 3), dtype=mx.float32)
                accum_alpha = mx.zeros((tile_h, tile_w), dtype=mx.float32)
                accum_depth = mx.zeros((tile_h, tile_w), dtype=mx.float32)
                accum_normal = mx.zeros((tile_h, tile_w, 3), dtype=mx.float32)

                for global_triangle_idx in tile_ranges_np.get(tile_id, []):
                    local_idx = visible_local_index.get(global_triangle_idx)
                    if local_idx is None:
                        continue

                    edge_normals = preprocess_normals[local_idx]
                    edge_offsets = preprocess_offsets[local_idx]
                    max_val = None
                    outside = None

                    for edge_idx in range(3):
                        dist = (
                            edge_normals[edge_idx, 0] * grid_x
                            + edge_normals[edge_idx, 1] * grid_y
                            + edge_offsets[edge_idx]
                        )
                        max_val = dist if max_val is None else mx.maximum(max_val, dist)
                        current_outside = dist > 0.0
                        outside = current_outside if outside is None else mx.logical_or(outside, current_outside)

                    phi_final = max_val * preprocess_inv_phi[local_idx]
                    phi_final_clamped = mx.maximum(phi_final, 1.0e-8)
                    cx = mx.where(
                        phi_final > 0.0,
                        mx.power(phi_final_clamped, visible_triangle_sigma[local_idx]),
                        0.0,
                    )
                    alpha = mx.minimum(0.99, visible_triangle_opacities[local_idx] * cx)
                    alpha = mx.where(outside, 0.0, alpha)
                    alpha = mx.where(alpha < (1.0 / 255.0), 0.0, alpha)

                    blending = alpha * transmittance
                    accum = accum + mx.expand_dims(blending, axis=-1) * mx.reshape(
                        visible_triangle_colors[local_idx], (1, 1, 3)
                    )
                    accum_alpha = accum_alpha + blending
                    accum_depth = accum_depth + blending * visible_triangle_depths[local_idx]
                    accum_normal = accum_normal + mx.expand_dims(blending, axis=-1) * mx.reshape(
                        visible_triangle_view_normals[local_idx], (1, 1, 3)
                    )
                    transmittance = transmittance * (1.0 - alpha)

                safe_alpha = mx.maximum(accum_alpha, 1.0e-8)
                expected_depth = mx.where(accum_alpha > 0.0, accum_depth / safe_alpha, 0.0)
                expected_normal = mx.where(
                    mx.expand_dims(accum_alpha > 0.0, axis=-1),
                    accum_normal / mx.expand_dims(safe_alpha, axis=-1),
                    0.0,
                )

                image_row_tiles.append(accum + transmittance[..., None] * bg)
                alpha_row_tiles.append(mx.expand_dims(accum_alpha, axis=-1))
                depth_row_tiles.append(mx.expand_dims(expected_depth, axis=-1))
                normal_row_tiles.append(expected_normal)

            image_tile_rows.append(mx.concatenate(image_row_tiles, axis=1))
            alpha_tile_rows.append(mx.concatenate(alpha_row_tiles, axis=1))
            depth_tile_rows.append(mx.concatenate(depth_row_tiles, axis=1))
            normal_tile_rows.append(mx.concatenate(normal_row_tiles, axis=1))

        image_hwc = mx.concatenate(image_tile_rows, axis=0)
        alpha_hwc = mx.concatenate(alpha_tile_rows, axis=0)
        depth_hwc = mx.concatenate(depth_tile_rows, axis=0)
        normal_hwc = mx.concatenate(normal_tile_rows, axis=0)

        return {
            "image": mx.transpose(image_hwc, (2, 0, 1)),
            "alpha": mx.transpose(alpha_hwc, (2, 0, 1)),
            "expected_depth": mx.transpose(depth_hwc, (2, 0, 1)),
            "view_normal": mx.transpose(normal_hwc, (2, 0, 1)),
        }

    def compute_triangle_depths_and_candidates(
        self,
        triangle_centers: mx.array,
        counts_np: np.ndarray,
        opacities_np: np.ndarray,
        viewmatrix_np: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Bestimmt Dreieckstiefen im View-Space und die ersten sichtbaren
        Kandidaten auf Dreiecksebene.
        """
        triangle_count = triangle_centers.shape[0]
        if triangle_count == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        triangle_centers_np = np.array(triangle_centers)
        centers_hom_np = np.concatenate(
            [triangle_centers_np, np.ones((triangle_count, 1), dtype=np.float32)],
            axis=1,
        )
        centers_view_np = centers_hom_np @ viewmatrix_np
        triangle_depths_np = centers_view_np[:, 2]

        triangle_is_visible_np = (
            (counts_np[:triangle_count] >= 3)
            & (opacities_np[:triangle_count] > 0.01)
            & (triangle_depths_np > 0.2)
        )
        candidate_visible_triangle_indices_np = np.nonzero(triangle_is_visible_np)[0]
        return triangle_depths_np.astype(np.float32), candidate_visible_triangle_indices_np.astype(np.int32)

    def preprocess_visible_triangles(
        self,
        candidate_visible_triangle_indices_np: np.ndarray,
        candidate_visible_triangle_points_screen: mx.array,
        candidate_visible_triangle_centers_screen: mx.array,
        candidate_visible_triangle_colors: mx.array,
        candidate_visible_triangle_opacities: mx.array,
        candidate_visible_triangle_sigma: mx.array,
        candidate_visible_triangle_depths: mx.array,
        candidate_visible_triangle_view_normals: mx.array,
        triangle_depths_np: np.ndarray,
        tiles_x: int,
        tiles_y: int,
    ) -> dict:
        """
        CUDA-nahe Vorverarbeitung pro sichtbarem Dreieck:
        Blur-Footprint, Kantenparameter, Tile-Überlappungen und Sortierschlüssel.
        """
        visible_triangle_indices_np = []
        triangle_bboxes_np = []
        expanded_triangle_bboxes_np = []
        rect_min_np = []
        rect_max_np = []
        tiles_touched_np = []
        overlap_tile_ids_np = []
        overlap_triangle_indices_np = []
        visible_triangle_depths_np = []
        sorting_keys_np = []
        preprocess_normals = []
        preprocess_offsets = []
        preprocess_inv_phi = []
        visible_triangle_points_screen_list = []
        visible_triangle_centers_screen_list = []
        visible_triangle_colors_list = []
        visible_triangle_opacities_list = []
        visible_triangle_sigma_list = []
        visible_triangle_depths_list = []
        visible_triangle_view_normals_list = []

        for local_idx, triangle_idx in enumerate(candidate_visible_triangle_indices_np.tolist()):
            points_2d = candidate_visible_triangle_points_screen[local_idx]
            center_2d = candidate_visible_triangle_centers_screen[local_idx]

            p0 = points_2d[0]
            p1 = points_2d[1]
            p2 = points_2d[2]

            triangle_points_screen_np = np.array(points_2d)
            min_xy = triangle_points_screen_np.min(axis=0)
            max_xy = triangle_points_screen_np.max(axis=0)
            raw_bbox = np.array([min_xy[0], min_xy[1], max_xy[0], max_xy[1]], dtype=np.float32)
            distances_to_center = mx.sqrt(mx.sum((points_2d - center_2d) * (points_2d - center_2d), axis=1))
            max_point_distance = mx.max(distances_to_center)

            side_a = mx.sqrt(mx.sum((p1 - p2) * (p1 - p2)))
            side_b = mx.sqrt(mx.sum((p0 - p2) * (p0 - p2)))
            side_c = mx.sqrt(mx.sum((p0 - p1) * (p0 - p1)))
            side_sum = mx.maximum(side_a + side_b + side_c, 1e-8)
            incenter = (side_a * p0 + side_b * p1 + side_c * p2) / side_sum

            sigma_value = mx.maximum(candidate_visible_triangle_sigma[local_idx], 1e-8)
            opacity_value = mx.maximum(candidate_visible_triangle_opacities[local_idx], 1.0e-6)
            ratio = 0.01 / opacity_value
            exponent = 1.0 / sigma_value

            triangle_normals_local = []
            triangle_offsets_local = []
            phi_center_min = None
            size = None

            for edge_idx in range(3):
                edge_start = points_2d[edge_idx]
                edge_end = points_2d[(edge_idx + 1) % 3]

                nx = edge_end[1] - edge_start[1]
                ny = -(edge_end[0] - edge_start[0])
                normal = mx.stack([nx, ny], axis=0)
                normal_norm = mx.maximum(mx.sqrt(mx.sum(normal * normal)), 1e-8)
                normal = normal / normal_norm

                offset = -(normal[0] * edge_start[0] + normal[1] * edge_start[1])
                dist = normal[0] * incenter[0] + normal[1] * incenter[1] + offset

                if float(np.array(dist > 0.0)):
                    normal = -normal
                    offset = -offset
                    dist = -dist

                if phi_center_min is None:
                    phi_center_min = dist

                if size is None:
                    size = phi_center_min * mx.power(ratio, exponent)

                triangle_normals_local.append(normal)
                triangle_offsets_local.append(offset - size)

            phi_center_min_scalar = float(np.array(phi_center_min))
            max_point_distance_scalar = float(np.array(max_point_distance))
            if max_point_distance_scalar > 1600.0 or max_point_distance_scalar < 1.0 or phi_center_min_scalar > -1.0:
                continue

            normals_local = mx.stack(triangle_normals_local, axis=0)
            offsets_local = mx.stack(triangle_offsets_local, axis=0)
            expanded_bbox, tile_min, tile_max = self.compute_expanded_tile_coverage(
                np.array(normals_local),
                np.array(offsets_local),
                tiles_x,
                tiles_y,
            )
            if expanded_bbox is None:
                continue

            visible_triangle_indices_np.append(triangle_idx)
            triangle_bboxes_np.append(raw_bbox)
            expanded_triangle_bboxes_np.append(expanded_bbox)
            rect_min_np.append(tile_min)
            rect_max_np.append(tile_max)
            visible_triangle_depths_np.append(triangle_depths_np[triangle_idx])

            tiles_touched = int((tile_max[0] - tile_min[0]) * (tile_max[1] - tile_min[1]))
            tiles_touched_np.append(tiles_touched)
            depth_bits = np.float32(triangle_depths_np[triangle_idx]).view(np.uint32).item()
            for tile_y in range(int(tile_min[1]), int(tile_max[1])):
                for tile_x in range(int(tile_min[0]), int(tile_max[0])):
                    tile_id = tile_y * tiles_x + tile_x
                    overlap_tile_ids_np.append(tile_id)
                    overlap_triangle_indices_np.append(triangle_idx)
                    sorting_keys_np.append(
                        np.int64((np.uint64(tile_id) << np.uint64(32)) | np.uint64(depth_bits))
                    )

            preprocess_normals.append(normals_local)
            preprocess_offsets.append(offsets_local)
            preprocess_inv_phi.append(1.0 / mx.minimum(phi_center_min, -1e-8))
            visible_triangle_points_screen_list.append(points_2d)
            visible_triangle_centers_screen_list.append(center_2d)
            visible_triangle_colors_list.append(candidate_visible_triangle_colors[local_idx])
            visible_triangle_opacities_list.append(candidate_visible_triangle_opacities[local_idx])
            visible_triangle_sigma_list.append(candidate_visible_triangle_sigma[local_idx])
            visible_triangle_depths_list.append(candidate_visible_triangle_depths[local_idx])
            visible_triangle_view_normals_list.append(candidate_visible_triangle_view_normals[local_idx])

        if triangle_bboxes_np:
            return {
                "triangle_bboxes": mx.array(np.stack(triangle_bboxes_np).astype(np.float32)),
                "expanded_triangle_bboxes": mx.array(np.stack(expanded_triangle_bboxes_np).astype(np.float32)),
                "rect_min": mx.array(np.stack(rect_min_np).astype(np.int32)),
                "rect_max": mx.array(np.stack(rect_max_np).astype(np.int32)),
                "tiles_touched": mx.array(np.array(tiles_touched_np, dtype=np.int32)),
                "overlap_tile_ids": mx.array(np.array(overlap_tile_ids_np, dtype=np.int32)),
                "overlap_triangle_indices": mx.array(np.array(overlap_triangle_indices_np, dtype=np.int32)),
                "visible_triangle_indices": mx.array(np.array(visible_triangle_indices_np, dtype=np.int32)),
                "visible_triangle_depths": mx.array(np.array(visible_triangle_depths_np, dtype=np.float32)),
                "sorting_keys": mx.array(np.array(sorting_keys_np, dtype=np.int64)),
                "visible_triangle_points_screen": mx.stack(visible_triangle_points_screen_list, axis=0),
                "visible_triangle_centers_screen": mx.stack(visible_triangle_centers_screen_list, axis=0),
                "visible_triangle_colors": mx.stack(visible_triangle_colors_list, axis=0),
                "visible_triangle_opacities": mx.stack(visible_triangle_opacities_list, axis=0),
                "visible_triangle_sigma": mx.stack(visible_triangle_sigma_list, axis=0),
                "visible_triangle_depths": mx.stack(visible_triangle_depths_list, axis=0),
                "visible_triangle_view_normals": mx.stack(visible_triangle_view_normals_list, axis=0),
                "preprocess_normals": mx.stack(preprocess_normals, axis=0),
                "preprocess_offsets": mx.stack(preprocess_offsets, axis=0),
                "preprocess_inv_phi": mx.stack(preprocess_inv_phi, axis=0),
                "visible_triangle_indices_np": visible_triangle_indices_np,
            }

        return {
            "triangle_bboxes": mx.zeros((0, 4), dtype=mx.float32),
            "expanded_triangle_bboxes": mx.zeros((0, 4), dtype=mx.float32),
            "rect_min": mx.zeros((0, 2), dtype=mx.int32),
            "rect_max": mx.zeros((0, 2), dtype=mx.int32),
            "tiles_touched": mx.zeros((0,), dtype=mx.int32),
            "overlap_tile_ids": mx.zeros((0,), dtype=mx.int32),
            "overlap_triangle_indices": mx.zeros((0,), dtype=mx.int32),
            "visible_triangle_indices": mx.zeros((0,), dtype=mx.int32),
            "visible_triangle_depths": mx.zeros((0,), dtype=mx.float32),
            "sorting_keys": mx.zeros((0,), dtype=mx.int64),
            "visible_triangle_points_screen": mx.zeros((0, 3, 2), dtype=mx.float32),
            "visible_triangle_centers_screen": mx.zeros((0, 2), dtype=mx.float32),
            "visible_triangle_colors": mx.zeros((0, 3), dtype=mx.float32),
            "visible_triangle_opacities": mx.zeros((0,), dtype=mx.float32),
            "visible_triangle_sigma": mx.zeros((0,), dtype=mx.float32),
            "visible_triangle_depths": mx.zeros((0,), dtype=mx.float32),
            "visible_triangle_view_normals": mx.zeros((0, 3), dtype=mx.float32),
            "preprocess_normals": mx.zeros((0, 3, 2), dtype=mx.float32),
            "preprocess_offsets": mx.zeros((0, 3), dtype=mx.float32),
            "preprocess_inv_phi": mx.zeros((0,), dtype=mx.float32),
            "visible_triangle_indices_np": [],
        }

    def __call__(
        self, 
        triangles_points: mx.array, 
        sigma: mx.array, 
        num_points_per_triangle: mx.array, 
        cumsum_of_points_per_triangle: mx.array, 
        number_of_points: int, 
        opacities: mx.array, 
        means2D: mx.array, 
        scaling: mx.array, 
        density_factor: mx.array, 
        shs: Optional[mx.array] = None, 
        colors_precomp: Optional[mx.array] = None
    ) -> mx.array:
        
        # 1. Sicherheits-Checks
        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise ValueError('Please provide exactly one of either SHs or precomputed colors!')
        
        if shs is None:
            shs = mx.array([])
        if colors_precomp is None:
            colors_precomp = mx.array([])

        flat_triangle_points = self.flatten_triangle_points(triangles_points)

        # ==========================================
        # 2. PROJEKTION (Unser Schritt 1)
        # ==========================================
        # Wir jagen alle Millionen Dreiecks-Punkte durch deine neue Funktion!
        p_ndc, p_z, w = self.project_points(flat_triangle_points, self.raster_settings.projmatrix)

        # ==========================================
        # 3. FRUSTUM CULLING (Der Türsteher)
        # ==========================================
        # Wir erstellen eine "Maske" (True = Behalten, False = Wegwerfen)
        is_visible = mx.squeeze(p_z > 0.2)
        
        # MLX-WORKAROUND FÜR DYNAMISCHE SHAPES:
        # Wir zwingen MLX die Maske auszuwerten und übergeben sie an NumPy
        is_visible_np = np.array(is_visible) 
        
        # NumPy holt uns die konkreten Index-Positionen (0, 1, 2...)
        valid_indices_np = np.nonzero(is_visible_np)[0]
        
        # Wir schieben die festen Indices zurück in die Apple Silicon Welt!
        valid_indices = mx.array(valid_indices_np)
        
        # Wie viele Punkte haben überlebt?
        num_visible = valid_indices.shape[0]
        self.last_num_visible = int(num_visible)

        # ==========================================
        # 4. SCREEN SPACE TRANSFORMATION (NDC zu Pixeln)
        # ==========================================
        # Wir transformieren alle projizierten Punkte; die diskrete Dreiecks-
        # Selektion passiert danach über den NumPy-Bridge-Weg.
        points_screen = self.ndc_to_screen(p_ndc)

        # ==========================================
        # 5. TASK A: TILING & BOUNDING BOXES
        # ==========================================
        W = self.raster_settings.image_width
        H = self.raster_settings.image_height
        tiles_x = (W + self.TILE_SIZE - 1) // self.TILE_SIZE
        tiles_y = (H + self.TILE_SIZE - 1) // self.TILE_SIZE

        triangle_count = int(number_of_points)
        counts_np = np.array(num_points_per_triangle).astype(np.int32).reshape(-1)
        cumsum_np = np.array(cumsum_of_points_per_triangle).astype(np.int32).reshape(-1)
        opacities_np = np.array(opacities).reshape(-1)
        viewmatrix_np = np.array(self.raster_settings.viewmatrix)

        triangle_count = min(
            triangle_count,
            counts_np.shape[0] if counts_np.size else 0,
            cumsum_np.shape[0] if cumsum_np.size else 0,
            opacities_np.shape[0] if opacities_np.size else 0,
        )

        triangle_centers = self.compute_triangle_centers(
            triangles_points,
            flat_triangle_points,
            num_points_per_triangle,
            cumsum_of_points_per_triangle,
            triangle_count,
        )

        has_shs = len(shs.shape) > 0 and shs.shape[0] > 0
        if has_shs:
            triangle_colors, triangle_color_clamped = self.compute_color_from_sh(
                triangle_centers,
                shs[:triangle_count],
            )
        else:
            triangle_colors = colors_precomp[:triangle_count]
            triangle_color_clamped = triangle_colors < 0.0

        triangle_depths_np, candidate_visible_triangle_indices_np = self.compute_triangle_depths_and_candidates(
            triangle_centers=triangle_centers,
            counts_np=counts_np,
            opacities_np=opacities_np,
            viewmatrix_np=viewmatrix_np,
        )

        triangle_points_screen = self.group_points_by_triangle(
            triangles_points,
            points_screen,
            num_points_per_triangle,
            cumsum_of_points_per_triangle,
            triangle_count,
        )

        if triangle_count > 0:
            triangle_centers_ndc, _, _ = self.project_points(triangle_centers, self.raster_settings.projmatrix)
            triangle_centers_screen = self.ndc_to_screen(triangle_centers_ndc)
        else:
            triangle_centers_screen = mx.zeros((0, 2), dtype=mx.float32)

        candidate_visible_triangle_indices = mx.array(candidate_visible_triangle_indices_np.astype(np.int32))
        candidate_visible_triangle_points_screen = (
            triangle_points_screen[candidate_visible_triangle_indices]
            if candidate_visible_triangle_indices.shape[0] > 0
            else mx.zeros((0, 3, 2), dtype=mx.float32)
        )
        candidate_visible_triangle_centers_screen = (
            triangle_centers_screen[candidate_visible_triangle_indices]
            if candidate_visible_triangle_indices.shape[0] > 0
            else mx.zeros((0, 2), dtype=mx.float32)
        )
        candidate_visible_triangle_colors = (
            triangle_colors[candidate_visible_triangle_indices]
            if candidate_visible_triangle_indices.shape[0] > 0
            else mx.zeros((0, 3), dtype=mx.float32)
        )
        candidate_visible_triangle_opacities = (
            opacities[:triangle_count][candidate_visible_triangle_indices]
            if candidate_visible_triangle_indices.shape[0] > 0
            else mx.zeros((0,), dtype=mx.float32)
        )
        candidate_visible_triangle_sigma = (
            mx.reshape(sigma[:triangle_count], (-1,))[candidate_visible_triangle_indices]
            if candidate_visible_triangle_indices.shape[0] > 0
            else mx.zeros((0,), dtype=mx.float32)
        )
        flat_triangle_points_h = mx.concatenate(
            [flat_triangle_points, mx.ones((flat_triangle_points.shape[0], 1), dtype=flat_triangle_points.dtype)],
            axis=1,
        )
        flat_triangle_points_view = (flat_triangle_points_h @ self.raster_settings.viewmatrix)[:, :3]
        triangle_points_view = self.group_points_by_triangle(
            triangles_points,
            flat_triangle_points_view,
            num_points_per_triangle,
            cumsum_of_points_per_triangle,
            triangle_count,
        )
        candidate_visible_triangle_points_view = (
            triangle_points_view[candidate_visible_triangle_indices]
            if candidate_visible_triangle_indices.shape[0] > 0
            else mx.zeros((0, 3, 3), dtype=mx.float32)
        )
        if candidate_visible_triangle_points_view.shape[0] > 0:
            edge_1 = candidate_visible_triangle_points_view[:, 1] - candidate_visible_triangle_points_view[:, 0]
            edge_2 = candidate_visible_triangle_points_view[:, 2] - candidate_visible_triangle_points_view[:, 0]
            visible_triangle_view_normals = mx.stack(
                [
                    edge_1[:, 1] * edge_2[:, 2] - edge_1[:, 2] * edge_2[:, 1],
                    edge_1[:, 2] * edge_2[:, 0] - edge_1[:, 0] * edge_2[:, 2],
                    edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0],
                ],
                axis=1,
            )
            visible_triangle_view_normals = visible_triangle_view_normals / mx.maximum(
                mx.sqrt(mx.sum(visible_triangle_view_normals * visible_triangle_view_normals, axis=1, keepdims=True)),
                1.0e-8,
            )
            view_direction = -mx.mean(candidate_visible_triangle_points_view, axis=1)
            normal_alignment = mx.sum(visible_triangle_view_normals * view_direction, axis=1, keepdims=True)
            visible_triangle_view_normals = mx.where(
                normal_alignment >= 0.0,
                visible_triangle_view_normals,
                -visible_triangle_view_normals,
            )
        else:
            visible_triangle_view_normals = mx.zeros((0, 3), dtype=mx.float32)
        candidate_visible_triangle_depths = (
            mx.array(triangle_depths_np.astype(np.float32))[candidate_visible_triangle_indices]
            if candidate_visible_triangle_indices.shape[0] > 0
            else mx.zeros((0,), dtype=mx.float32)
        )

        preprocess_state = self.preprocess_visible_triangles(
            candidate_visible_triangle_indices_np=candidate_visible_triangle_indices_np,
            candidate_visible_triangle_points_screen=candidate_visible_triangle_points_screen,
            candidate_visible_triangle_centers_screen=candidate_visible_triangle_centers_screen,
            candidate_visible_triangle_colors=candidate_visible_triangle_colors,
            candidate_visible_triangle_opacities=candidate_visible_triangle_opacities,
            candidate_visible_triangle_sigma=candidate_visible_triangle_sigma,
            candidate_visible_triangle_depths=candidate_visible_triangle_depths,
            candidate_visible_triangle_view_normals=visible_triangle_view_normals,
            triangle_depths_np=triangle_depths_np,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
        )

        triangle_bboxes = preprocess_state["triangle_bboxes"]
        expanded_triangle_bboxes = preprocess_state["expanded_triangle_bboxes"]
        rect_min = preprocess_state["rect_min"]
        rect_max = preprocess_state["rect_max"]
        tiles_touched = preprocess_state["tiles_touched"]
        overlap_tile_ids = preprocess_state["overlap_tile_ids"]
        overlap_triangle_indices = preprocess_state["overlap_triangle_indices"]
        visible_triangle_indices = preprocess_state["visible_triangle_indices"]
        visible_triangle_depths = preprocess_state["visible_triangle_depths"]
        sorting_keys = preprocess_state["sorting_keys"]
        visible_triangle_points_screen = preprocess_state["visible_triangle_points_screen"]
        visible_triangle_centers_screen = preprocess_state["visible_triangle_centers_screen"]
        visible_triangle_colors = preprocess_state["visible_triangle_colors"]
        visible_triangle_opacities = preprocess_state["visible_triangle_opacities"]
        visible_triangle_sigma = preprocess_state["visible_triangle_sigma"]
        visible_triangle_depths = preprocess_state["visible_triangle_depths"]
        visible_triangle_view_normals = preprocess_state["visible_triangle_view_normals"]
        preprocess_normals = preprocess_state["preprocess_normals"]
        preprocess_offsets = preprocess_state["preprocess_offsets"]
        preprocess_inv_phi = preprocess_state["preprocess_inv_phi"]
        visible_triangle_indices_np = preprocess_state["visible_triangle_indices_np"]

        if sorting_keys.shape[0] > 0:
            sort_order = mx.argsort(sorting_keys)
            sorted_keys = sorting_keys[sort_order]
            sorted_tile_ids = overlap_tile_ids[sort_order]
            sorted_triangle_indices = overlap_triangle_indices[sort_order]
        else:
            sort_order = mx.zeros((0,), dtype=mx.int32)
            sorted_keys = sorting_keys
            sorted_tile_ids = overlap_tile_ids
            sorted_triangle_indices = overlap_triangle_indices

        tile_ranges_np = {}
        sorted_tile_ids_np = np.array(sorted_tile_ids).astype(np.int32)
        sorted_triangle_indices_np = np.array(sorted_triangle_indices).astype(np.int32)
        for overlap_idx, tile_id in enumerate(sorted_tile_ids_np):
            tile_ranges_np.setdefault(int(tile_id), []).append(int(sorted_triangle_indices_np[overlap_idx]))

        visible_local_index = {int(global_idx): local_idx for local_idx, global_idx in enumerate(visible_triangle_indices_np)}

        render_state = self.render_tiles(
            W=W,
            H=H,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            tile_ranges_np=tile_ranges_np,
            visible_local_index=visible_local_index,
            preprocess_normals=preprocess_normals,
            preprocess_offsets=preprocess_offsets,
            preprocess_inv_phi=preprocess_inv_phi,
            visible_triangle_sigma=visible_triangle_sigma,
            visible_triangle_opacities=visible_triangle_opacities,
            visible_triangle_colors=visible_triangle_colors,
            visible_triangle_depths=visible_triangle_depths,
            visible_triangle_view_normals=visible_triangle_view_normals,
        )
        rendered_image = render_state["image"]
        self.last_render_state = render_state

        self.last_tiling_state = {
            "tile_size": self.TILE_SIZE,
            "tile_grid_shape": (tiles_y, tiles_x),
            "triangle_indices": visible_triangle_indices,
            "triangle_centers": triangle_centers,
            "triangle_bboxes": triangle_bboxes,
            "expanded_triangle_bboxes": expanded_triangle_bboxes,
            "rect_min": rect_min,
            "rect_max": rect_max,
            "tiles_touched": tiles_touched,
            "overlap_tile_ids": overlap_tile_ids,
            "overlap_triangle_indices": overlap_triangle_indices,
            "triangle_depths": visible_triangle_depths,
            "triangle_colors": triangle_colors,
            "triangle_color_clamped": triangle_color_clamped,
            "visible_triangle_points_screen": visible_triangle_points_screen,
            "visible_triangle_centers_screen": visible_triangle_centers_screen,
            "visible_triangle_colors": visible_triangle_colors,
            "visible_triangle_opacities": visible_triangle_opacities,
            "visible_triangle_sigma": visible_triangle_sigma,
            "visible_triangle_view_normals": visible_triangle_view_normals,
            "preprocess_normals": preprocess_normals,
            "preprocess_offsets": preprocess_offsets,
            "preprocess_inv_phi": preprocess_inv_phi,
            "sorting_keys": sorting_keys,
            "sort_order": sort_order,
            "sorted_keys": sorted_keys,
            "sorted_tile_ids": sorted_tile_ids,
            "sorted_triangle_indices": sorted_triangle_indices,
        }

        return rendered_image
