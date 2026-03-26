import math

import mlx.core as mx
import numpy as np

from core.mlx_compat import MLXCameraAdapter, MLXTriangleModelAdapter, to_mx_array
from core.mlx_rasterizer.rasterizer import TriangleRasterizationSettings, TriangleRasterizer


def _normalize_triangle_points(triangles_points: mx.array, num_points_per_triangle: mx.array) -> mx.array:
    if len(triangles_points.shape) == 3:
        return triangles_points
    if len(triangles_points.shape) != 2:
        raise ValueError(f"Expected triangle points with rank 2 or 3, got shape {triangles_points.shape}.")

    counts_np = np.array(num_points_per_triangle).astype(np.int32).reshape(-1)
    if counts_np.size == 0 or not np.all(counts_np == counts_np[0]):
        raise ValueError("Renderer wrapper currently expects a uniform number of points per triangle.")

    points_per_triangle = int(counts_np[0])
    triangle_count = counts_np.shape[0]
    return mx.reshape(triangles_points, (triangle_count, points_per_triangle, triangles_points.shape[-1]))


def _coerce_camera(camera) -> MLXCameraAdapter:
    return MLXCameraAdapter.from_object(camera)


def _coerce_triangle_model(model) -> MLXTriangleModelAdapter:
    if isinstance(model, tuple) and len(model) == 2:
        return MLXTriangleModelAdapter.from_reference_snapshot(model[0], model[1])
    return MLXTriangleModelAdapter.from_object(model)


def _scatter_visible_values(number_of_points: int, visible_indices, visible_values, dtype) -> mx.array:
    output = np.zeros((number_of_points,), dtype=dtype)
    visible_indices_np = np.array(visible_indices).astype(np.int32).reshape(-1)
    if visible_indices_np.size > 0:
        output[visible_indices_np] = np.array(visible_values).reshape(-1)
    return mx.array(output)


def _compute_auxiliary_outputs(
    rasterizer: TriangleRasterizer,
    number_of_points: int,
    image_height: int,
    image_width: int,
    world_view_transform: mx.array,
):
    state = rasterizer.last_tiling_state
    render_state = rasterizer.last_render_state
    visible_indices = state["triangle_indices"]
    raw_bboxes = np.array(state["triangle_bboxes"], dtype=np.float32)
    expanded_bboxes = np.array(state["expanded_triangle_bboxes"], dtype=np.float32)
    visible_opacities = np.array(state["visible_triangle_opacities"], dtype=np.float32).reshape(-1)

    if raw_bboxes.size > 0:
        raw_wh = np.maximum(raw_bboxes[:, 2:4] - raw_bboxes[:, 0:2], 0.0)
        expanded_wh = np.maximum(expanded_bboxes[:, 2:4] - expanded_bboxes[:, 0:2], 0.0)
        radii_visible = 0.5 * np.maximum(raw_wh[:, 0], raw_wh[:, 1])
        density_visible = raw_wh[:, 0] * raw_wh[:, 1]
        scaling_visible = np.sqrt(np.sum(expanded_wh * expanded_wh, axis=1))
        max_blending_visible = visible_opacities
    else:
        radii_visible = np.zeros((0,), dtype=np.float32)
        density_visible = np.zeros((0,), dtype=np.float32)
        scaling_visible = np.zeros((0,), dtype=np.float32)
        max_blending_visible = np.zeros((0,), dtype=np.float32)

    radii = _scatter_visible_values(number_of_points, visible_indices, radii_visible, np.float32)
    density_factor = _scatter_visible_values(number_of_points, visible_indices, density_visible, np.float32)
    scaling = _scatter_visible_values(number_of_points, visible_indices, scaling_visible, np.float32)
    max_blending = _scatter_visible_values(number_of_points, visible_indices, max_blending_visible, np.float32)

    zero_map = mx.zeros((1, image_height, image_width), dtype=mx.float32)
    view_normal = render_state.get("view_normal", mx.zeros((3, image_height, image_width), dtype=mx.float32))
    normal_transform = mx.transpose(world_view_transform[:3, :3], (1, 0))
    world_normal = mx.transpose(mx.transpose(view_normal, (1, 2, 0)) @ normal_transform, (2, 0, 1))
    rend_alpha = render_state.get("alpha", zero_map)
    surf_depth = render_state.get("expected_depth", zero_map)
    surf_normal = world_normal * rend_alpha

    return {
        "radii": radii,
        "density_factor": density_factor,
        "scaling": scaling,
        "max_blending": max_blending,
        "rend_alpha": rend_alpha,
        "rend_normal": world_normal,
        "rend_dist": zero_map,
        "surf_depth": surf_depth,
        "surf_normal": surf_normal,
    }


def render(viewpoint_camera, pc, pipe, bg_color: mx.array, scaling_modifier: float = 1.0, override_color: mx.array = None):
    """
    MLX-side compatibility wrapper analogous to the reference `triangle_renderer.render`.
    Returns a reduced but usable dictionary for end-to-end integration work.
    """
    viewpoint_camera = _coerce_camera(viewpoint_camera)
    pc = _coerce_triangle_model(pc)
    bg_color = to_mx_array(bg_color, dtype=mx.float32)

    tanfovx = math.tan(float(viewpoint_camera.FoVx) * 0.5)
    tanfovy = math.tan(float(viewpoint_camera.FoVy) * 0.5)

    raster_settings = TriangleRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=int(pc.active_sh_degree),
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=bool(getattr(pipe, "debug", False)) if pipe is not None else False,
    )

    rasterizer = TriangleRasterizer(raster_settings=raster_settings)

    triangles_points = _normalize_triangle_points(pc.get_triangles_points, pc.get_num_points_per_triangle)
    sigma = mx.reshape(pc.get_sigma, (-1,))
    num_points_per_triangle = pc.get_num_points_per_triangle
    cumsum_of_points_per_triangle = pc.get_cumsum_of_points_per_triangle
    number_of_points = int(pc.get_number_of_points)
    opacities = mx.reshape(pc.get_opacity, (-1,))

    means2d = mx.zeros((number_of_points, 2), dtype=mx.float32)
    scaling = mx.zeros((number_of_points,), dtype=mx.float32)
    density_factor = mx.zeros((number_of_points,), dtype=mx.float32)

    if override_color is not None:
        colors_precomp = to_mx_array(override_color, dtype=mx.float32)
        shs = None
    else:
        colors_precomp = None
        shs = pc.get_features if pc.get_features is not None else mx.zeros((number_of_points, 16, 3), dtype=mx.float32)

    rendered_image = rasterizer(
        triangles_points=triangles_points,
        sigma=sigma,
        num_points_per_triangle=num_points_per_triangle,
        cumsum_of_points_per_triangle=cumsum_of_points_per_triangle,
        number_of_points=number_of_points,
        opacities=opacities,
        means2D=means2d,
        scaling=scaling,
        density_factor=density_factor,
        shs=shs,
        colors_precomp=colors_precomp,
    )

    auxiliary_outputs = _compute_auxiliary_outputs(
        rasterizer=rasterizer,
        number_of_points=number_of_points,
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        world_view_transform=viewpoint_camera.world_view_transform,
    )

    result = {
        "render": rendered_image,
        "viewspace_points": means2d,
        "visibility_filter": auxiliary_outputs["radii"] > 0,
        "num_visible": rasterizer.last_num_visible,
        "triangle_indices": rasterizer.last_tiling_state["triangle_indices"],
        "rasterizer_state": rasterizer.last_tiling_state,
        "radii": auxiliary_outputs["radii"],
        "scaling": auxiliary_outputs["scaling"],
        "density_factor": auxiliary_outputs["density_factor"],
        "max_blending": auxiliary_outputs["max_blending"],
        "rend_alpha": auxiliary_outputs["rend_alpha"],
        "rend_normal": auxiliary_outputs["rend_normal"],
        "rend_dist": auxiliary_outputs["rend_dist"],
        "surf_depth": auxiliary_outputs["surf_depth"],
        "surf_normal": auxiliary_outputs["surf_normal"],
    }
    return result
