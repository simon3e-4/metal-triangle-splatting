import numpy as np
import mlx.core as mx

from core.mlx_rasterizer.rasterizer import TriangleRasterizationSettings, TriangleRasterizer


def assert_finite_grad(name, loss_fn, value):
    value_and_grad = mx.value_and_grad(loss_fn)
    loss, grad = value_and_grad(value)
    mx.eval(loss, grad)

    grad_np = np.array(grad)
    print(f"{name} grad abs sum:", float(np.abs(grad_np).sum()))
    assert np.isfinite(grad_np).all(), f"{name} gradient contains non-finite values"


def run_basic_smoke_test():
    settings = TriangleRasterizationSettings(
        image_height=32,
        image_width=32,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=mx.array([0.0, 0.0, 0.0]),
        scale_modifier=1.0,
        viewmatrix=mx.eye(4),
        projmatrix=mx.eye(4),
        sh_degree=3,
        campos=mx.array([0.0, 0.0, 0.0]),
        prefiltered=False,
        debug=False,
    )

    rasterizer = TriangleRasterizer(settings)

    triangles_points = mx.array([
        [
            [-0.5, -0.5, 1.0],
            [0.5, -0.5, 1.0],
            [0.0, 0.5, 1.0],
        ],
        [
            [-0.5, -0.5, 2.0],
            [0.5, -0.5, 2.0],
            [0.0, 0.5, 2.0],
        ],
        [
            [-0.2, -0.2, -1.0],
            [0.2, -0.2, -1.0],
            [0.0, 0.2, -1.0],
        ],
    ])

    num_points_per_triangle = mx.array([3, 3, 3])
    cumsum_of_points_per_triangle = mx.array([0, 3, 6])
    opacities = mx.array([1.0, 1.0, 1.0])

    image = rasterizer(
        triangles_points=triangles_points,
        sigma=mx.zeros((3,)),
        num_points_per_triangle=num_points_per_triangle,
        cumsum_of_points_per_triangle=cumsum_of_points_per_triangle,
        number_of_points=3,
        opacities=opacities,
        means2D=mx.zeros((3, 2)),
        scaling=mx.zeros((3,)),
        density_factor=mx.zeros((3,)),
        shs=mx.zeros((3, 16, 3)),
    )
    num_visible = rasterizer.last_num_visible

    state = rasterizer.last_tiling_state
    image_np = np.array(image)
    triangle_indices = np.array(state["triangle_indices"])
    triangle_bboxes = np.array(state["triangle_bboxes"])
    rect_min = np.array(state["rect_min"])
    rect_max = np.array(state["rect_max"])
    tiles_touched = np.array(state["tiles_touched"])
    overlap_tile_ids = np.array(state["overlap_tile_ids"])
    overlap_triangle_indices = np.array(state["overlap_triangle_indices"])
    sorted_tile_ids = np.array(state["sorted_tile_ids"])
    sorted_triangle_indices = np.array(state["sorted_triangle_indices"])
    triangle_colors = np.array(state["triangle_colors"])

    print("Visible projected points:", int(num_visible))
    print("Rendered image shape:", list(image_np.shape))
    print("Visible triangles:", triangle_indices.tolist())
    print("Triangle colors:", triangle_colors.tolist())
    print("Triangle bboxes:", triangle_bboxes.tolist())
    print("Tile rect min:", rect_min.tolist())
    print("Tile rect max:", rect_max.tolist())
    print("Tiles touched:", tiles_touched.tolist())
    print("Overlap tile ids:", overlap_tile_ids.tolist())
    print("Overlap triangle ids:", overlap_triangle_indices.tolist())
    print("Sorted tile ids:", sorted_tile_ids.tolist())
    print("Sorted triangle ids:", sorted_triangle_indices.tolist())

    assert int(num_visible) == 6
    assert image_np.shape == (3, 32, 32)
    assert float(image_np.sum()) > 0.0
    assert triangle_indices.tolist() == [0, 1]
    assert np.allclose(triangle_colors, 0.5, atol=1e-5)
    assert np.allclose(triangle_bboxes[0], [7.5, 7.5, 23.5, 23.5], atol=1e-5)
    assert np.allclose(triangle_bboxes[1], [7.5, 7.5, 23.5, 23.5], atol=1e-5)
    assert rect_min.tolist() == [[0, 0], [0, 0]]
    assert rect_max.tolist() == [[2, 2], [2, 2]]
    assert tiles_touched.tolist() == [4, 4]
    assert overlap_tile_ids.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert overlap_triangle_indices.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert sorted_tile_ids.tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert sorted_triangle_indices.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]

    def render_with(triangles_points_arg, sigma_arg, opacities_arg, shs_arg):
        image_out = rasterizer(
            triangles_points=triangles_points_arg,
            sigma=sigma_arg,
            num_points_per_triangle=num_points_per_triangle,
            cumsum_of_points_per_triangle=cumsum_of_points_per_triangle,
            number_of_points=3,
            opacities=opacities_arg,
            means2D=mx.zeros((3, 2)),
            scaling=mx.zeros((3,)),
            density_factor=mx.zeros((3,)),
            shs=shs_arg,
        )
        return mx.sum(image_out)

    sigma = mx.ones((3,), dtype=mx.float32)
    shs = mx.zeros((3, 16, 3), dtype=mx.float32)

    assert_finite_grad(
        "triangles_points",
        lambda tp: render_with(tp, sigma, opacities, shs),
        triangles_points,
    )
    assert_finite_grad(
        "sigma",
        lambda sig: render_with(triangles_points, sig, opacities, shs),
        sigma,
    )
    assert_finite_grad(
        "opacities",
        lambda op: render_with(triangles_points, sigma, op, shs),
        opacities,
    )
    assert_finite_grad(
        "shs",
        lambda sh: render_with(triangles_points, sigma, opacities, sh),
        shs,
    )

    print("Task A/B/C/D/E smoke test passed.")


def run_view_dependent_test():
    viewmatrix = mx.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.1, -0.05, 0.2, 1.0],
    ])
    projmatrix = mx.array([
        [0.75, 0.0, 0.0, 0.0],
        [0.0, 0.6, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    settings = TriangleRasterizationSettings(
        image_height=48,
        image_width=64,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=mx.array([0.05, 0.1, 0.15]),
        scale_modifier=1.0,
        viewmatrix=viewmatrix,
        projmatrix=projmatrix,
        sh_degree=3,
        campos=mx.array([0.35, -0.25, 0.1]),
        prefiltered=False,
        debug=False,
    )
    rasterizer = TriangleRasterizer(settings)

    triangles_points = mx.array([
        [
            [-0.8, -0.4, 1.2],
            [0.1, -0.5, 1.2],
            [-0.2, 0.6, 1.2],
        ],
        [
            [0.1, -0.3, 2.0],
            [0.9, -0.2, 2.0],
            [0.45, 0.7, 2.0],
        ],
    ], dtype=mx.float32)
    sigma = mx.array([1.2, 0.9], dtype=mx.float32)
    opacities = mx.array([0.85, 0.7], dtype=mx.float32)
    num_points_per_triangle = mx.array([3, 3])
    cumsum_of_points_per_triangle = mx.array([0, 3])
    shs = mx.zeros((2, 16, 3), dtype=mx.float32)
    shs[:, 0, :] = mx.array([[0.2, 0.1, -0.05], [0.05, 0.25, 0.15]])
    shs[:, 1, :] = mx.array([[0.15, -0.05, 0.1], [-0.1, 0.2, 0.05]])
    shs[:, 2, :] = mx.array([[0.0, 0.1, 0.05], [0.15, -0.05, 0.1]])
    shs[:, 3, :] = mx.array([[-0.05, 0.0, 0.2], [0.1, 0.05, -0.15]])

    image = rasterizer(
        triangles_points=triangles_points,
        sigma=sigma,
        num_points_per_triangle=num_points_per_triangle,
        cumsum_of_points_per_triangle=cumsum_of_points_per_triangle,
        number_of_points=2,
        opacities=opacities,
        means2D=mx.zeros((2, 2)),
        scaling=mx.zeros((2,)),
        density_factor=mx.zeros((2,)),
        shs=shs,
    )
    num_visible = rasterizer.last_num_visible

    image_np = np.array(image)
    triangle_colors = np.array(rasterizer.last_tiling_state["triangle_colors"])
    expanded_bboxes = np.array(rasterizer.last_tiling_state["expanded_triangle_bboxes"])

    print("View-dependent image sum:", float(image_np.sum()))
    print("View-dependent colors:", triangle_colors.tolist())
    print("Expanded bboxes:", expanded_bboxes.tolist())

    assert int(num_visible) == 6
    assert image_np.shape == (3, 48, 64)
    assert float(image_np.sum()) > 0.0
    assert triangle_colors.shape == (2, 3)
    assert not np.allclose(triangle_colors, 0.5, atol=1e-4)
    assert expanded_bboxes.shape == (2, 4)
    assert np.all(expanded_bboxes[:, 2] >= expanded_bboxes[:, 0])
    assert np.all(expanded_bboxes[:, 3] >= expanded_bboxes[:, 1])

    assert_finite_grad(
        "view_dependent_triangles_points",
        lambda tp: mx.sum(
            rasterizer(
                triangles_points=tp,
                sigma=sigma,
                num_points_per_triangle=num_points_per_triangle,
                cumsum_of_points_per_triangle=cumsum_of_points_per_triangle,
                number_of_points=2,
                opacities=opacities,
                means2D=mx.zeros((2, 2)),
                scaling=mx.zeros((2,)),
                density_factor=mx.zeros((2,)),
                shs=shs,
            )
        ),
        triangles_points,
    )

    print("View-dependent camera/SH test passed.")


def main():
    run_basic_smoke_test()
    run_view_dependent_test()


if __name__ == "__main__":
    main()
