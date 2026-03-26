import mlx.core as mx
import numpy as np

from core.mlx_triangle_renderer import render


class DummyCamera:
    FoVx = 1.0
    FoVy = 1.0
    image_height = 32
    image_width = 32
    world_view_transform = mx.eye(4)
    full_proj_transform = mx.eye(4)
    camera_center = mx.array([0.0, 0.0, 0.0])


class DummyPointCloud:
    active_sh_degree = 3
    get_triangles_points = mx.array(
        [
            [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]],
            [[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.0, 0.5, 2.0]],
        ],
        dtype=mx.float32,
    )
    get_sigma = mx.array([1.0, 1.0], dtype=mx.float32)
    get_num_points_per_triangle = mx.array([3, 3], dtype=mx.int32)
    get_cumsum_of_points_per_triangle = mx.array([0, 3], dtype=mx.int32)
    get_number_of_points = 2
    get_opacity = mx.array([0.8, 0.6], dtype=mx.float32)
    get_features = mx.zeros((2, 16, 3), dtype=mx.float32)


class DummyPipe:
    debug = False


class NumpyCamera:
    FoVx = np.float32(1.0)
    FoVy = np.float32(1.0)
    image_height = np.int32(32)
    image_width = np.int32(32)
    world_view_transform = np.eye(4, dtype=np.float32)
    full_proj_transform = np.eye(4, dtype=np.float32)
    camera_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)


class NumpyPointCloud:
    active_sh_degree = np.int32(3)
    get_triangles_points = np.array(
        [
            [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]],
            [[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.0, 0.5, 2.0]],
        ],
        dtype=np.float32,
    )
    get_sigma = np.array([1.0, 1.0], dtype=np.float32)
    get_num_points_per_triangle = np.array([3, 3], dtype=np.int32)
    get_cumsum_of_points_per_triangle = np.array([0, 3], dtype=np.int32)
    get_number_of_points = np.int32(2)
    get_opacity = np.array([0.8, 0.6], dtype=np.float32)
    get_features = np.zeros((2, 16, 3), dtype=np.float32)


def assert_wrapper_result(result):
    image = np.array(result["render"])
    visibility = np.array(result["visibility_filter"])
    indices = np.array(result["triangle_indices"])
    radii = np.array(result["radii"])
    scaling = np.array(result["scaling"])
    density_factor = np.array(result["density_factor"])
    max_blending = np.array(result["max_blending"])
    rend_alpha = np.array(result["rend_alpha"])
    rend_normal = np.array(result["rend_normal"])
    rend_dist = np.array(result["rend_dist"])
    surf_depth = np.array(result["surf_depth"])
    surf_normal = np.array(result["surf_normal"])

    print("render shape:", list(image.shape))
    print("num_visible:", int(result["num_visible"]))
    print("visibility_filter:", visibility.tolist())
    print("triangle_indices:", indices.tolist())
    print("radii:", radii.tolist())
    print("scaling:", scaling.tolist())
    print("density_factor:", density_factor.tolist())
    print("max_blending:", max_blending.tolist())

    assert image.shape == (3, 32, 32)
    assert int(result["num_visible"]) == 6
    assert visibility.tolist() == [True, True]
    assert indices.tolist() == [0, 1]
    assert radii.shape == (2,)
    assert scaling.shape == (2,)
    assert density_factor.shape == (2,)
    assert max_blending.shape == (2,)
    assert np.all(radii > 0.0)
    assert np.all(scaling > 0.0)
    assert np.all(density_factor > 0.0)
    assert np.allclose(max_blending, np.array([0.8, 0.6], dtype=np.float32), atol=1e-5)
    assert rend_alpha.shape == (1, 32, 32)
    assert rend_normal.shape == (3, 32, 32)
    assert rend_dist.shape == (1, 32, 32)
    assert surf_depth.shape == (1, 32, 32)
    assert surf_normal.shape == (3, 32, 32)
    assert float(rend_alpha.sum()) > 0.0
    assert float(np.abs(rend_normal).sum()) > 0.0
    assert float(surf_depth.sum()) > 0.0
    assert float(np.abs(surf_normal).sum()) > 0.0


def main():
    result = render(
        viewpoint_camera=DummyCamera(),
        pc=DummyPointCloud(),
        pipe=DummyPipe(),
        bg_color=mx.array([0.0, 0.0, 0.0]),
    )
    assert_wrapper_result(result)

    numpy_result = render(
        viewpoint_camera=NumpyCamera(),
        pc=NumpyPointCloud(),
        pipe=DummyPipe(),
        bg_color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert_wrapper_result(numpy_result)

    print("Renderer wrapper test passed.")


if __name__ == "__main__":
    main()
