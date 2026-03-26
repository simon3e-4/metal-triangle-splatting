import math

import mlx.core as mx
import numpy as np

from core import adapt_reference_camera_snapshot, adapt_reference_model_snapshot, render


class DummyPipe:
    debug = False


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


def build_camera_snapshot():
    return {
        "FoVx": np.float32(1.0),
        "FoVy": np.float32(1.0),
        "original_image": np.zeros((3, 32, 32), dtype=np.float32),
        "world_view_transform": np.eye(4, dtype=np.float32),
        "projection_matrix": np.eye(4, dtype=np.float32),
    }


def build_model_snapshot():
    point_cloud_state_dict = {
        "triangles_points": np.array(
            [
                [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]],
                [[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.0, 0.5, 2.0]],
            ],
            dtype=np.float32,
        ),
        "sigma": np.array([[math.log(0.99)], [math.log(1.49)]], dtype=np.float32),
        "active_sh_degree": np.int32(3),
        "features_dc": np.zeros((2, 1, 3), dtype=np.float32),
        "features_rest": np.zeros((2, 15, 3), dtype=np.float32),
        "opacity": np.array([[_logit(0.8)], [_logit(0.6)]], dtype=np.float32),
    }
    hyperparameters = {
        "num_points_per_triangle": np.array([3, 3], dtype=np.int32),
        "cumsum_of_points_per_triangle": np.array([0, 3], dtype=np.int32),
        "number_of_points": np.int32(2),
    }
    return point_cloud_state_dict, hyperparameters


def assert_render_result(result):
    image = np.array(result["render"])
    visibility = np.array(result["visibility_filter"])
    indices = np.array(result["triangle_indices"])
    radii = np.array(result["radii"])
    scaling = np.array(result["scaling"])
    density_factor = np.array(result["density_factor"])
    max_blending = np.array(result["max_blending"])
    rend_alpha = np.array(result["rend_alpha"])
    surf_depth = np.array(result["surf_depth"])

    print("render shape:", list(image.shape))
    print("num_visible:", int(result["num_visible"]))
    print("visibility_filter:", visibility.tolist())
    print("triangle_indices:", indices.tolist())

    assert image.shape == (3, 32, 32)
    assert int(result["num_visible"]) == 6
    assert visibility.tolist() == [True, True]
    assert indices.tolist() == [0, 1]
    assert radii.shape == (2,)
    assert scaling.shape == (2,)
    assert density_factor.shape == (2,)
    assert max_blending.shape == (2,)
    assert rend_alpha.shape == (1, 32, 32)
    assert surf_depth.shape == (1, 32, 32)
    assert float(rend_alpha.sum()) > 0.0
    assert float(surf_depth.sum()) > 0.0


def main():
    camera_snapshot = build_camera_snapshot()
    point_cloud_state_dict, hyperparameters = build_model_snapshot()

    camera = adapt_reference_camera_snapshot(camera_snapshot)
    model = adapt_reference_model_snapshot(point_cloud_state_dict, hyperparameters)

    print("camera image size:", camera.image_width, camera.image_height)
    print("camera center:", np.array(camera.camera_center).tolist())
    print("triangle points shape:", list(model.get_triangles_points.shape))
    print("features shape:", list(model.get_features.shape))
    print("sigma:", np.array(model.get_sigma).reshape(-1).tolist())
    print("opacity:", np.array(model.get_opacity).reshape(-1).tolist())

    assert camera.image_width == 32
    assert camera.image_height == 32
    assert np.allclose(np.array(camera.camera_center), np.array([0.0, 0.0, 0.0], dtype=np.float32))
    assert list(model.get_triangles_points.shape) == [2, 3, 3]
    assert list(model.get_features.shape) == [2, 16, 3]
    assert np.allclose(np.array(model.get_sigma).reshape(-1), np.array([1.0, 1.5], dtype=np.float32), atol=1e-5)
    assert np.allclose(np.array(model.get_opacity).reshape(-1), np.array([0.8, 0.6], dtype=np.float32), atol=1e-5)

    nested_result = render(
        viewpoint_camera=camera_snapshot,
        pc={"point_cloud_state_dict": point_cloud_state_dict, "hyperparameters": hyperparameters},
        pipe=DummyPipe(),
        bg_color=np.array([0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert_render_result(nested_result)

    tuple_result = render(
        viewpoint_camera=camera_snapshot,
        pc=(point_cloud_state_dict, hyperparameters),
        pipe=None,
        bg_color=mx.array([0.0, 0.0, 0.0], dtype=mx.float32),
    )
    assert_render_result(tuple_result)

    print("Reference snapshot test passed.")


if __name__ == "__main__":
    main()
