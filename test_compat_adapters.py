import numpy as np

from core import MLXCameraAdapter, MLXTriangleModelAdapter


class FakeTorchLikeTensor:
    def __init__(self, value):
        self._value = np.array(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._value

    def item(self):
        return self._value.item()


def main():
    camera_dict = {
        "FoVx": FakeTorchLikeTensor(1.0),
        "FoVy": FakeTorchLikeTensor(1.0),
        "image_height": FakeTorchLikeTensor(32),
        "image_width": FakeTorchLikeTensor(32),
        "world_view_transform": FakeTorchLikeTensor(np.eye(4, dtype=np.float32)),
        "full_proj_transform": FakeTorchLikeTensor(np.eye(4, dtype=np.float32)),
        "camera_center": FakeTorchLikeTensor(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
    }
    model_dict = {
        "active_sh_degree": FakeTorchLikeTensor(3),
        "get_triangles_points": FakeTorchLikeTensor(
            np.array(
                [
                    [[-0.5, -0.5, 1.0], [0.5, -0.5, 1.0], [0.0, 0.5, 1.0]],
                    [[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.0, 0.5, 2.0]],
                ],
                dtype=np.float32,
            )
        ),
        "get_sigma": FakeTorchLikeTensor(np.array([1.0, 1.0], dtype=np.float32)),
        "get_num_points_per_triangle": FakeTorchLikeTensor(np.array([3, 3], dtype=np.int32)),
        "get_cumsum_of_points_per_triangle": FakeTorchLikeTensor(np.array([0, 3], dtype=np.int32)),
        "get_number_of_points": FakeTorchLikeTensor(2),
        "get_opacity": FakeTorchLikeTensor(np.array([0.8, 0.6], dtype=np.float32)),
        "get_features": FakeTorchLikeTensor(np.zeros((2, 16, 3), dtype=np.float32)),
    }

    camera = MLXCameraAdapter.from_object(camera_dict)
    model = MLXTriangleModelAdapter.from_object(model_dict)

    print("camera image size:", camera.image_width, camera.image_height)
    print("camera center:", np.array(camera.camera_center).tolist())
    print("triangle points shape:", list(model.get_triangles_points.shape))
    print("sigma shape:", list(model.get_sigma.shape))
    print("active_sh_degree:", model.active_sh_degree)

    assert camera.image_width == 32
    assert camera.image_height == 32
    assert list(model.get_triangles_points.shape) == [2, 3, 3]
    assert list(model.get_sigma.shape) == [2]
    assert model.active_sh_degree == 3

    print("Compatibility adapter test passed.")


if __name__ == "__main__":
    main()
