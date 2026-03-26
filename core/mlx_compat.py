from dataclasses import dataclass
from typing import Mapping
from typing import Any

import mlx.core as mx
import numpy as np


def to_mx_array(value: Any, dtype=None) -> mx.array:
    if isinstance(value, mx.array):
        return value.astype(dtype) if dtype is not None else value

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()

    array = mx.array(value)
    return array.astype(dtype) if dtype is not None else array


def to_python_scalar(value: Any):
    if isinstance(value, (int, float, bool, str)):
        return value
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    return np.asarray(value).item()


def _has_field(obj: Any, name: str) -> bool:
    return (isinstance(obj, Mapping) and name in obj) or hasattr(obj, name)


def _read_field(obj: Any, name: str):
    value = obj[name] if isinstance(obj, Mapping) else getattr(obj, name)
    return value() if callable(value) else value


def _reference_sigma_activation(raw_sigma: Any) -> mx.array:
    return 0.01 + mx.exp(to_mx_array(raw_sigma, dtype=mx.float32))


def _reference_opacity_activation(raw_opacity: Any) -> mx.array:
    return mx.sigmoid(to_mx_array(raw_opacity, dtype=mx.float32))


def _combine_reference_features(features_dc: Any, features_rest: Any | None) -> mx.array:
    dc = to_mx_array(features_dc, dtype=mx.float32)
    if features_rest is None:
        return dc
    rest = to_mx_array(features_rest, dtype=mx.float32)
    return mx.concatenate([dc, rest], axis=1)


@dataclass
class MLXCameraAdapter:
    FoVx: float
    FoVy: float
    image_height: int
    image_width: int
    world_view_transform: mx.array
    full_proj_transform: mx.array
    camera_center: mx.array

    @classmethod
    def from_object(cls, camera: Any) -> "MLXCameraAdapter":
        if isinstance(camera, cls):
            return camera

        world_view_transform = to_mx_array(_read_field(camera, "world_view_transform"), dtype=mx.float32)
        if _has_field(camera, "full_proj_transform"):
            full_proj_transform = to_mx_array(_read_field(camera, "full_proj_transform"), dtype=mx.float32)
        elif _has_field(camera, "projection_matrix"):
            projection_matrix = to_mx_array(_read_field(camera, "projection_matrix"), dtype=mx.float32)
            full_proj_transform = mx.matmul(world_view_transform, projection_matrix)
        else:
            raise AttributeError("Camera adapter expects `full_proj_transform` or `projection_matrix`.")

        if _has_field(camera, "image_height") and _has_field(camera, "image_width"):
            image_height = int(to_python_scalar(_read_field(camera, "image_height")))
            image_width = int(to_python_scalar(_read_field(camera, "image_width")))
        elif _has_field(camera, "original_image"):
            original_image = to_mx_array(_read_field(camera, "original_image"), dtype=mx.float32)
            image_height = int(original_image.shape[1])
            image_width = int(original_image.shape[2])
        else:
            raise AttributeError("Camera adapter expects `image_height`/`image_width` or `original_image`.")

        if _has_field(camera, "camera_center"):
            camera_center = to_mx_array(_read_field(camera, "camera_center"), dtype=mx.float32)
        else:
            world_view_np = np.array(world_view_transform, dtype=np.float32)
            camera_center = mx.array(np.linalg.inv(world_view_np)[3, :3], dtype=mx.float32)

        return cls(
            FoVx=float(to_python_scalar(_read_field(camera, "FoVx"))),
            FoVy=float(to_python_scalar(_read_field(camera, "FoVy"))),
            image_height=image_height,
            image_width=image_width,
            world_view_transform=world_view_transform,
            full_proj_transform=full_proj_transform,
            camera_center=camera_center,
        )

    @classmethod
    def from_reference_snapshot(cls, camera_snapshot: Any) -> "MLXCameraAdapter":
        return cls.from_object(camera_snapshot)


@dataclass
class MLXTriangleModelAdapter:
    active_sh_degree: int
    get_triangles_points: mx.array
    get_sigma: mx.array
    get_num_points_per_triangle: mx.array
    get_cumsum_of_points_per_triangle: mx.array
    get_number_of_points: int
    get_opacity: mx.array
    get_features: mx.array | None = None

    @classmethod
    def from_object(cls, model: Any) -> "MLXTriangleModelAdapter":
        if isinstance(model, cls):
            return model

        if isinstance(model, Mapping):
            if "point_cloud_state_dict" in model and "hyperparameters" in model:
                return cls.from_reference_snapshot(model["point_cloud_state_dict"], model["hyperparameters"])
            if "triangles_points" in model and "num_points_per_triangle" in model:
                return cls.from_reference_snapshot(model, model)

        features = _read_field(model, "get_features") if _has_field(model, "get_features") else None
        triangles_points = (
            _read_field(model, "get_triangles_points")
            if _has_field(model, "get_triangles_points")
            else _read_field(model, "get_triangles_points_flatten")
        )

        return cls(
            active_sh_degree=int(to_python_scalar(_read_field(model, "active_sh_degree"))),
            get_triangles_points=to_mx_array(triangles_points, dtype=mx.float32),
            get_sigma=to_mx_array(_read_field(model, "get_sigma"), dtype=mx.float32),
            get_num_points_per_triangle=to_mx_array(_read_field(model, "get_num_points_per_triangle"), dtype=mx.int32),
            get_cumsum_of_points_per_triangle=to_mx_array(_read_field(model, "get_cumsum_of_points_per_triangle"), dtype=mx.int32),
            get_number_of_points=int(to_python_scalar(_read_field(model, "get_number_of_points"))),
            get_opacity=to_mx_array(_read_field(model, "get_opacity"), dtype=mx.float32),
            get_features=to_mx_array(features, dtype=mx.float32) if features is not None else None,
        )

    @classmethod
    def from_reference_snapshot(
        cls,
        point_cloud_state_dict: Mapping[str, Any],
        hyperparameters: Mapping[str, Any],
    ) -> "MLXTriangleModelAdapter":
        features_rest = _read_field(point_cloud_state_dict, "features_rest") if _has_field(point_cloud_state_dict, "features_rest") else None
        return cls(
            active_sh_degree=int(to_python_scalar(_read_field(point_cloud_state_dict, "active_sh_degree"))),
            get_triangles_points=to_mx_array(_read_field(point_cloud_state_dict, "triangles_points"), dtype=mx.float32),
            get_sigma=_reference_sigma_activation(_read_field(point_cloud_state_dict, "sigma")),
            get_num_points_per_triangle=to_mx_array(_read_field(hyperparameters, "num_points_per_triangle"), dtype=mx.int32),
            get_cumsum_of_points_per_triangle=to_mx_array(_read_field(hyperparameters, "cumsum_of_points_per_triangle"), dtype=mx.int32),
            get_number_of_points=int(to_python_scalar(_read_field(hyperparameters, "number_of_points"))),
            get_opacity=_reference_opacity_activation(_read_field(point_cloud_state_dict, "opacity")),
            get_features=_combine_reference_features(
                _read_field(point_cloud_state_dict, "features_dc"),
                features_rest,
            ),
        )


def adapt_reference_camera_snapshot(camera_snapshot: Any) -> MLXCameraAdapter:
    return MLXCameraAdapter.from_reference_snapshot(camera_snapshot)


def adapt_reference_model_snapshot(
    point_cloud_state_dict: Mapping[str, Any],
    hyperparameters: Mapping[str, Any],
) -> MLXTriangleModelAdapter:
    return MLXTriangleModelAdapter.from_reference_snapshot(point_cloud_state_dict, hyperparameters)
