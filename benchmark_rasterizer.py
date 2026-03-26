import argparse
import time

import mlx.core as mx
import numpy as np

from core.mlx_rasterizer.rasterizer import TriangleRasterizationSettings, TriangleRasterizer


def build_random_scene(num_triangles: int, sh_degree: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    points_per_triangle = 3
    max_coeffs = (sh_degree + 1) ** 2

    centers_xy = mx.random.uniform(low=-1.0, high=1.0, shape=(num_triangles, 2))
    centers_z = mx.random.uniform(low=1.0, high=4.0, shape=(num_triangles, 1))
    centers = mx.concatenate([centers_xy, centers_z], axis=1)

    base_shape = mx.array(
        [
            [-1.0, -0.7, 0.0],
            [1.0, -0.6, 0.0],
            [0.0, 0.9, 0.0],
        ],
        dtype=mx.float32,
    )
    radii = mx.random.uniform(low=0.03, high=0.12, shape=(num_triangles, 1, 1))
    jitter = mx.random.normal(shape=(num_triangles, points_per_triangle, 3)) * 0.03
    triangles_points = mx.expand_dims(centers, axis=1) + radii * mx.expand_dims(base_shape, axis=0) + jitter

    sigma = mx.random.uniform(low=0.8, high=1.4, shape=(num_triangles,))
    opacities = mx.random.uniform(low=0.4, high=0.95, shape=(num_triangles,))
    shs = mx.random.normal(shape=(num_triangles, max_coeffs, 3)) * 0.12

    return triangles_points, sigma, opacities, shs


def build_rasterizer(width: int, height: int, sh_degree: int) -> TriangleRasterizer:
    settings = TriangleRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=mx.array([0.02, 0.02, 0.02]),
        scale_modifier=1.0,
        viewmatrix=mx.eye(4),
        projmatrix=mx.eye(4),
        sh_degree=sh_degree,
        campos=mx.array([0.0, 0.0, 0.0]),
        prefiltered=False,
        debug=False,
    )
    return TriangleRasterizer(settings)


def forward_pass(
    rasterizer: TriangleRasterizer,
    triangles_points: mx.array,
    sigma: mx.array,
    opacities: mx.array,
    shs: mx.array,
) -> mx.array:
    num_triangles = triangles_points.shape[0]
    num_points_per_triangle = mx.full((num_triangles,), 3, dtype=mx.int32)
    cumsum_of_points_per_triangle = mx.arange(0, num_triangles * 3, 3, dtype=mx.int32)
    return rasterizer(
        triangles_points=triangles_points,
        sigma=sigma,
        num_points_per_triangle=num_points_per_triangle,
        cumsum_of_points_per_triangle=cumsum_of_points_per_triangle,
        number_of_points=num_triangles,
        opacities=opacities,
        means2D=mx.zeros((num_triangles, 2)),
        scaling=mx.zeros((num_triangles,)),
        density_factor=mx.zeros((num_triangles,)),
        shs=shs,
    )


def benchmark_forward(
    rasterizer: TriangleRasterizer,
    triangles_points: mx.array,
    sigma: mx.array,
    opacities: mx.array,
    shs: mx.array,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        image = forward_pass(rasterizer, triangles_points, sigma, opacities, shs)
        mx.eval(image)

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        image = forward_pass(rasterizer, triangles_points, sigma, opacities, shs)
        mx.eval(image)
        timings.append(time.perf_counter() - start)

    return float(np.mean(timings)), float(np.min(timings))


def benchmark_backward(
    rasterizer: TriangleRasterizer,
    triangles_points: mx.array,
    sigma: mx.array,
    opacities: mx.array,
    shs: mx.array,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    def loss_fn(tp):
        image = forward_pass(rasterizer, tp, sigma, opacities, shs)
        return mx.sum(image)

    value_and_grad = mx.value_and_grad(loss_fn)

    for _ in range(warmup):
        value, grad = value_and_grad(triangles_points)
        mx.eval(value, grad)

    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        value, grad = value_and_grad(triangles_points)
        mx.eval(value, grad)
        timings.append(time.perf_counter() - start)

    return float(np.mean(timings)), float(np.min(timings))


def main():
    parser = argparse.ArgumentParser(description="Benchmark the MLX triangle rasterizer.")
    parser.add_argument("--triangles", type=int, default=256)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()

    mx.random.seed(0)

    rasterizer = build_rasterizer(args.width, args.height, args.sh_degree)
    triangles_points, sigma, opacities, shs = build_random_scene(args.triangles, args.sh_degree)

    mean_forward_s, min_forward_s = benchmark_forward(
        rasterizer,
        triangles_points,
        sigma,
        opacities,
        shs,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    state = rasterizer.last_tiling_state
    visible_triangles = int(np.array(state["triangle_indices"]).shape[0])
    overlaps = int(np.array(state["overlap_tile_ids"]).shape[0])
    pixels = args.width * args.height

    print("Forward benchmark")
    print(f"triangles: {args.triangles}")
    print(f"image: {args.width}x{args.height}")
    print(f"visible_triangles: {visible_triangles}")
    print(f"triangle_tile_overlaps: {overlaps}")
    print(f"pixels: {pixels}")
    print(f"mean_forward_ms: {mean_forward_s * 1000.0:.3f}")
    print(f"min_forward_ms: {min_forward_s * 1000.0:.3f}")
    print(f"triangles_per_s: {args.triangles / mean_forward_s:.1f}")

    if args.backward:
        mean_backward_s, min_backward_s = benchmark_backward(
            rasterizer,
            triangles_points,
            sigma,
            opacities,
            shs,
            warmup=max(1, args.warmup // 2),
            repeats=args.repeats,
        )
        print("Backward benchmark")
        print(f"mean_backward_ms: {mean_backward_s * 1000.0:.3f}")
        print(f"min_backward_ms: {min_backward_s * 1000.0:.3f}")


if __name__ == "__main__":
    main()
