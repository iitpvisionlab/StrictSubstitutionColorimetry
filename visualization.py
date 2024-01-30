#!/usr/bin/env python3
from __future__ import annotations
from typing import Iterator, TypeAlias, Literal, NamedTuple, TypedDict
from typing_extensions import NotRequired
from pathlib import Path
from json import load
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
from statistics import median as calc_median, StatisticsError
import numpy as np
import numpy.typing as npt
import scipy
from math import radians, cos, sin

from benchmarking import calc_median_measurements


FArray: TypeAlias = npt.NDArray[np.float_]
Distance = None | float | Literal["Failure"]


class xyYNone(NamedTuple):
    x: None = None
    y: None = None
    Y: None = None


class xyY(NamedTuple):
    x: float
    y: float
    Y: float


class Measurement(TypedDict):
    x: float
    y: float
    Y: float
    Yc: NotRequired[float]
    angle: int
    angle2: int
    distance: Distance
    mode: Literal["substitution", "calibration", "scd/lcd"]
    dt: NotRequired[str]


class MeasurementsJson(TypedDict):
    username: str
    age: str
    measurements: list[Measurement]


def shift_center(m: Measurement, distance: float) -> xyY:
        angle = radians(m["angle"])
        angle2 = radians(m.get("angle2", 0))
        x, y, Y = m["x"], m["y"], m["Y"]
        distance1 = distance * cos(angle2)
        x1 = x + cos(angle) * distance1
        y1 = y + sin(angle) * distance1
        Yc = m.get("Yc", 100.0)
        Y1 = Y + distance * Yc * sin(angle2)
        return xyY(x1, y1, Y1)


def calc_median_color(colors: list[xyY]) -> xyY | xyYNone:
    x, y, Y = zip(*colors)
    try:
        return xyY(
            x=calc_median([val for val in x if val is not None]),
            y=calc_median([val for val in y if val is not None]),
            Y=calc_median([val for val in Y if val is not None]),
        )
    except StatisticsError:
        return xyYNone()


def ellipsoids_fitting(x: FArray, y: FArray, Y: FArray) -> FArray:
    A = np.asfarray(
        [
            x**2,
            y**2,
            Y**2,
            2 * x * y,
            2 * x * Y,
            2 * y * Y,
        ]
    )
    q = np.linalg.inv(A @ A.T) @ A.sum(axis=1)
    Q = np.asfarray(
        [
            [q[0], q[3], q[4]],
            [q[3], q[1], q[5]],
            [q[4], q[5], q[2]],
        ]
    )
    if np.linalg.eig(Q)[0].min() <= 0.0:
        raise ValueError("The regression surface is not an ellipsoid")
    return Q


def measurement_to_xyY(m: Measurement) -> xyY | xyYNone:
    if m["distance"] == "Failure":
        return xyYNone()
    return shift_center(m, m["distance"])


def group_measurements(
    ms: list[Measurement], yield_all_tries: bool = False
) -> Iterator[tuple[xyY, list[float], list[float], list[float]]]:
    print(len(ms["color_centers"]))
    all_centers = [m for m in ms["color_centers"]]
    for c in all_centers:
        measurements = c["measurements"]
        sorted_m = sorted(measurements, key=lambda k: (k['angle_1'], k['angle_2']))
        directions = {}
        for m in sorted_m:
            if f"{m['angle_1']}_{m['angle_2']}" not in directions :
                directions[f"{m['angle_1']}_{m['angle_2']}"] = []
            directions[f"{m['angle_1']}_{m['angle_2']}"].append([m["x"], m["y"], m["Y"]])
        median_colors = calc_median_measurements(directions)
        xs = ys = Ys = []
        for mc in median_colors:
            xs.append(mc[0])
            ys.append(mc[0])
            Ys.append(mc[0])
        yield [c["center_x"], c["center_y"], c["center_Y"]], xs, ys, Ys, c["ellipsoid_matrix"]


def get_xyY_pairs_angles(centers: list) -> list[list[float]]:
    xyY_user: list[list[float]] = []
    angles = []
    for c in centers:
        valid_measurements = [m for m in c["measurements"] if m['angle_2'] == 0]
        sorted_m = sorted(valid_measurements, key=lambda k: (k['angle_1'], k['angle_2']))
        directions = {}
        for m in sorted_m:
            if f"{m['angle_1']}_{m['angle_2']}" not in directions :
                directions[f"{m['angle_1']}_{m['angle_2']}"] = []
            directions[f"{m['angle_1']}_{m['angle_2']}"].append([m["x"], m["y"], m["Y"]])
        median_colors = calc_median_measurements(directions)
        for mc in median_colors:
            xyY_user.append([[c["center_x"], c["center_y"], c["center_Y"]], mc])
        angles.extend(directions.keys())

    return np.moveaxis(np.asfarray(xyY_user), 0, 1), angles


def measurements_to_shape(
    ms: list[Measurement], yield_all_tries: bool = False
) -> Iterator[tuple[xyY, FArray | None, FArray, FArray, FArray]]:
    for center, xs, ys, Ys, q in group_measurements(ms, yield_all_tries):
        yield center, q, np.asfarray(xs), np.asfarray(ys), np.asfarray(Ys)


from colour.plotting import (
    plot_chromaticity_diagram_CIE1931,
)


def plot_ellipsoid(ax, center: xyY, q: FArray):
    N = 61
    stride = 2
    u = np.linspace(0, 2 * np.pi, N)
    v = np.linspace(0, np.pi, N)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    S = np.asfarray(center) + np.dstack((x, y, z)) @ np.linalg.inv(
        scipy.linalg.cholesky(q).T
    )
    x, y, z = S.T

    ax.plot_surface(
        x,
        y,
        z,
        linewidth=21.2,
        cstride=stride,
        rstride=stride,
        color=cie_xyY_to_somewhat_rgb(center),
        alpha=0.5,
    )


def plot_ellipse(ax, center: xyY, q: FArray, plane: Literal["xy", "xY", "yY"], linestyle):
    if q is None:
        print("no q")
        return
    N = 41
    u = np.linspace(0, 2 * np.pi, N)
    if plane == "xy":
        x = np.cos(u)[..., None]
        y = np.sin(u)[..., None]
        z = np.zeros_like(x)
    elif plane == "xY":
        x = np.cos(u)[..., None]
        y = np.zeros_like(x)
        z = np.sin(u)[..., None]
    elif plane == "yY":
        y = np.sin(u)[..., None]
        z = np.cos(u)[..., None]
        x = np.zeros_like(y)

    S = np.asfarray(center) + np.dstack((x, y, z)) @ np.linalg.inv(
        scipy.linalg.cholesky(q).T
    )
    x, y, z = S.T
    if plane == "xy":
        pass
    elif plane == "xY":
        x, y = x, z
    elif plane == "yY":
        x, y = y, z

    ax.plot(
        x[0],
        y[0],
        linewidth=1.5,
        color="0",  # cie_xyY_to_somewhat_rgb(center),
        alpha=1,
        linestyle=linestyle
    )


LIN_RGB_MATRIX = np.asfarray(
    (
        (3.2404542, -1.5371385, -0.4985314),
        (-0.9692660, 1.8760108, 0.0415560),
        (0.0556434, -0.2040259, 1.0572252),
    )
).T


def cie_xyY_to_somewhat_rgb(x_y_Y_input: xyY) -> str:
    x_y_Y = xyY(x=x_y_Y_input[0], y=x_y_Y_input[1], Y=0.4)
    Y_div_y = x_y_Y[2] / x_y_Y[1]
    XYZ = x_y_Y[0] * Y_div_y, x_y_Y[2], (1.0 - x_y_Y[0] - x_y_Y[1]) * Y_div_y
    linRGB = np.asfarray(XYZ) @ LIN_RGB_MATRIX

    thres = 0.0031308
    a = 0.055

    linRGB = linRGB.clip(0, 10000)
    color_clipped = linRGB / linRGB.max()  # experimenting
    color_clipped_f = color_clipped.reshape(-1)

    low = color_clipped_f <= thres

    color_clipped_f[low] *= 12.92
    color_clipped_f[~low] = (1 + a) * color_clipped_f[~low] ** (1 / 2.4) - a

    r, g, b = color_clipped
    ret = f"#{round(r*255.0):02x}{round(g*255.0):02x}{round(b*255.0):02x}"
    return ret


# Point: TypeAlias = tuple[float, float]
class Point(NamedTuple):
    x: float
    y: float


def sort_points(x: FArray, y: FArray):
    def less(a: Point, b: Point) -> bool:
        if a.x - center.x >= 0 and b.x - center.x < 0:
            return True
        if a.x - center.x < 0 and b.x - center.x >= 0:
            return False
        if a.x - center.x == 0 and b.x - center.x == 0:
            if a.y - center.y >= 0 or b.y - center.y >= 0:
                return a.y > b.y
            return b.y > a.y

        # compute the cross product of vectors (center -> a) x (center -> b)
        det = (a.x - center.x) * (b.y - center.y) - (b.x - center.x) * (
            a.y - center.y
        )
        if det < 0:
            return True
        if det > 0:
            return False

        # points a and b are on the same line from the center
        # check which point is closer to the center
        d1 = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (
            a.y - center.y
        )
        d2 = (b.x - center.x) * (b.x - center.x) + (b.y - center.y) * (
            b.y - center.y
        )
        return d1 > d2

    from functools import cmp_to_key

    center = Point(x.mean(), y.mean())
    points = [Point(*_) for _ in zip(x, y)]
    points.sort(key=cmp_to_key(lambda a, b: -1 if less(a, b) else 1))
    return zip(*points)


def visualize_3d(
    data: list[MeasurementsJson],
    plot_tries: bool,
    plot_fixed8: bool,
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.0, 1.0, 2.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Y")
    # ax.grid()
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.set_zlim(0, 220)
    plane = "xY"
    if plane == "xy":
        ax.view_init(elev=90.0, azim=-90.0)
    elif plane == "xY":
        ax.view_init(elev=0.0, azim=-90.0)
    elif plane == "yY":
        ax.view_init(elev=0.0, azim=0.0)

    for user in data:
        for center, q, xs, ys, Ys in measurements_to_shape(
            user, plot_tries
        ):
            if q is not None:
                # ellipsoid can be plotted
                plot_ellipsoid(ax, center, q)
                ax.text(xs.max(), ys.max(), center[2], f"{center[2]:0.1f}", None)
            else:
                ax.scatter(xs, ys, Ys, marker="^", color="0.2", alpha=0.4)


def visualize_2d(
    data: list[MeasurementsJson],
    plane: Literal["xy", "xY", "yY"],
    plot_tries: bool,
    plot_fixed8: bool,
) -> None:
    assert plane in ["xy", "xY", "yY"], plane
    fig, ax = plt.subplots(dpi=150, figsize=(5.0, 5.0), ncols=1)

    if plane == "xy":
        plot_chromaticity_diagram_CIE1931(
            "CIE 1964 10 Degree Standard Observer",
            bounding_box=(-0.025, 0.8, -0.015, 0.85),
            standalone=False,
            axes=ax,
        )

    ax.set_xlabel(plane[0])
    ax.set_ylabel(plane[1])
    ax.grid()

    print(f"Users: {len(data)}")
    markers = ["o", "v", "s", "X"]
    lines = ['solid', 'dotted', 'dashed', 'dashdot']
    marker_colors = ["0", "#8c801b", "#9a15d4", "0.5"]
    for idx, user in enumerate(data):

        def plot(plot_tries: bool, color: str, **kwargs: str):
            c_to_q = {}
            for center, q, xs, ys, Ys in measurements_to_shape(
                user, yield_all_tries=plot_tries
            ):
                center_color = cie_xyY_to_somewhat_rgb(center)
                if plane == "xy":
                    x, y = xs, ys
                    ax.plot([center[0]], [center[1]], "o", color="0.4", markersize=4)
                    ax.set_title("")
                elif plane == "xY":
                    x, y = xs, Ys
                    ax.plot([center[0]], [center[2]], "o", color=center_color, markersize=4)
                elif plane == "yY":
                    x, y = ys, Ys
                    ax.plot([center[1]], [center[2]], "o", color=center_color, markersize=4)
                else:
                    assert False
                if not plot_tries:
                    plot_ellipse(ax, center, q, plane, lines[idx])
                x_sorted, y_sorted = sort_points(x, y)
                x_sorted += (x_sorted[0],)
                y_sorted += (y_sorted[0],)

        if plot_tries:
            plot(plot_tries, "0.3", linestyle="none")
        plot(False, marker_colors[idx], linestyle="none", alpha=0.9)


def main():
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("json_paths", nargs="+", type=Path)
    parser.add_argument(
        "--plane", nargs="+", choices=["3d", "xy", "xY", "yY"], default=["3d"]
    )
    parser.add_argument(
        "--try",
        dest="plot_tries",
        action="store_true",
        help="plot each try instead of median",
    )
    parser.add_argument(
        "--fixed8",
        action="store_true",
        help="only plot fixed 8 centers",
    )
    args = parser.parse_args()

    data: list[MeasurementsJson] = []
    for json_path in args.json_paths:
        with json_path.open() as f:
            data.append(load(f))

    for plane in args.plane:
        match plane:
            case "3d":
                visualize_3d(data, args.plot_tries, args.fixed8)
            case _:
                visualize_2d(data, plane, args.plot_tries, args.fixed8)

    plt.title("")
    plt.show()


if __name__ == "__main__":
    main()
