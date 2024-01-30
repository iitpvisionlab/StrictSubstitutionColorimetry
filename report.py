from __future__ import annotations
import numpy as np
from numpy.linalg import norm
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy.typing as npt
from copy import deepcopy
from warnings import warn
from typing import Callable, List
from functools import partial

import colour
import inspect


D65 = np.array([0.95047, 1, 1.08883])



def check_xyz(xyz: np.array):
    magnitude_order = 1
    if type(xyz) != np.ndarray:
        raise TypeError(f"'xyz' type must be np.ndarray, not {type(xyz)}")
    max_value = np.max(xyz)
    if max_value > magnitude_order * 2:
        warn(
            f"order of magnitude for CIE XYZ coordinates must be {magnitude_order}. "
            f"Array with maximum element {max_value} is specified"
        )


def XYZ_to_CAM_UCS(
    xyz: np.array,
    type_of_CAM="CAM16",
    type_of_UCS="UCS",
    illuminant=D65,
    L_A=80,
    Y_b=50,
    surround=colour.appearance.VIEWING_CONDITIONS_CAM16["Average"],
):
    check_xyz(xyz)
    check_xyz(illuminant)
    assert type_of_CAM in ["CAM16", "CAM02"]
    assert type_of_UCS in ["UCS", "LCD", "SCD"]
    xyz = xyz * 100.0
    ref_illum_XYZ = illuminant * 100.0

    CAM_fun = {"CAM16": colour.XYZ_to_CAM16, "CAM02": colour.XYZ_to_CIECAM02}[
        type_of_CAM
    ]
    UCS_fun = {
        "CAM16": {
            "UCS": colour.JMh_CAM16_to_CAM16UCS,
            "LCD": colour.JMh_CAM16_to_CAM16LCD,
            "SCD": colour.JMh_CAM16_to_CAM16SCD,
        },
        "CAM02": {
            "UCS": colour.JMh_CIECAM02_to_CAM02UCS,
            "LCD": colour.JMh_CIECAM02_to_CAM02LCD,
            "SCD": colour.JMh_CIECAM02_to_CAM02SCD,
        },
    }[type_of_CAM][type_of_UCS]
    K_L = {"UCS": 1.00, "LCD": 0.77, "SCD": 1.24}

    def _XYZ_to_CAM_UCS(xyz_points):
        cam = CAM_fun(xyz_points, ref_illum_XYZ, Y_b=Y_b, L_A=L_A, surround=surround)
        # JMh = np.array([cam.J, cam.M, cam.h])
        # JMh = JMh.transpose()
        JMh = np.stack([cam.J, cam.M, cam.h], axis=-1)
        Jab = UCS_fun(JMh)
        Jab[..., 0] = Jab[..., 0] / K_L[type_of_UCS]
        return Jab

    return _XYZ_to_CAM_UCS(xyz)


XYZ_to_CAM16UCS = partial(
    XYZ_to_CAM_UCS,
    type_of_CAM="CAM16",
    type_of_UCS="UCS",
)


def convert_color_paris_to_CS(
    xyz_list1, xyz_list2, white_point, La, Yb, cs_name, cs_fun, patch_size=None
):
    crd = np.array(
        [
            convert_to_CS(
                xyz_list1, white_point, La, Yb, cs_name, cs_fun, patch_size=patch_size
            ),
            convert_to_CS(
                xyz_list2, white_point, La, Yb, cs_name, cs_fun, patch_size=patch_size
            ),
        ]
    )
    if crd.ndim == 2:
        crd = crd.reshape(crd.shape[0], 1, crd.shape[1])
    return crd


def convert_to_CS(
    xyz_list,
    white_point,
    La,
    Yb,
    cs_name,
    cs_fun,
    surround_name="Average",
    patch_size=None,
):
    func_args = inspect.getfullargspec(cs_fun)
    if "illuminant" in func_args[0] or "illuminant" in func_args[4]:
        if cs_name.find("sdf") != -1:
            return np.array(
                cs_fun(xyz_list, illuminant=white_point, patch_size=patch_size)
            )
        if cs_name.find("BC-CAT") != -1:
            F = colour.appearance.VIEWING_CONDITIONS_CAM16[surround_name].F
            return np.array(cs_fun(xyz_list, illuminant=white_point, F=F, L_A=La))
        if (
            cs_name == "HCT"
            or cs_name.find("proLab") != -1
            or cs_name.find("PolyS") != -1
            or cs_name.find("ConeS") != -1
        ):
            return np.array(cs_fun(xyz_list, illuminant=white_point))
        if cs_name in ["CIELAB", "CIEDE2000", "CIE xyY", "CIE LUV"]:
            return np.array(cs_fun(xyz_list, illuminant=colour.XYZ_to_xy(white_point)))
        if cs_name.find("CAM") != -1:
            surround = colour.appearance.VIEWING_CONDITIONS_CAM16[surround_name]
            return np.array(
                cs_fun(
                    xyz_list, illuminant=white_point, L_A=La, Y_b=Yb, surround=surround
                )
            )
        return np.array(cs_fun(xyz_list, illuminant=white_point))
    return np.array(cs_fun(xyz_list))


def calculate_metric(
    xyz_list1, xyz_list2, white_point, La, Yb, cs_name, cs_fun, patch_size=None
):

    if cs_name == "CIEDE2000":
        crd = convert_color_paris_to_CS(
            xyz_list1, xyz_list2, white_point, La, Yb, cs_name, cs_fun
        )
        return colour.difference.delta_E_CIE2000(crd[0], crd[1])
    if cs_name in ["HyAB", "cbLAB", "HyCH", "cbLCH"]:
        assert cs_name in ["HyAB", "HyCH"], "not tested for cbLAB, cbLCH"
        wp_xyY = colour.XYZ_to_xyY(white_point)
        fun = {"HyAB": HyAB, "HyCH": HyCH}[cs_name]
        lab1 = colour.XYZ_to_Lab(xyz_list1, wp_xyY)
        lab2 = colour.XYZ_to_Lab(xyz_list2, wp_xyY)
        diff = fun(lab1, lab2)
        return diff
    crd = convert_color_paris_to_CS(
        xyz_list1,
        xyz_list2,
        white_point,
        La,
        Yb,
        cs_name,
        cs_fun,
        patch_size=patch_size,
    )
    if cs_name.find("power-cor") != -1:
        return 1.41 * (np.linalg.norm(crd[0] - crd[1], axis=1) ** 0.63)
    return np.linalg.norm(crd[0] - crd[1], axis=-1)


def get_group_stress(
    x_list: List[npt.NDArray[np.float_]],
    y_list: List[npt.NDArray[np.float_]],
    ord: int = 2,
    weights: Optional[List[float]] = None,
) -> float:
    """
    STRESS modification for a set of pairs of color differences vectors.
    For the first vectors in pairs, scale of color differences can be different.
    For the second vectors in pairs, scale of color differences must be the same.

    Returns
    -------
    stress : float
        STRESS value.

    Parameters
    ----------
    x_list : list
        List of color differences vectors. Scales of vectors can be different.
    y_list : list
        List of color differences vectors. Scales of vectors must be the same.
    ord : int
        Order of STRESS. 1 and 2 options are available.

    Notes:
    -----
    Not symmetric function. Symmetry is only correct for the first argument.
    """

    if type(x_list) != list or type(y_list) != list:
        raise TypeError("'x_list' and 'y_list' types must be list")
    assert len(x_list) == len(y_list)

    k_list = list(
        map(
            lambda CDs: [find_optimal_k(CDs[0], CDs[1], ord=ord)] * CDs[0].shape[0],
            zip(x_list, y_list),
        )
    )
    k = np.hstack(k_list)
    x = np.hstack(x_list)
    y = np.hstack(y_list)
    if weights is None:
        stress = norm(k * x - y, ord=ord) / norm(y, ord=ord)
    else:
        assert len(weights) == len(x_list)
        weights_list = [[w ** (1 / ord)] * x.shape[0] for w, x in zip(weights, x_list)]
        w = np.hstack(weights_list)
        stress = norm(w * (k * x - y), ord=ord) / norm(w * y, ord=ord)
    return stress


def stress_report(
    x: list,
    y: list,
    wp_list: list,
    La_list: list,
    Yb_list: list,
    patch_size_list: list = [],
    extra_metrics: dict = {},
    sort: bool = True,
    quality_function: Callable[
        [List[npt.NDArray[np.float_]], List[npt.NDArray[np.float_]]], float
    ] = get_group_stress,
) -> dict:
    """
    Group STRESS computation for different CSs on a specified meta-dataset (hereafter just 'dataset').
    Default CSs for which STRESS is computed are presented in config.color_space_transforms_list.
    Other CSs can be specified in the argument 'extra'.
    Specified dataset

    Returns
    -------
    quality : dict
        Dictionary with computed group STRESS for CSs.
        Structure: {CS name (str): group STRESS (float)}.

    Parameters
    ----------
    x : list
        List of arrays with CIE XYZ coordinates of color pairs for each group in dataset.
        Each array of the shape (2, N, 3), where N - number of pairs.

    y : list
        List of ground truth vectors of color differences for each group in dataset.
        Each array of the shape (N,).

    extra : dict
        Dictionary with extra CSs.
        Structure: {CS name (str): transform function (Callable)}.
        Transform function must transform CIE XYZ data.

    sort : bool
        Is sort output dictionary by group STRESS value.

    wp_list : list
        List of illuminants for each group in dataset.

    La_list : list
        List of the luminances of the adapting field for each group in dataset.

    Yb_list : list
        List of the luminance factors of background for each group in dataset.
    """
    COLOR_SPACE_TRANSFORMS = {"CAM16-UCS": XYZ_to_CAM16UCS}
    transforms = deepcopy(COLOR_SPACE_TRANSFORMS)
    quality = {}
    if len(patch_size_list) == 0:
        patch_size_list = [None] * len(x)
    for k, f in transforms.items():
        l2 = []
        for i in range(len(x)):
            l2.append(
                calculate_metric(
                    x[i][0],
                    x[i][1],
                    wp_list[i],
                    La_list[i],
                    Yb_list[i],
                    k,
                    f,
                    patch_size_list[i],
                )
            )
        quality[k] = quality_function(y, l2)
    for k, metric_func in extra_metrics.items():
        l2 = []
        for i in range(len(x)):
            l2.append(metric_func(x[i][0], x[i][1]))
        quality[k] = quality_function(y, l2)
    if sort == True:
        quality = {k: v for k, v in sorted(quality.items(), key=lambda item: item[1])}
    elif isinstance(sort, list):
        quality = {k: quality[k] for k in sort}
    return quality


def find_optimal_k(x: np.array, y: np.array, ord: int = 2) -> float:
    """
    Find normalization factor k for STRESS between x, y,
    where x, y are vectors of color differences.

    ---
    L2-based STRESS case:
        STRESS_l2 = ||k * x - y|| / ||y||,
        where ||A|| -- l2-norm of vector A.

        k = argmin ||k * x - y|| = (x,y) / ||x||^2.
    ---
    L1-based STRESS case:
        STRESS_l1 = |k * x - y| / |y|,
        where |A| -- l1-norm of vector A.

        k = argmin |k * x - y|
    ---

    Returns
    -------
    k_opt : float
        Normalization factor k

    Parameters
    ----------
    x : array-like
        First vector of color differences.

    y : array-like
        Second vector of color differences.
    """
    if np.any(x < 0) or np.any(y < 0):
        raise ValueError("color differences in x and y must be >= 0")
    if ord == 2:
        return np.sum(x * y) / np.sum(x**2)
    elif ord == 1:
        sorted_yx_x = sorted(zip(y / x, x))
        yx_sorted = np.array([i for i, _ in sorted_yx_x])
        x_sorted = np.array([i for _, i in sorted_yx_x] + [0])
        derivative = np.zeros_like(x_sorted)
        for i in range(len(x_sorted)):
            derivative[i] = np.sum(x_sorted[:i]) - np.sum(x_sorted[i:])
        is_minimum = (derivative[1:] > 0) * (derivative[:-1] <= 0)
        k_opt = yx_sorted[is_minimum][0]
        return k_opt
    else:
        raise ValueError(f"unknown ord value: {ord}")
