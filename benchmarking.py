wittfrom typing import NamedTuple
import warnings
import numpy as np
import numpy.typing as npt
import json

from report import stress_report, get_group_stress

import pandas as pd
import colour

warnings.filterwarnings("ignore", category=UserWarning)

FArray = npt.NDArray[np.float_]

d65 = np.array([0.95047, 1, 1.08883])



class Dataset(NamedTuple):
    """
    name: name of the dataset, for example: 'combvd', 'munsell', 'sdcth'
    x: list of arrays of xyz pairs
    y: list of arrays of xyz pairs
    wp_list: list of white points for each group
    La_list: list of La values for each group (used in CAM* models)
    Yb_list: list of Yb values for each group (used in CAM* models)
    """

    name: str
    x: list[FArray]
    y: list[FArray]
    wp_list: list[FArray]
    La_list: list[FArray]
    Yb_list: list[FArray]


def get_cv(e: list[FArray], v: list[FArray]) -> float:
    """
    :returns: CV metric of e and v lists
    """
    e, v = np.hstack(e), np.hstack(v)
    f = np.sum(e * v) / np.sum(np.square(v))
    return 100.0 * np.sqrt(
        np.mean(np.square(e - f * v) / np.square(np.mean(e)))
    )


def get_pf3(e: list[FArray], v: list[FArray]) -> float:
    """
    :returns: PF/3 metric of e and v lists
    """
    e, v = np.hstack(e), np.hstack(v)
    e_div_v = e / v  # division by 0 is possible here. live with it.
    F = np.sqrt(np.sum(e_div_v) / np.sum(v / e))
    f = np.sum(e * v) / np.sum(np.square(v))
    Vab = np.sqrt(np.mean(np.square(e - F * v) / e * F * v))
    CV_ = np.sqrt(np.mean(np.square(e - f * v) / np.square(np.mean(e))))

    log10_e_div_v = np.log10(e_div_v)
    γ = 10.0 ** np.sqrt(
        np.mean(np.square(log10_e_div_v - np.mean(log10_e_div_v)))
    )
    return (100.0 * (γ - 1.0) + Vab + CV_) / 3


def get_fixed8() -> dict:
    fixed8 = {
            "MaxBlue100": (0.180, 0.120, 100), 
            "Luo_MaxRed200": (0.654, 0.324, 200), 
            "MaxGreen200": (0.200, 0.650, 200), 
            "CIE_Gray": (0.314, 0.331, 30.0), 
            "CIE_Red": (0.484, 0.342, 14.1), 
            "CIE_Yellow": (0.388, 0.428, 69.3),
            "CIE_Green": (0.248, 0.362, 24.0), 
            "CIE_Blue": (0.219, 0.216, 8.8), 
            "Lao_Cyan": (0.193, 0.348, 62.3),
            "Lao_Red": (0.654, 0.324, 21.1),
    }
    return fixed8

from statistics import median as calc_median, StatisticsError
def calc_median_measurements(directions: dict) -> list[float]:
    median_colors = []
    for d in directions:
        ms = np.moveaxis(np.asarray(directions[d]), 0, 1)
        x = ms[0]
        y = ms[1]
        Y = ms[2]
        try:
            median_colors.append([
                calc_median([val for val in x if val is not None]),
                calc_median([val for val in y if val is not None]),
                calc_median([val for val in Y if val is not None]),
            ])
        except StatisticsError:
            return None
    return median_colors


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
         
        
def read_dataset_from_json(paths: list[str], centers: list[str], group: bool, flat: bool = False) -> Dataset:
    import json

    if "all" not in centers:
        fixed8 = get_fixed8()
        fixed_centers = [fixed8[c] for c in centers]
        print(fixed_centers)

    xyz_all = []
    angles_all = []
    for path in paths:
        with open(path) as f:
            measurements = json.load(f)["color_centers"]

        def is_in_fixed_centers(m: list) -> bool:
            return any(
                ((m["center_x"] - x) ** 2 + (m["center_y"] - y) ** 2 + (m["center_Y"] - Y) ** 2)
                < 1e-4
                for x, y, Y in fixed_centers
            )

        if "all" not in centers:
            measurements = [m for m in measurements if is_in_fixed_centers(m)]

        if len(measurements) > 0:
            xyY_pairs, angles = get_xyY_pairs_angles(measurements)            
            xyz_all.append([colour.xyY_to_XYZ(xyY_pairs) / 100])
            angles_all.append(angles)

    # print(xyz_all)

    median_xyz = np.median(xyz_all, axis=0)
    x = median_xyz
    wp_list = np.array([[d65] for vals in x])
    La_list = [np.full(len(_), 100.0) for _ in wp_list]
    Yb_list = [np.full(len(_), 70.0) for _ in wp_list]
    y = [np.ones(x[i].shape[1]) for i in range(len(x))]
    print(np.array(x).shape, np.array(y).shape, wp_list.shape, np.array(La_list).shape, np.array(Yb_list).shape)
    return Dataset("json files", x, y, wp_list, La_list, Yb_list)


def ciede2000_for_xyz_pairs(xyz_pairs: np.array, ref_illum_xyY: np.array):
    lab_pairs = xyz_pairs.copy()
    lab_pairs[0] = colour.XYZ_to_Lab(xyz_pairs[0], ref_illum_xyY)
    lab_pairs[1] = colour.XYZ_to_Lab(xyz_pairs[1], ref_illum_xyY)
    ciede2000 = colour.difference.delta_E_CIE2000(lab_pairs[0], lab_pairs[1])
    return ciede2000


def witt():

    ds = {
        "xyz_pairs": [],
        "CD": [],
        "group_names": [],
        "dataset_name": "COMBVD",
        "illuminants": [],
        "L_A": [],
        "Y_b": [],
        "c": [],
    }

    dataset_path = r'data\witt\witt.json'
    # print("combvd path", DATASETS_NAMES_PATHS["COMBVD"][name])
    # for data_path in files_path:
    ds = {}
    with open(dataset_path) as f:
        data = json.load(f)
    xyz = np.asarray(data["xyz"])
    pairs = np.asarray(data["pairs"])
    ds["CD"] = np.array(data["dv"])
    ds["xyz_pairs"] = np.transpose(xyz[pairs], (1, 0, 2)) / 100
    wp = np.array(data["reference_white"]) / 100
    ds["illuminants"] = wp
    ds["L_A"] = data["L_A"]
    ds["Y_b"] = data["Y_b"]
    ds["c"] = data["c"]
    ds["group_names"] = "Witt"
    ds["patch_sizes"] = [120] * len(ds["group_names"])
    return ds


def read_dataset_witt(centers: list[str], group: bool) -> Dataset:
    """
    :returns: `Dataset` object of Witt dataset
    """
    
    dataset = witt()

    xyz_all = dataset["xyz_pairs"]
    
    fixed8 = get_fixed8()
    fixed_centers = [fixed8[c] for c in centers]
    fixed_centers = [colour.xyY_to_XYZ(x) / 100 for x in fixed_centers]

    def is_in_fixed8(pair: list, dist=2) -> bool:
        return any(
            ciede2000_for_xyz_pairs([pair[0], (x, y, z)], d65)
            < dist and
            ciede2000_for_xyz_pairs([pair[1], (x, y, z)], d65)
            < dist 
            for x, y, z in fixed_centers
        )
    
    cd_centers = []
    pairs1 = []
    pairs2 = []
    for idx in range(len(dataset['CD'])):
        if is_in_fixed8([xyz_all[0][idx], xyz_all[1][idx]], dist=2):
            pairs1.append(xyz_all[0][idx].tolist())
            pairs2.append(xyz_all[1][idx].tolist())
            cd_centers.append(dataset["CD"][idx])
    xyz_centers = [[pairs1, pairs2]]

    x = [np.hstack(xyz_centers)]
    y = [np.hstack([cd_centers])]

    wp_list = np.array([[d65] for vals in x])
    La_list = [np.full(len(_),  dataset["L_A"]) for _ in wp_list]
    Yb_list = [np.full(len(_), dataset["Y_b"]) for _ in wp_list]
    print(np.array(x).shape, np.array(y).shape, wp_list.shape, np.array(La_list).shape, np.array(Yb_list).shape)
    return Dataset("combvd", x, y, wp_list, La_list, Yb_list)


def read_dataset(paths: list[str], centers: list[str], group: bool) -> Dataset:
    """
    Universal dataset reader
    :returns: `Dataset` object
    """
    if len(paths) == 1:
        if paths[0].lower() == "witt":
            return read_dataset_witt(centers, group)
    if paths and paths[0].endswith(".json"):
        return read_dataset_from_json(paths, centers, group)


def full_report(
    paths: list[str],
    group: bool,
    centers: list[str],
    metric: str,
) -> tuple[str, dict[str, float]]:
    dataset = read_dataset(paths, centers, group)
    metric_func = {
        "stress": get_group_stress,
        "cv": get_cv,
        "pf/3": get_pf3,
    }[metric]

    stress_report_both = stress_report(
        dataset.x,
        dataset.y,
        dataset.wp_list,
        dataset.La_list,
        dataset.Yb_list,
        quality_function=metric_func,
    )

    return {
        "dataset": dataset.name if dataset.name != "ods files" else paths,
        "metric": metric,
        "report": stress_report_both,
        "centers": str(centers)
    }


def main():
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "paths",
        nargs="+",
        help="can be paths to json files or 'witt'",
    )
    parser.add_argument("--no-group", action="store_false", dest="group")
    parser.add_argument(
        "--centers",
        nargs="+",
        default=[
            "all"
        ],
    )
    parser.add_argument(
        "--metric", choices=["stress", "cv", "pf/3"], default="stress"
    )
    args = parser.parse_args()

    report = full_report(args.paths, args.group, args.centers, args.metric)
    json.dump(report, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
