import numpy as np
import os
import os.path as osp
from pathlib import Path
import json
from math import sin, radians



def calc_dists(center: dict, measurements: dict) -> float:
    if measurements["angle_2"] == 0:
        distance = np.sqrt((measurements["x"] - center["center_x"])**2 + (measurements["y"] - center["center_y"])**2)
    elif (measurements["angle_1"] in (0, 180) and measurements["angle_2"] in (-90, 90)):
        distance = np.abs(measurements["Y"] - center["center_Y"]) / center["Yc"]
    elif measurements["angle_2"] in (-45, 45):
        distance = np.abs(measurements["Y"] - center["center_Y"]) / sin(radians(measurements["angle_2"])) / center["Yc"]
    return distance


def get_dists(path: Path) -> [np.array, np.array]:
    all_dists = []
    mean_dist = []

    with open(path) as f:
        user = json.load(f)
        for center in user["color_centers"]:
            measurements = [m for m in center["measurements"]]
            sorted_m = sorted(measurements, key=lambda k: (k['angle_1'], k['angle_2']))
            directions = {}
            for m in sorted_m:
                distance = calc_dists(center, m)
                all_dists.append(np.abs(distance))
                if f"{m['angle_1']}_{m['angle_2']}" not in directions:
                    directions[f"{m['angle_1']}_{m['angle_2']}"] = []
                directions[f"{m['angle_1']}_{m['angle_2']}"].append(np.abs(distance))

            for d in directions:
                for _ in range(len(directions[d])):
                    mean_dist.append(np.mean(directions[d]))
                    
    return np.array(all_dists), np.array(mean_dist)
            

def calc_intraobserver_v(path: Path) -> float:

    all_dists, mean_dist = get_dists(path)
    sum_of_diffs_sqr = np.sum(np.square(all_dists - mean_dist))
    sum_of_sqr = np.sum(np.square(mean_dist))
    nrmses = np.sqrt(sum_of_diffs_sqr / sum_of_sqr)
    return np.mean(nrmses)


def get_all_unique(ms: np.array) -> list:
    all_unique = []
    for row in range(ms.shape[0]):
        center_angle = ms[row, :5]
        if list(center_angle) not in all_unique:
            all_unique.append(list(center_angle))
    return all_unique


def calc_avg_dist(all_ms: np.array) -> list[list, list]:
    avg_dists = []
    all_unique = get_all_unique(all_ms)
    for unique_c_a in all_unique:
        dists = []
        for row in range(all_ms.shape[0]):
            if list(all_ms[row, :5]) == unique_c_a:
                dists.append(all_ms[row, 5])
        avg_dists.append(np.mean(dists))
    return avg_dists, all_unique


def calc_avg_observer_dists(users: dict) -> list:
    all_ms = []
    for user in users:
        all_ms.append(users[user])
    all_ms = np.vstack(all_ms)
    all_ms = np.asarray(all_ms)
    return calc_avg_dist(all_ms)


def key(m: dict):
    return m["x"], m["y"], m["Y"], m["angle"], m.get("angle2", 0)


def calc_interobserver_v(path: Path, term: str = "stress") -> list[float]:
    files = os.listdir(path)
    participants = {}
    interobserver_v = []

    for file_name in files:
        with open(osp.join(path, file_name)) as f:
            user = json.load(f)
            valid_measurements = []
            for center in user["color_centers"]:
                for m in center["measurements"]:
                    if m["angle_2"] == 0:
                        distance = calc_dists(center, m)
                        valid_measurements.append([
                            center['center_x'], center['center_y'],
                            center['center_Y'], m["angle_1"], m["angle_2"],
                            distance
                        ])
        participants[file_name] = valid_measurements
    avg_dist, all_unique = calc_avg_observer_dists(participants)
    avg_dist = np.asarray(avg_dist)
    for user in participants:
        user_avg_dist, user_unique = calc_avg_dist(np.asarray(participants[user]))
        user_avg_dist = np.asarray(user_avg_dist)
        try:
            all_unique = np.array(all_unique)
            user_unique = np.array(user_unique)
            available_dist = avg_dist[(all_unique[:, None] == user_unique).all(-1).any(-1)]
            if term == "stress":
                factor = np.sum(user_avg_dist * available_dist) / np.sum(np.square(user_avg_dist))
            elif term == "nrmse":
                factor = 1
            else:
                raise RuntimeError("Term should be strees or nrmse")
            sum_of_diffs_sqr = np.sum(np.square(factor * user_avg_dist - available_dist))
            sum_of_sqr = np.sum(np.square(available_dist))
            interobserver_v.append(np.sqrt(sum_of_diffs_sqr / sum_of_sqr))
            print(user, np.sqrt(sum_of_diffs_sqr / sum_of_sqr))
        except Exception as e:
            print(e)

    return interobserver_v


def main():
    path2ms = Path("data/participants")
    for name in os.listdir(path2ms):
        path = Path(osp.join(path2ms, name))
        print(path)
        print(calc_intraobserver_v(path))
    calc_interobserver_v(path2ms, term="stress")


if __name__ == "__main__":
    main()