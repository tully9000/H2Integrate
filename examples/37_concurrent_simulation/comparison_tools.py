import numpy as np


def get_io(name, io_list):
    # Get the value of a specific input or output from the H2I inputs and outputs
    return [io[1]["val"] for io in io_list if io[0] == name][0]


def percent_diff(v1, v2):
    # Calculate the percent difference between two numbers or arrays
    pd = np.nan_to_num((v2 - v1) / (0.5 * (v1 + v2)))
    pd = np.where(np.max(np.abs(np.stack([v1, v2])), axis=0) <= 1e-8, 0, pd)
    return pd


def percent_diff_dicts(d1, d2):
    # Construct a dict of percent differences from two dicts with the same entries
    d1 = dict(d1)
    d2 = dict(d2)

    d_out = {}
    for k1, v1 in d1.items():
        assert k1 in d2.keys()
        v2 = d2[k1]

        if isinstance(v1["val"], dict | bool):
            # If the H2I input or output is more complicated than an array, skip it
            continue

        pd = percent_diff(v1["val"], v2["val"])

        d_out.update({k1: np.linalg.norm(pd)})

    return d_out


def find_nonzero_percent_diffs(pd_dict, ref_dict):
    # Return only the dict items that are non-zero
    abs_pd = {k: v for k, v in pd_dict.items() if np.abs(v) > 1e-8}
    rel_pd = {k: v for k, v in abs_pd.items() if np.linalg.norm(ref_dict[k]["val"]) > 1e-8}
    return abs_pd, rel_pd


# def plot_diff(key, io="outputs"):
#     if io == "outputs":
#         seq = dict(outputs_seq)
#         con = dict(outputs_con)
#     elif io == "inputs":
#         seq = dict(inputs_seq)
#         con = dict(inputs_con)

#     fig, ax = plt.subplots(2, 1, sharex="all", sharey="all", layout="constrained")

#     ax[0].plot(seq[key]["val"], label="sequential")
#     ax[0].plot(con[key]["val"], label="concurrent")
#     ax[0].legend()

#     ax[1].axhline(0, color="black", linewidth=1)
#     ax[1].fill_between(
#         np.arange(0, len(seq[key]["val"]), 1),
#         np.zeros_like(seq[key]["val"]),
#         seq[key]["val"] - con[key]["val"],
#     )

#     ax[0].set_title(key)
