import matplotlib.pyplot as plt
import numpy as np
import sys
import yaml

yaml_file = "/home/stark/catkin_mushr/src/mushr_pixelart_mpc/config/autotest.yaml"

config = None
with open(yaml_file) as f:
    config = yaml.safe_load(f)
if(config == None):
    print("no config found")
    exit()

base_dir = config["bagdir"] +"/" + config["bench_name"]

def get_m_std(x):
    y = x[np.isfinite(x)]  # because sometimes we get inf
    if len(y) == 0:
        return np.nan, np.nan
    m, s = np.mean(y), np.std(y)
    return m, s

output = []
labels = config["sys_name"]
colors = ["red", "purple", "blue", "green"]
block_tolerance = 0.15

sys_type = []
name_dict = config["sys_name"]
for name in name_dict:
    sys_type.append(name_dict[name])

for name in sys_type:
    sys = np.load(base_dir + "/" + name + "/output_raw.npy", allow_pickle=True)
    success_filter = (sys[0, :] > 0) & (sys[1, :] == 0)
    cte_m, cte_std = get_m_std(sys[2,success_filter])
    block_e_m, block_e_std = get_m_std(sys[3,success_filter])
    time_m, time_std = get_m_std(sys[4,success_filter])
    min_dist_m, min_dist_std = get_m_std(sys[5,success_filter & (sys[5, :] > 0)])
    block_fail = len(np.where(sys[3,:] > block_tolerance)[0]) / len(sys[3, :])
    makespan_m, makespan_std = get_m_std(sys[4,success_filter])
    success_rate = (sys[3, success_filter] < block_tolerance).sum() / len(sys[0, :])
    print("sys name: ", name)
    print("success: ", success_rate)
    print("cte m/std: ", cte_m, cte_std)
    print("block error m/std: ", block_e_m, block_e_std)
    print("min dist m/std: ", min_dist_m, min_dist_std)
    print("makespan m/std: ", makespan_m, makespan_std)
    print("==================")
