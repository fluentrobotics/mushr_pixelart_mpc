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
    m, s = np.mean(y), np.std(y)
    return m, s

output = []
labels = config["sys_name"]
colors = ["red", "purple", "blue", "green"]
bar_width = 0.15

sys_type = []
name_dict = config["sys_name"]
for name in name_dict:
    sys_type.append(name_dict[name])

for name in sys_type:
    sys = np.load(base_dir + "/" + name + "/output_raw.npy", allow_pickle=True)
    success = sys[0,:].mean()
    collision = sys[1,:].mean()
    cte_m, cte_std = get_m_std(sys[2,:])
    block_e_m, block_e_std = get_m_std(sys[3,:])
    time_m, time_std = get_m_std(sys[4,:])
    min_dist_m, min_dist_std = get_m_std(sys[5,:])
    block_fail = len(np.where(sys[3,:] > 0.15)[0])* 0.01
    makespan_m, makespan_std = get_m_std(sys[4,:])
    print("sys name: ", name)
    print("success: ", max(success - collision - block_fail,0), "valid for clcbs+pp")  # use this for CLCBS+PP
    print("success: ", max(success - block_fail,0), "valid for every other system with collision avoidance")  # use this for everything else.
    print("cases: ", sys[0,:])
    print("cte m/std: ", cte_m, cte_std)
    print("block error m/std: ", block_e_m, block_e_std)
    print("makespan_m/std: ", makespan_m, makespan_std)
    print("==================")
    # print("time m/std: ", time_m, time_std)
    # exit()

# index1 = np.where(full_sys[0,:])
# index2 = np.where(full_sys_ded[0,:])
# index3 = np.where(locl_sys[0,:])
# index4 = np.where(glob_sys[0,:])
# index5 = np.where(idot_sys[0,:])
# index6 = np.where(new_sys[0,:])
# index7 = np.where(new_sys2[0,:])


# final_index_1 = np.intersect1d(index1, index2) 
# final_index_2 = np.intersect1d(index3, index4)
# final_index_3 = np.intersect1d(index5, index6)
# final_index = np.intersect1d(final_index_1, final_index_2)
# final_index = np.intersect1d(final_index, final_index_3)
# final_index = np.intersect1d(final_index, index7)
# time_weight = 30 + 20*(final_index//10)
# collision_weight = 30 + 20*(final_index%10)
# print(collision_weight, time_weight)
# # X = np.arange(100)*2 + 10
# # plt.xlabel("along track weight")
# # plt.ylabel("minimum distance")
# # plt.scatter(X, data[4,:])
# # plt.savefig('/home/stark/catkin_mushr/src/nhttc_ros/bags/'+bagdir+'/'+'ate_collision'+'.jpg')
# X = np.arange(10)*20 + 30
# Y = np.copy(X)
# plt.title("minimum distance")
# plt.ylabel("along track weight")
# plt.xlabel("collision weight")

# cm = plt.get_cmap('gist_rainbow')
# index = np.where(data[1,:] > 0)
# min_dist = np.min(data[3,index])
# max_dist = np.max(data[3,index])
# range_dist = max_dist - min_dist
# for i in range(10):
#     for j in range(10):
#         index = i + 10*j
#         dist = data[3, index]
#         col = dist/range_dist
#         clr = cm(col)
#         if(data[1, index] < 1):
#             clr = 'white'
#         plt.scatter(X[i], Y[j], color=clr)
# print(min_dist, max_dist)
# for i in range(100):
#     plt.scatter(i+30, 250, color=cm(i*0.01))
# plt.savefig('/home/stark/catkin_mushr/src/nhttc_ros/bags/'+bagdir+'/'+'2D min dist wind 1'+'.jpg')


'''
def open_bag(filepath):
    try:
        return rosbag.Bag(filepath)
    except (rosbag.ROSBagException, IOError) as e:
        print("Error: bag at " + filepath + " not readable!")
        sys.exit(1)

def read_messages(bag, topic_list, car_names):
    waypoint_pub_time = None
    waypoint_topic_list = [topic[:-9] + "/waypoints" for topic in topic_list]
    for topic, msg, t in bag.read_messages(topics=waypoint_topic_list):
        waypoint_pub_time = t
        break;
    for topic, msg, t in bag.read_messages(topics=topic_list, start_time=waypoint_pub_time):
        if(msg.pose.orientation.x==1):
            return 1
    return 0


def update_collision(x):
    x[1,:] = np.array(x[4,:] < 0.3, dtype=float)
    x[0,:] = x[0,:]*np.array(x[1,:] < 1, dtype =float)
    return x

def update_success(x):
    x[0,:] = x[1,:] - 1
    x[0,:] *= -1
    return x



def deadlock_report(sys, sys_dir):
    index = (np.where((sys[0,:]<1) & (sys[1,:] == 0)))  # where we expect a deadlock
    car_sim_name = [str(i + 1) for i in range(4)]
    sim_topics = ["/car" + name + "/cur_goal" for name in car_sim_name]
    deadlock = np.zeros(100)
    # for bag_number in range(100):
    #     sim_bag_path = base_dir + bagdir + '/' + sys_dir+'/clcbs_data_test_'+ str(bag_number)+'.bag'
    #     sim_bag = open_bag(sim_bag_path)
    #     deadlock[bag_number] = read_messages(sim_bag, sim_topics, car_sim_name)
    # print(sys_dir)
    # print("measured deadlock case numbers:",np.where(deadlock))
    print("expected deadlock case numbers:", index)

# full_sys = update_success(full_sys)
# locl_sys = update_success(locl_sys)
# full_sys_ded = update_success(full_sys_ded)
# glob_sys = update_success(glob_sys)
# idot_sys = update_success(idot_sys)

deadlock_report(full_sys, name[0])
deadlock_report(full_sys_ded,  name[1])
deadlock_report(locl_sys,  name[2])
deadlock_report(glob_sys,  name[3])
deadlock_report(idot_sys,  name[4])
# update_collision(full_sys)
# update_collision(full_sys_ded)
# update_collision(locl_sys)
# update_collision(glob_sys)
# update_collision(idot_sys)

mean = np.zeros((5,5))  # order: success, collisions, final error, relative travel time, minimum separation (center to center)
stds = np.zeros((5,5))

for i in range(5):
    mean[i,0], stds[i,0] = get_m_std(full_sys[i, :], i<2)
    mean[i,1], stds[i,1] = get_m_std(full_sys_ded[i,:], i<2)
    mean[i,2], stds[i,2] = get_m_std(locl_sys[i, :], i<2)
    mean[i,3], stds[i,3] = get_m_std(glob_sys[i, :], i<2)
    mean[i,4], stds[i,4] = get_m_std(idot_sys[i, :], i<2)

metrics = ['success rate (0-1)', 'collision rate (0-1)', 'final_error(meters)', 'relative travel time (ratio)', 'minimum separation(meters)']
x_pos = np.arange(len(metrics))


labels = name #['mpc time cost=2', 'mpc time_cost=1.5','mpc time cost =1', 'mpc vanilla heading', 'mpc vanilla no head']
colors = ['red', 'purple','blue', 'green', 'grey']
bar_width = 0.15

for j in range(5):
    fig, ax = plt.subplots()
    for i in range(5):
        ax.bar(bar_width*i, mean[j,i], bar_width, yerr=stds[j,i], align='center', color = colors[i], capsize=10, label = labels[i])
    ax.set_xticks(x_pos[:2])
    ax.set_ylabel(metrics[j])
    ax.set_xlim(-0.1,2)
    if(j==0 or j ==1):
        ax.set_ylim(0,1)
    # ax.set_xticklabels()
    # ax.set_title('performance comparison between systems')
    ax.yaxis.grid(True)
    ax.legend()
    plt.savefig('/home/stark/catkin_mushr/src/nhttc_ros/bags/'+bagdir+'/'+metrics[j]+'.jpg')
    # plt.show()
'''
