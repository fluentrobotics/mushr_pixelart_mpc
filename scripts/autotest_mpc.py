#!/usr/bin/env python
import roslaunch
import rospy
from geometry_msgs.msg import PoseStamped, PoseArray
import os
import subprocess
import numpy as np
import time 
from bs4 import BeautifulSoup as bs
from shapely.geometry import Polygon
import tf.transformations
import yaml  # for loading config


rospy.init_node('autotest', anonymous=True)

## this helps us track whether the launch file has been completely shut down yet or not
process_generate_running = True
class ProcessListener(roslaunch.pmon.ProcessListener):
    global process_generate_running

    def process_died(self, name, exit_code):
        global process_generate_running  # set this variable to false when process has died cleanly.
        if(name[:11] != "rosbag_play" or name[:13] != "rosbag_record"):  # prevent these guys from killing the whole process
            process_generate_running = False
            rospy.logwarn("%s died with code %s", name, exit_code)

def quaternion_to_angle(q):
    """
    Convert a quaternion _message_ into an angle in radians.
    The angle represents the yaw.
    This is not just the z component of the quaternion.

    This function was finessed from the mushr_base/src/utils.py
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return roll, pitch, yaw

def check_collision_rectangle(p1,p2):
    L = 0.25
    W = 0.12
    c1 = np.cos(p1[2])
    s1 = np.sin(p1[2])
    c2 = np.cos(p2[2])
    s2 = np.cos(p2[2])
    Lc1 = L*c1
    Wc1 = W*c1
    Ls1 = L*s1
    Ws1 = W*s1
    Lc2 = L*c2
    Wc2 = W*c2
    Ls2 = L*s2
    Ws2 = W*s2
    polygon1 = Polygon([(p1[0] + Lc1 + Ws1, p1[1] + Ls1 - Wc1), (p1[0] + Lc1 - Ws1, p1[1] + Ls1 + Wc1), (p1[0] - Lc1 - Ws1, p1[1] - Ls1 + Wc1), (p1[0] - Lc1 + Ws1, p1[1] - Ls1 - Wc1)])
    polygon2 = Polygon([(p2[0] + Lc1 + Ws1, p2[1] + Ls1 - Wc1), (p2[0] + Lc1 - Ws1, p2[1] + Ls1 + Wc1), (p2[0] - Lc1 - Ws1, p2[1] - Ls1 + Wc1), (p2[0] - Lc1 + Ws1, p2[1] - Ls1 - Wc1)])
    return polygon1.intersects(polygon2)

class autotest():
    """autotest class"""
    def __init__(self, yaml_file):
        config = None
        with open(yaml_file) as f:
            config = yaml.safe_load(f)
        if(config == None):
            print("no config found")
            exit()
        self.sys_type = []
        name_dict = config["sys_name"]
        for name in name_dict:
            self.sys_type.append(name_dict[name])

        self.bench_name = config["bench_name"]

        self.launchfile = config["launchfile"]

        self.bagdir = config["bagdir"]

        self.benchdir = config["benchdir"]

        self.min_dist = 2
        self.N = 2
        self.plan_pub = False
        self.car_poses = np.zeros((4,3))
        self.finished = np.zeros(4)  # storing goal-reach status of each car
        self.deadlock = np.zeros(4)
        self.time_error = np.zeros(4)
        self.cte = np.zeros(4)
        self.collision = False
        self.subs = []
        self.subs_ = []
        self.goal_sub = None
        self.success = np.zeros(self.N)
        self.recovery = np.zeros((self.N,2))
        self.CTE_list = np.zeros(self.N)
        self.min_dist_list = np.zeros(self.N)
        self.time_list = np.zeros(self.N)
        self.collision_log = np.zeros(self.N)
        self.deadlock_log = np.zeros(self.N)

        self.timeout = 60 # 60 second time out
        

    def adjust_launch_file(self, filename, bagdir, i, sys):
        with open(filename, 'r') as f:
            data = f.read()
        bs_data = bs(data, 'xml')
        # bag_name = bs_data.find("arg", {"name":"bag_name"})
        rec_name = bs_data.find("arg", {"name" : "record_name"})
        # bag_name["default"] = "clcbs_data_" + str(i)  # can put custom name according to "i" here
        rec_name["default"] = "/clcbs_data_test_"+str(i)
        # output_dir
        rec_name = bs_data.find("arg", {"name" : "output_dir"})
        rec_name["default"] = bagdir

        output = bs_data.prettify()  # prettify doesn't actually make it prettier :(
        with open(filename, 'w') as f:
            f.write(output)

    def adjust_rosparams_mpc(self, sys):
        if(sys == "TA+CLCBS+MPC" or sys == "CLCBS+MPC" or sys == "MPC"): 
            rospy.set_param("/car1/rhcontroller/control/cte_weight", 200)
            rospy.set_param("/car1/rhcontroller/control/time_weight", 20)
            rospy.set_param("/car1/rhcontroller/control/collision_weight", 150)

        if(sys == "PP"):
            rospy.set_param("/car1/rhcontroller/control/cte_weight", 200)
            rospy.set_param("/car1/rhcontroller/control/time_weight", 20)
            rospy.set_param("/car1/rhcontroller/control/collision_weight", 0)

    def pose_callback(self, msg, i):
        self.car_poses[i,0] = msg.pose.position.x
        self.car_poses[i,1] = msg.pose.position.y
        _,_, self.car_poses[i,2] = quaternion_to_angle(msg.pose.orientation)
        for j in range(4):
            dist = np.linalg.norm(self.car_poses[i,:2] - self.car_poses[j,:2])
            if(self.min_dist > dist and j!=i):
                self.min_dist = dist  # keep track of min distance between any 2 cars
                if(dist < 0.4):
                    self.collision = True

    def goal_callback(self, msg):
        self.plan_pub = True  # plan has been published

    def fin_callback(self, msg, i):  # callback for the same
        self.finished[i] = msg.pose.position.z  # nhttc sets it to 1 when it has reached goal
        self.finished[i] = msg.pose.orientation.y < 0.5
        if(msg.pose.orientation.x and self.deadlock[i] == 0):  # did we face a deadlock?
            self.deadlock[i] = 1
        if(self.finished[i]):
            self.cte[i] = msg.pose.orientation.y
        self.time_error[i] = msg.pose.orientation.z  # keep updating time error until timeout


    def sub_unsub(self, sub, goal_listen):
        if(sub and not goal_listen):
            for i in range(4):
                subscriber = rospy.Subscriber("/car" + str(i + 1) + "/cur_goal",
                                              PoseStamped, self.fin_callback, i)  # subscriber
                subscriber_ = rospy.Subscriber("/car" + str(i + 1) + "/car_pose",
                                               PoseStamped, self.pose_callback, i)  # subscribe to car-pose to check collisions
                self.subs.append(subscriber)
                self.subs_.append(subscriber_)
        elif goal_listen:
            self.goal_sub = rospy.Subscriber("/car1/waypoints",
                                             PoseArray, self.goal_callback)  # subscribe to car-pose to check collisions for first car (plans pubbed simultaneously)
        else:
            for i in range(4):
                self.subs[i].unregister()
                self.subs_[i].unregister()
            self.goal_sub.unregister()

    def run_autotest(self):
        global process_generate_running
        for test_sys in self.sys_type:
            ## change this
            launchfile = self.launchfile
            bagdir = self.bagdir + "/" + test_sys
            try:
                os.mkdir(bagdir)
            except:
                pass

            self.adjust_rosparams_mpc(test_sys)

            self.success = np.zeros(self.N)
            self.recovery = np.zeros((self.N,2))
            self.CTE_list = np.zeros(self.N)
            self.min_dist_list = np.zeros(self.N)
            self.time_list = np.zeros(self.N)
            self.collision_log = np.zeros(self.N)
            self.deadlock_log = np.zeros(self.N)

            for i in range(self.N):
                print("iteration " + str(i) + "starting")

                self.adjust_launch_file(launchfile, bagdir, i, test_sys)
                process_generate_running = True
                uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
                roslaunch.configure_logging(uuid)
                launch = roslaunch.parent.ROSLaunchParent(uuid, 
                                                          [launchfile],
                                                          process_listeners=[ProcessListener()])
                launch.start()  # launch the damn thing
                time.sleep(10)  # time it takes for the whole thing to boot into existence

                self.plan_pub = False
                self.sub_unsub(sub=False, goal_listen = True)
                ## change this
                pp = subprocess.Popen(['rosbag', 'play', '-q', 'clcbs_data_'+str(i)+'.bag'],
                                      cwd = self.benchdir + "/" + self.bench_name)

                wait_start = time.time()
                while(not self.plan_pub):
                    time.sleep(0.1)
                    collision = False
                    min_dist = 2
                    if(time.time() - wait_start > 10):
                        print("retrying rosbag play")
                        pp = subprocess.Popen(['rosbag', 'play', '-q', 'clcbs_data_'+str(i)+'.bag'],
                                              cwd = self.benchdir + "/" + self.bench_name)
                        wait_start = time.time()
                time.sleep(1)
                self.sub_unsub(sub=True, goal_listen = False)

                start_time = time.time()  # note the start time
                
                self.finished = np.zeros(4)  # reset the finished status
                self.deadlock = np.zeros(4)
                self.cte = np.zeros(4)
                self.time_error = np.zeros(4)
                
                self.collision = False
                self.min_dist = 2
                
                while(self.finished.all() != 1 and time.time() - start_time < self.timeout):
                    rospy.sleep(1) # check once per second if all cars have reached their goal or if we have time-out
                if(self.finished.all() != 1 or self.collision):  # happens if the cars don't finish within some stipulated time, usually due to deadlock
                    self.success[i] = 0 # failure
                else:
                    self.success[i] = (not self.collision) # success if not collision

                print("success, collision: ", self.success[i], self.collision)
                self.time_list[i] = self.time_error.mean()
                self.CTE_list[i] = self.cte.mean()
                self.min_dist_list[i] = self.min_dist
                self.deadlock_log[i] = self.deadlock.any() == 1
                if(self.deadlock.all()):
                    self.recovery[i,0] = self.success[i]
                    self.recovery[i,1] = 1  # indicate that it was indeed a deadlock
                self.collision_log[i] = self.collision
                self.sub_unsub(sub=False, goal_listen = False)  # unsub from topics
                
                launch.shutdown()
                print("waiting 5 seconds for clean exit")
                time.sleep(5)


            raw_data = []
            raw_data.append(self.success)
            raw_data.append(self.collision_log)
            raw_data.append(self.CTE_list)
            raw_data.append(self.time_list)
            raw_data.append(self.min_dist_list)
            raw_data.append(self.deadlock_log)
            # change this
            np.save(bagdir + "/output_raw.npy", raw_data)
            files = os.listdir(bagdir)
            files.sort(key = lambda x: os.path.getmtime(bagdir+'/'+x))
            for i in range(self.N):
                source = bagdir + '/' + files[i]
                dest = bagdir + '/' + 'clcbs_data_test_' + str(i) + '.bag'
                os.rename(source, dest)
            print("success rate:", self.success.mean())

test = autotest("/home/stark/catkin_mushr/src/mushr_pixelart_mpc/config/autotest.yaml")
test.run_autotest()