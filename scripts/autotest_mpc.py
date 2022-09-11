#!/usr/bin/env python
import roslaunch
import rospy
from geometry_msgs.msg import PoseStamped, PoseArray, PoseWithCovarianceStamped, Quaternion, Pose
import os
import subprocess
import numpy as np
import time 
from bs4 import BeautifulSoup as bs
from shapely.geometry import Polygon
import tf.transformations
import yaml  # for loading config
import random
from mushr_coordination.msg import GoalPoseArray as GoalPoseArray_coord
from clcbs_ros.msg import GoalPoseArray as GoalPoseArray_clcbs
from visualization_msgs.msg import Marker


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

def angle_to_quaternion(angle):
    """Convert an angle in radians into a quaternion _message_."""
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))

def check_collision_rectangle(p1,p2):
    L = 0.25
    W = 0.12
    c1 = np.cos(p1[2])
    s1 = np.sin(p1[2])
    c2 = np.cos(p2[2])
    s2 = np.sin(p2[2])
    Lc1 = L*c1
    Wc1 = W*c1
    Ls1 = L*s1
    Ws1 = W*s1
    Lc2 = L*c2
    Wc2 = W*c2
    Ls2 = L*s2
    Ws2 = W*s2
    polygon1 = Polygon([(p1[0] + Lc1 + Ws1, p1[1] + Ls1 - Wc1), (p1[0] + Lc1 - Ws1, p1[1] + Ls1 + Wc1), (p1[0] - Lc1 - Ws1, p1[1] - Ls1 + Wc1), (p1[0] - Lc1 + Ws1, p1[1] - Ls1 - Wc1)])
    polygon2 = Polygon([(p2[0] + Lc2 + Ws2, p2[1] + Ls2 - Wc2), (p2[0] + Lc2 - Ws2, p2[1] + Ls2 + Wc2), (p2[0] - Lc2 - Ws2, p2[1] - Ls2 + Wc2), (p2[0] - Lc2 + Ws2, p2[1] - Ls2 - Wc2)])
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

        self.launchfile_mpc = config["launchfile_mpc"]
        self.launchfile_nhttc = config["launchfile_nhttc"]
        self.launchfile_TA = config["launchfile_TA"]
        self.launchfile_clcbs = config["launchfile_clcbs"]

        self.bagdir = config["bagdir"] + "/" + self.bench_name

        self.benchdir = config["benchdir"]

        self.task_file = config["task_file"] + self.bench_name + ".yaml"

        self.min_dist = 2
        self.N = config["num_iterations"]
        self.plan_pub = False
        self.num_agent = 4
        self.car_poses = np.zeros((self.num_agent,3))
        self.block_poses = np.zeros((self.num_agent,2))
        self.block_dist = np.zeros(self.num_agent)
        self.finished = np.zeros(self.num_agent)  # storing goal-reach status of each car
        self.deadlock = np.zeros(self.num_agent)
        self.time_error = np.zeros(self.num_agent)
        self.cte = np.zeros(self.num_agent)
        self.collision = False
        self.subs = []
        self.subs_ = []
        self.marker_subs = []
        self.goal_sub = None
        self.success = np.zeros(self.N)
        self.recovery = np.zeros((self.N,2))
        self.CTE_list = np.zeros(self.N)
        self.block_dist_list = np.zeros(self.N)
        self.min_dist_list = np.zeros(self.N)
        self.time_list = np.zeros(self.N)
        self.collision_log = np.zeros(self.N)
        self.deadlock_log = np.zeros(self.N)

        for i in range(4):
            sub_marker = rospy.Subscriber("/car"+ str(i+1) + "/marker",
                                           Marker, self.marker_cb, i, queue_size = 10)
            self.marker_subs.append(sub_marker)

        self.timeout = 100 # 60 second time out
        

    def adjust_launch_file(self, filename, bagdir, i, sys):
        with open(filename, 'r') as f:
            data = f.read()
        bs_data = bs(data, 'xml')
        # bag_name = bs_data.find("arg", {"name":"bag_name"})
        rec_name = bs_data.find("arg", {"name" : "record_name"})
        # bag_name["default"] = "clcbs_data_" + str(i)  # can put custom name according to "i" here
        rec_name["default"] = "clcbs_data_test_"+str(i)
        # output_dir
        rec_name = bs_data.find("arg", {"name" : "output_dir"})
        rec_name["default"] = bagdir

        output = bs_data.prettify()  # prettify doesn't actually make it prettier :(
        with open(filename, 'w') as f:
            f.write(output)

    def adjust_rosparams_mpc(self, sys):
        if(sys == "CLCBS+PP"):
            rospy.set_param("/car1/rhcontroller/control/cte_weight", 200)
            rospy.set_param("/car1/rhcontroller/control/time_weight", 20)
            # rospy.set_param("/car1/rhcontroller/control/ate_weight", 20)
            rospy.set_param("/car1/rhcontroller/control/collision_weight", 0)
            print("setting weights for PP")
        else:
            rospy.set_param("/car1/rhcontroller/control/cte_weight", 200)
            rospy.set_param("/car1/rhcontroller/control/time_weight", 20)
            # rospy.set_param("/car1/rhcontroller/control/ate_weight", 20)
            rospy.set_param("/car1/rhcontroller/control/collision_weight", 15)


    def pose_callback(self, msg, i):
        self.car_poses[i,0] = msg.pose.position.x
        self.car_poses[i,1] = msg.pose.position.y
        _,_, self.car_poses[i,2] = quaternion_to_angle(msg.pose.orientation)
        for j in range(self.num_agent):
            dist = np.linalg.norm(self.car_poses[i,:2] - self.car_poses[j,:2])
            if(self.min_dist > dist and j!=i):
                self.min_dist = dist  # keep track of min distance between any 2 cars
                if(dist < 0.4 and dist > 0.1):
                    self.collision = check_collision_rectangle(self.car_poses[i,:], self.car_poses[j,:])
                    if(self.collision):
                        print("collision")

        dist = np.linalg.norm(self.car_poses[i,:2] - self.block_poses[i])
        if(self.block_dist[i] > dist):
            self.block_dist[i] = dist
            # print(self.block_dist[i])

    def goal_callback(self, msg):
        self.plan_pub = True  # plan has been published

    def fin_callback(self, msg, i):  # callback for the same
        self.finished[i] = msg.pose.position.z or msg.pose.orientation.y < 0.3
        if(msg.pose.orientation.x and self.deadlock[i] == 0):  # did we face a deadlock?
            self.deadlock[i] = 1
        if(self.finished[i]):
            self.cte[i] = msg.pose.orientation.y
        self.time_error[i] = msg.pose.orientation.z  # keep updating time error until timeout

    def marker_cb(self, msg, i):
        if(msg.type == Marker.CUBE):  
            self.block_poses[i,0] = msg.pose.position.x
            self.block_poses[i,1] = msg.pose.position.y
    

    def sub_unsub(self, sub, goal_listen):
        if(sub and not goal_listen):
            for i in range(self.num_agent):
                subscriber = rospy.Subscriber("/car" + str(i + 1) + "/cur_goal",
                                              PoseStamped, self.fin_callback, i,queue_size = 10)  # subscriber
                subscriber_ = rospy.Subscriber("/car" + str(i + 1) + "/car_pose",
                                               PoseStamped, self.pose_callback, i, queue_size = 1)  # subscribe to car-pose to check collisions
                self.subs.append(subscriber)
                self.subs_.append(subscriber_)
        elif goal_listen:
            self.goal_sub = rospy.Subscriber("/car1/waypoints",
                                             PoseArray, self.goal_callback)  # subscribe to car-pose to check collisions for first car (plans pubbed simultaneously)
        else:
            for i in range(self.num_agent):
                self.subs[i].unregister()
                self.subs_[i].unregister()
            self.goal_sub.unregister()

    def init_planner(self, no_TA):
        config = None
        with open(self.task_file) as f:
            config = yaml.safe_load(f)
        if(config == None):
            print("no task file found")
            exit()

        num_agent = config["num_agent"]
        num_waypoint = config["num_waypoint"]
        randomness = config["randomness"]
        x_min = config["minx"]
        x_max = config["maxx"]
        y_min = config["miny"]
        y_max = config["maxy"]
        pubs = []
        pose_pubs = []
        target_pub = []
        mocap_subs = []
        # this is basically initializing all the subscribers for counting the
        # number of cars and publishers for initializing pose and goal points.
        for i in range(num_agent):
            name = "car"+str(i+1)
            print(name)
            publisher = rospy.Publisher(
                name + "/init_pose", PoseStamped, queue_size=5)
            pubs.append(publisher)
            pose_publisher = rospy.Publisher(
                name + "/initialpose", PoseWithCovarianceStamped, queue_size=5)
            pose_pubs.append(pose_publisher)
            # target point publishing in case we want to test nhttc standalone
            target = rospy.Publisher(
                name + "/waypoints", PoseArray, queue_size=5)
            target_pub.append(target)
        if(no_TA):
            goal_pub = rospy.Publisher(
                "/clcbs_ros/goals", GoalPoseArray_clcbs, queue_size=5)
        else:
            goal_pub = rospy.Publisher("/mushr_coordination/goals", GoalPoseArray_coord, queue_size=5)


        obs_pub1 = rospy.Publisher(
            "/clcbs_ros/obstacles", PoseArray, queue_size=5)
        obs_pub2 = rospy.Publisher(
            "/mushr_coordination/obstacles", PoseArray, queue_size=5)
        rospy.sleep(1)

        rospy.set_param("/init_clcbs/num_task", num_waypoint)
        for i in range(num_agent):
            now = rospy.Time.now()
            carmsg = PoseStamped()
            carmsg.header.frame_id = "/map"
            carmsg.header.stamp = now

            start_pose = config["car" + str(i + 1)]["start"]
            carmsg.pose.position.x = min(x_max, max(x_min, start_pose[0] + random.uniform(-randomness[0], randomness[0])))
            carmsg.pose.position.y = min(y_max, max(y_min, start_pose[1] + random.uniform(-randomness[1], randomness[1])))
            carmsg.pose.position.z = 0.0
            carmsg.pose.orientation = angle_to_quaternion(start_pose[2] + random.uniform(-randomness[2], randomness[2]))

            cur_pose = PoseWithCovarianceStamped()
            cur_pose.header.frame_id = "/map"
            cur_pose.header.stamp = now
            cur_pose.pose.pose = carmsg.pose
            # print(carmsg)
            rospy.sleep(1)
            pubs[i].publish(carmsg)
            pose_pubs[i].publish(cur_pose)

        now = rospy.Time.now()
        obsmsg = PoseArray()
        obsmsg.header.frame_id = "/map"
        obsmsg.header.stamp = now
        obs_pub1.publish(obsmsg)
        obs_pub2.publish(obsmsg)
        if(no_TA):
            goalmsg = GoalPoseArray_clcbs()
            goalmsg.num_agent = num_agent
            goalmsg.num_waypoint = num_waypoint
        else:
            goalmsg = GoalPoseArray_coord()
        goalmsg.header.frame_id = "/map"
        goalmsg.header.stamp = now
        goalmsg.scale = config["scale"]
        goalmsg.minx = x_min
        goalmsg.miny = y_min
        goalmsg.maxx = x_max
        goalmsg.maxy = y_max
        for i in range(num_agent):
            goalmsg.goals.append(PoseArray())
            waypoints = config["car" + str(i + 1)]["waypoints"]
            for j in range(num_waypoint):
                goal = Pose()
                goal.position.x = waypoints[j][0]
                goal.position.y = waypoints[j][1]
                goal.position.z = 0.0
                goal.orientation = angle_to_quaternion(waypoints[j][2])
                goalmsg.goals[i].poses.append(goal)
        print("tasks:", len(goalmsg.goals[i].poses))
        goal_pub.publish(goalmsg)

    def run_autotest(self):
        global process_generate_running
        for test_sys in self.sys_type:

            config = None
            with open(self.task_file) as f:
                config = yaml.safe_load(f)
            if(config == None):
                print("we got a problem")
                exit()

            self.num_agent = config["num_agent"]
            rospy.set_param("/init_clcbs/num_agent", self.num_agent)
            TA_launch = None
            clcbs_launch = None
            if(test_sys == "TA+CLCBS+MPC"):
                rospy.set_param("mushr_coordination/num_agent", self.num_agent)
                rospy.set_param("mushr_coordination/car1/name", "car1")
                rospy.set_param("mushr_coordination/car1/color","FF0000")
                rospy.set_param("mushr_coordination/car2/name", "car2")
                rospy.set_param("mushr_coordination/car2/color","00FF00")
                if(self.num_agent > 2):
                    rospy.set_param("mushr_coordination/car3/name", "car3")
                    rospy.set_param("mushr_coordination/car3/color","0000FF")
                if(self.num_agent > 3):
                    rospy.set_param("mushr_coordination/car4/name", "car4")
                    rospy.set_param("mushr_coordination/car4/color","DB921D")

                uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
                roslaunch.configure_logging(uuid)
                TA_launch = roslaunch.parent.ROSLaunchParent(uuid, 
                                                          [self.launchfile_TA],
                                                          process_listeners=[ProcessListener()])
                print("starting TA")
                TA_launch.start()
                time.sleep(1)

            uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
            roslaunch.configure_logging(uuid)
            clcbs_launch = roslaunch.parent.ROSLaunchParent(uuid, 
                                          [self.launchfile_clcbs],
                                          process_listeners=[ProcessListener()])
            clcbs_launch.start()
            print("starting CLCBS")
            time.sleep(1)

            if(test_sys == "NHTTC"):
                launchfile = self.launchfile_nhttc + str(self.num_agent) + ".launch"
            else:
                launchfile = self.launchfile_mpc + str(self.num_agent) + ".launch"

            bagdir = self.bagdir + "/" + test_sys
            
            ## try making the main directory
            try:
                os.mkdir(self.bagdir)
            except:
                pass

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
                self.block_poses = np.zeros((self.num_agent,2))

                self.adjust_launch_file(launchfile, bagdir, i, test_sys)
                process_generate_running = True
                uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
                roslaunch.configure_logging(uuid)
                launch = roslaunch.parent.ROSLaunchParent(uuid, 
                                                          [launchfile],
                                                          process_listeners=[ProcessListener()])
                launch.start()  # launch the damn thing
                time.sleep(5)
                if(test_sys == "TA+CLCBS+MPC"):
                    self.init_planner(no_TA = False)
                else:
                    self.init_planner(no_TA = True)

                self.plan_pub = False
                self.sub_unsub(sub=False, goal_listen = True)

                wait_start = time.time()
                while(not self.plan_pub and test_sys != "NHTTC"):
                    time.sleep(0.1)
                    collision = False
                    min_dist = 2
                    if(time.time() - wait_start > 1000):
                        if(test_sys == "TA+CLCBS+MPC"):
                            self.init_planner(no_TA = False)
                        else:
                            self.init_planner(no_TA = True)
                        wait_start = time.time()
                time.sleep(5)
                print("starting autotest")
                self.sub_unsub(sub=True, goal_listen = False)
                start_time = time.time()  # note the start time
                
                self.car_poses = np.zeros((self.num_agent,3))
                self.block_dist = np.ones(self.num_agent)*10
                self.finished = np.zeros(self.num_agent)  # reset the finished status
                self.deadlock = np.zeros(self.num_agent)
                self.cte = np.zeros(self.num_agent)
                self.time_error = np.zeros(self.num_agent)
                
                self.collision = False

                self.min_dist = 2
                
                while(self.finished.all() != 1 and time.time() - start_time < self.timeout):
                    rospy.sleep(1) # check once per second if all cars have reached their goal or if we have time-out
                if(self.finished.all() != 1): # or self.collision):  # happens if the cars don't finish within some stipulated time, usually due to deadlock
                    self.success[i] = 0 # failure
                else:
                    self.success[i] = 1 #(not self.collision) # success if not collision
                makespan_time = time.time() - start_time
                print("success, collision: ", self.success[i], self.collision)
                self.time_list[i] = makespan_time #self.time_error.mean()
                self.CTE_list[i] = self.cte.mean()
                self.block_dist_list[i] = np.max(self.block_dist)
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
            raw_data.append(self.block_dist_list)
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
            
            clcbs_launch.shutdown()
            
            if(TA_launch is not None):
                TA_launch.shutdown()

test = autotest("/home/stark/catkin_mushr/src/mushr_pixelart_mpc/config/autotest.yaml")
test.run_autotest()
