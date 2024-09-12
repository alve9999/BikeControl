from casadi import *
import sys
sys.path.append('/home/alve/control-sp/control/control/')
from mpc import *
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
import matplotlib.pyplot as plt
from std_msgs.msg import Float32
from nav_msgs.msg import Path
from messages.msg import State

path_debug=False


class ControlNode(Node):
    def __init__(self):
        self.path_received = False
        self.cones_received = False
        super().__init__('control_node')
        self.get_logger().info("it started");
        self.path_publisher = self.create_publisher(Path, '/control/path', 10)
        self.path_subscription = self.create_subscription(Path, '/mapping/center_line', self.path_callback, 10)
        self.state_subscription = self.create_subscription(State,'/state/est',self.state_callback,10)
        self.cones_subscription = self.create_subscription(MarkerArray,'/mapping/cones',self.cones_callback,10)
        self.speed_control = self.create_publisher(Float32, '/control/pedal', 10)
        self.steer_control = self.create_publisher(Float32, '/control/steer', 10)

        
    def path_callback(self, msg):
        if not self.path_received:
            self.path_points = np.array([(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses][::-1])
            self.get_logger().info(f'Received path with {len(msg.poses)} waypoints')
            self.path_received = True

    def init_mpc(self):
        self.mpc = mpc(self.spline_x,self.spline_y,self.spline_dx,self.spline_dy,self.arc_length[-1])
        
        
            
    def cones_callback(self, msg):
        if not self.cones_received and self.path_received:
            self.get_logger().info(f'Received {len(msg.markers)} cones')
            self.cones_left = []
            self.cones_right = []

            for marker in msg.markers:
                cone_x = marker.pose.position.x
                cone_y = marker.pose.position.y


                differences = np.linalg.norm(self.path_points - np.array([cone_x,cone_y]), axis=1)
                closest_index = np.argmin(differences)
                next_index = (closest_index + 1) % len(self.path_points)
                prev_point = self.path_points[closest_index]
                next_point = self.path_points[next_index]

                # Compute the vector perpendicular to the path segment
                path_vector = (next_point[0] - prev_point[0], next_point[1] - prev_point[1])
                cone_vector = (cone_x - prev_point[0], cone_y - prev_point[1])

                # Use the cross product to determine which side of the path segment the cone is on
                cross_product = path_vector[0] * cone_vector[1] - path_vector[1] * cone_vector[0]

                if cross_product > 0:
                    self.cones_left.append((cone_x, cone_y))
                else:
                    self.cones_right.append((cone_x, cone_y))
                    
            self.cones_received = True
            self.process_track(self.path_points)
            self.init_mpc()
            if path_debug:
                self.plot_res()

    def plot_res(self):
        # Plot the cones using matplotlib
        plt.figure(figsize=(10, 6))
            
        x_left, y_left = zip(*self.cones_left)
        plt.scatter(x_left, y_left, label='Cones Left of Path', color='red', marker='o')
            
        x_right, y_right = zip(*self.cones_right)
        plt.scatter(x_right, y_right, label='Cones Right of Path', color='green', marker='x')

        x, y = zip(*self.path_points[0:500])  # Unzip the path points into x and y coordinates
        plt.plot(x, y, label='Path', color='blue')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Cones and Path Visualization')

        test_t = np.linspace(-100, 10, 100)
        print(self.spline_x(-10))
        # Evaluate splines
        spline_x_vals = [float(self.spline_x(t)) for t in test_t]
        spline_y_vals = [float(self.spline_y(t)) for t in test_t]
        derivative_x_vals = [5*float(self.spline_dx(t)) for t in test_t]
        derivative_y_vals = [5*float(self.spline_dy(t)) for t in test_t]
        plt.quiver(spline_x_vals, spline_y_vals, derivative_x_vals, derivative_y_vals, angles='xy', scale_units='xy', scale=1, color='purple')
 
        plt.legend()
        plt.grid(True)
        plt.show()
            
            
    def state_callback(self, msg):
        if self.cones_received and self.path_received:
            self.get_logger().info(f'Received ground truth marker')
            yaw = msg.heading
            distances = np.linalg.norm(self.path_points - np.array([msg.x,msg.y]), axis=1)
            closest_index = np.argmin(distances)
            arc = self.arc_length[closest_index]%self.arc_length[-1]

            self.get_logger().info(f'info: {msg.x} {msg.y} {yaw} {arc}')
            res = self.mpc.solve([msg.x,msg.y,yaw,arc],self.path_publisher,self)
            speed = Float32()
            speed.data = float(res[0])
            
            steer = Float32()
            steer.data = float(res[1])
            self.speed_control.publish(steer)
            self.steer_control.publish(speed)
            
    def cumulative_arc_length(self,points):
        cumulative_lengths = [0]
        # Iterate through the list of points
        for i in range(1, len(points)):
            # Calculate the Euclidean distance between consecutive points
            distance = np.linalg.norm(np.array(points[i]) - np.array(points[i-1]))
            # Add the distance to the last cumulative length
            cumulative_lengths.append(cumulative_lengths[-1] + distance)
        self.total_arc = cumulative_lengths[-1]
        return cumulative_lengths

    def interpolate(self,points,arc_length):
        spline_x = interpolant('spline_x', 'bspline', [arc_length], points[:, 0])
        spline_y = interpolant('spline_y', 'bspline', [arc_length], points[:, 1])
        return spline_x,spline_y

    def derivate(self,points,arc_length):
        points = np.vstack([points,points[0]])
        derivative = np.diff(points, axis=0)
        spline_dx = interpolant('spline_x', 'bspline', [arc_length], derivative[:, 0])
        spline_dy = interpolant('spline_y', 'bspline', [arc_length], derivative[:, 1])
        return spline_dx,spline_dy

    def process_track(self,points):
        self.arc_length = self.cumulative_arc_length(points)
        self.spline_x,self.spline_y=self.interpolate(np.concatenate((points[-10:-2],points,points[1:10])),np.concatenate(([i-self.arc_length[-1] for i in self.arc_length[-10:-2]], self.arc_length,[i+self.arc_length[-1] for i in self.arc_length[1:10]])))
        #self.spline_left_x,self.spline_left_y=self.interpolate(self.cones_left,self.arc_length)
        #self.spline_right_x,self.spline_right_y=self.interpolate(self.cones_right,self.arc_length)
        self.spline_dx,self.spline_dy=self.derivate(np.concatenate((points[-10:-2],points,points[1:10])), np.concatenate(([i-self.arc_length[-1] for i in self.arc_length[-10:-2]],self.arc_length,[i+self.arc_length[-1] for i in self.arc_length[1:10]])))


    
    
    
def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

