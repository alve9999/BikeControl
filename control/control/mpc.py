from casadi import *
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import rclpy

from rclpy.node import Node
from rclpy.time import Duration
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

show_init = False
show_predict = True

class mpc:
    radius = 0.425
    wheelbase = 0.8
    k = 5
    dt = 0.1


    def __init__(self,spline_x,spline_y,spline_dx,spline_dy,arc_length):
        self.spline_x=spline_x
        self.spline_y=spline_y
        self.spline_dx=spline_dx
        self.spline_dy=spline_dy
        self.arc_length=arc_length
        self.init_system()
        self.init_mpc()
        self.init_constraints()
        
    def init_system(self):
        x = MX.sym('x')
        y = MX.sym('y')
        theta = MX.sym('theta')
        arc = MX.sym('arc')
        delta = MX.sym('delta')
        u = MX.sym('u')
        v = u*self.radius*3.1415*5
        
        xdot = v * cos(theta)
        ydot = v * sin(theta)
        thetadot = v * tan(delta) / self.wheelbase
        
        state = vertcat(x, y, theta, arc)
        control = vertcat(delta, u)
        rhs = vertcat(xdot, ydot, thetadot, v)

        self.f = Function('f', [state, control], [rhs])
        self.states = MX.sym('states', 4, self.k + 1)
        self.control = MX.sym('control', 2, self.k)
        self.params = MX.sym('params',4)
        self.cons = []
        self.obj = 0

    def init_mpc(self):
        current_state = self.states[:, 0]
        self.cons = vertcat(self.cons,current_state-self.params[0:4])
        for i in range(self.k):
            current_control = self.control[:, i]
            current_state = self.states[:, i]
            next_state = self.states[:, i+1]
            dx, dy = self.spline_dx(next_state[3]), self.spline_dy(next_state[3])
            epsi = atan2(dy, dx)
            x_ref, y_ref = self.spline_x(next_state[3]), self.spline_y(next_state[3])
            cte = ((x_ref-next_state[0])**2 + (y_ref-next_state[1])**2)
            heading_error = ((epsi-next_state[2])**2)
            predicted_next = current_state + self.dt * self.f(current_state,current_control)
            self.cons = vertcat(self.cons, next_state - predicted_next)
            self.cons = vertcat(self.cons, heading_error)
            self.cons = vertcat(self.cons, cte)
            self.obj += (2-current_control[1])**2 + 0.1*heading_error**2 + 0.1*cte**2 + current_control[0]**2*0.0
        for i in range(self.k-1):
            current = self.control[:, i]
            nex = self.control[:, i+1]
            self.obj += 0.2*(current[0]-nex[0])**2 + 0.0*(current[1]-nex[1])**2
        vars = vertcat(reshape(self.states, -1, 1), reshape(self.control, -1, 1))
        self.nlp_prob = {'f': self.obj, 'x': vars, 'g': self.cons,'p':self.params}
        opts = {
            'ipopt': {
                'max_iter': 400,
                'print_level': 4,
                'acceptable_tol': 1e-3,
                'acceptable_obj_change_tol': 1e-3
            },
            'print_time': 0
        }
        self.solver = nlpsol('solver', 'ipopt', self.nlp_prob, opts)

    def init_constraints(self):
        self.lower_bound_vars = np.zeros((self.k+1)*4+self.k*2)
        self.upper_bound_vars = np.zeros((self.k+1)*4+self.k*2)
        self.lower_bound_cons = np.zeros((self.k+1)*4+self.k*2)
        self.upper_bound_cons = np.zeros((self.k+1)*4+self.k*2)
        for i in range((self.k+1)*4):
            self.lower_bound_vars[i]=-1000
            self.upper_bound_vars[i]=1000
        for i in range(self.k):
            index_offset = (self.k+1)*4 + i*2
            self.lower_bound_vars[index_offset]=-0.4*3.1415
            self.upper_bound_vars[index_offset]=0.4*3.1415
            self.lower_bound_vars[index_offset+1]=0
            self.upper_bound_vars[index_offset+1]=3
        for i in range(self.k):
            index_offset = 4 + i*6
            self.lower_bound_cons[index_offset+4]=-0.8
            self.upper_bound_cons[index_offset+4]=0.8
            self.lower_bound_cons[index_offset+5]=0.0
            self.upper_bound_cons[index_offset+5]=1.0

    def solve(self,state,publisher,node):
        initial = np.zeros((self.k+1)*4+self.k*2)
        for i in range(4):
            initial[i]=state[i]
        for i in range(self.k):
            index = 4*(i+1)
            initial[index+3]=(initial[index-1]+self.dt*3*5*self.radius*3.1415)%self.arc_length
            initial[index]=self.spline_x(initial[index+3])
            initial[index+1]=self.spline_y(initial[index+3])
            initial[index+2]=atan2(self.spline_dy(initial[index+3]),self.spline_dx(initial[index+3]))

        for i in range(self.k):
            #print(i)
            index = 4*i
            control_index = 4*(self.k+1) + 2*i
            prev_dx,prev_dy=self.spline_dx(initial[index+3]),self.spline_dy(initial[index+3])
            dx,dy=self.spline_dx(initial[index+7]),self.spline_dy(initial[index+7])
            heading_change = atan2(dy,dx)-atan2(prev_dy,prev_dx)
            #print(initial[index+3],initial[index+7])
            #print(prev_dx,prev_dy,dx,dy)
            #print(heading_change)
            initial[control_index]=heading_change
            initial[control_index+1]=1.0

        if show_init:
            path_msg = Path()
            center_points=[]
            path_msg.header.frame_id = 'map'
            for i in range(self.k):
                index = 4*i
                center_point = PoseStamped()
                center_point.header.frame_id = 'map'
                # YOU CAN CHANGE THE POSE TIME STAMP HERE                                                                                                                  
                center_point.header.stamp = node.get_clock().now().to_msg()
                # Pose                                                                                                                                      
                center_point.pose.position.x = initial[index]
                center_point.pose.position.y = initial[index+1]
                center_point.pose.position.z = 0.
                # Compute orientation                                      
                yaw = initial[index+2]
                center_point.pose.orientation.w = np.cos(.5 * yaw)
                center_point.pose.orientation.x = 0.
                center_point.pose.orientation.y = 0.
                center_point.pose.orientation.z = np.sin(.5 * yaw)
                center_points.append(center_point)
            path_msg.poses = center_points
            path_msg.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(path_msg)
            
        res = self.solver(
            x0=initial,
            lbx=self.lower_bound_vars,
            ubx=self.upper_bound_vars,
            lbg=self.lower_bound_cons,
            ubg=self.upper_bound_cons,
            p=state
        )

        x = res['x']
        if show_predict:
            path_msg = Path()
            center_points=[]
            path_msg.header.frame_id = 'map'
            for i in range(self.k):
                index = 4*i
                control_index = 4*(self.k+1) + 2*i
                center_point = PoseStamped()
                center_point.header.frame_id = 'map'
                
                center_point.header.stamp = node.get_clock().now().to_msg()
                # Pose                                                                                                                                      
                center_point.pose.position.x = float(x[index])
                center_point.pose.position.y = float(x[index+1])
                center_point.pose.position.z = 0.
                # Compute orientation
                print("x=",float(x[index]),"y=",float(x[index+1]),"s=",float(x[index+3]),"delta=",float(x[control_index]),"u=",float(x[control_index+1]))
                yaw = float(x[index+2])
                center_point.pose.orientation.w = np.cos(.5 * yaw)
                center_point.pose.orientation.x = 0.
                center_point.pose.orientation.y = 0.
                center_point.pose.orientation.z = np.sin(.5 * yaw)
                center_points.append(center_point)
            path_msg.poses = center_points
            path_msg.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(path_msg)
        
        index = 4*(self.k+1)
        return [x[index],x[index+1]]

        




            
