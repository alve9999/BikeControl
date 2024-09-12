#include <memory>
#include <rclcpp/publisher.hpp>
#include <visualization_msgs/msg/detail/marker__struct.hpp>
#include <time.h>
#include <chrono>
#include <vector>
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/path.hpp"
#include "std_msgs/msg/string.hpp"
#include "std_msgs/msg/float32.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include <cppad/cppad.hpp>
#include <cppad/ipopt/solve_result.hpp>
#include <cppad/ipopt/solve.hpp>
#include <Eigen/Core>
#include <Eigen/QR>
#include <visualization_msgs/msg/detail/marker_array__struct.hpp>
using std::placeholders::_1;
using namespace std::chrono_literals;
using CppAD::AD;

#define print(format, ...) RCLCPP_INFO(this->get_logger(), format, ##__VA_ARGS__)

#include <Eigen/Dense>
#define WHEELBASE 0.8
#define RADIUS 0.425
#define PI 3.14159265359

int N = 10; 
double dt = 0.1;

int x_start = 0;
int y_start = x_start + N;
int psi_start = y_start + N;
int cte_start = psi_start + N;
int epsi_start = cte_start + N;
int delta_start = epsi_start + N;
int u_start = delta_start + N - 1;

typedef struct {
  double speed;
  double steer;
} control_signal;
#include <iostream>

std::string statusToString(CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type status) {
    switch (status) {
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::not_defined: return "not_defined";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::success: return "success";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::maxiter_exceeded: return "maxiter_exceeded";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::stop_at_tiny_step: return "stop_at_tiny_step";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::stop_at_acceptable_point: return "stop_at_acceptable_point";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::local_infeasibility: return "local_infeasibility";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::user_requested_stop: return "user_requested_stop";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::feasible_point_found: return "feasible_point_found";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::diverging_iterates: return "diverging_iterates";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::restoration_failure: return "restoration_failure";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::error_in_step_computation: return "error_in_step_computation";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::invalid_number_detected: return "invalid_number_detected";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::too_few_degrees_of_freedom: return "too_few_degrees_of_freedom";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::internal_error: return "internal_error";
    case CppAD::ipopt::solve_result<CppAD::vector<double> >::status_type::unknown: return "unknown";
    default: return "Invalid status";
    }
}

class Bike_eval{
public:
  Eigen::VectorXd coeffs;
  Bike_eval(Eigen::VectorXd coeffs) {this->coeffs=coeffs;}

  typedef CPPAD_TESTVECTOR(AD<double>) ADvector;
  void operator()(ADvector& bike, const ADvector& vars) {
    bike[0] = 0;

    for(int i = 0; i < N; i++){
      bike[0] += 10 * CppAD::pow(vars[cte_start + i], 2);
      bike[0] += 1500 * CppAD::pow(vars[epsi_start + i], 2);
    }
    for(int i = 0; i < (N-1); i++){
      bike[0] += 100 * CppAD::pow(vars[u_start + i] - 3.0, 2);
    }

    bike[1 + x_start] = vars[x_start];
    bike[1 + y_start] = vars[y_start];
    bike[1 + psi_start] = vars[psi_start];
    
    for (int t = 1; t < N; t++) {
      
      AD<double> x0 = vars[x_start + t - 1];
      AD<double> y0 = vars[y_start + t - 1];
      AD<double> psi0 = vars[psi_start + t - 1];
      
      AD<double> x1 = vars[x_start + t];
      AD<double> y1 = vars[y_start + t];
      AD<double> psi1 = vars[psi_start + t];
      AD<double> cte = vars[cte_start + t];
      AD<double> epsi = vars[epsi_start + t];
      
      AD<double> delta0 = vars[delta_start + t - 1];
      AD<double> u0 = vars[u_start + t - 1];
      AD<double> true_velocity = u0*5.0*RADIUS;
      
      
      AD<double> position_along_polynomial = coeffs[4] * CppAD::pow(x0, 4) + coeffs[3] * CppAD::pow(x0, 3) + coeffs[2] * CppAD::pow(x0, 2) + coeffs[1] * x0 + coeffs[0];
      AD<double> derivative = 4 * coeffs[4] * CppAD::pow(x0,3) + 3 * coeffs[3] * CppAD::pow(x0, 2) + 2 * coeffs[2] * x0 + coeffs[1];
      AD<double> wanted_heading = CppAD::atan(derivative);
      
      bike[1 + x_start + t] = x1 - (x0 + true_velocity * CppAD::cos(delta0) * dt);
      bike[1 + y_start + t] = y1 - (y0 + true_velocity * CppAD::sin(delta0) * dt);
      bike[1 + psi_start + t] = psi1 - (psi0 - true_velocity * CppAD::tan(delta0) / WHEELBASE * dt);
      bike[1 + cte_start + t] = cte - ((position_along_polynomial - y0) + (true_velocity * CppAD::sin(epsi) * dt));
      bike[1 + epsi_start + t] = epsi - ((psi0 - wanted_heading) + true_velocity * CppAD::tan(delta0) / WHEELBASE * dt);
    }
  }
};


class Estimation : public rclcpp::Node
{
public:
  Estimation() : Node("estimation")
  {
    init();
    subscription1_ = this->create_subscription<visualization_msgs::msg::Marker>("gps/meas", 10, std::bind(&Estimation::kalman_filter, this, _1));
    subscription2_ = this->create_subscription<visualization_msgs::msg::Marker>("ground_truth", 10, std::bind(&Estimation::true_data, this, _1));
    subscription3_ = this->create_subscription<nav_msgs::msg::Path>("mapping/center_line", 10, std::bind(&Estimation::path_callback, this, _1));
    publisher1_ = this->create_publisher<visualization_msgs::msg::Marker>("state/viz", 10);
    publisher2_ = this->create_publisher<std_msgs::msg::Float32>("control/pedal", 10);
    publisher3_ = this->create_publisher<std_msgs::msg::Float32>("control/steer", 10);
	publisher4_ = this->create_publisher<visualization_msgs::msg::Marker>("control/path",10);
	timer1_ = this->create_wall_timer(500ms, std::bind(&Estimation::post_result, this));
  }

private:
  Eigen::MatrixXd observation_matrix;
  Eigen::MatrixXd observation_noise;
  Eigen::MatrixXd prosses_noise;
  Eigen::MatrixXd x;
  Eigen::MatrixXd true_x;
  Eigen::MatrixXd covariance;
  double avreage_error = 0;
  int n = 0;
  double u_speed = 0.0;
  double u_steering = 0.0;
  int current_closest_point = 0;
  rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr subscription1_;
  rclcpp::Subscription<visualization_msgs::msg::Marker>::SharedPtr subscription2_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr subscription3_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr publisher1_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher2_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher3_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr publisher4_;
  rclcpp::TimerBase::SharedPtr timer1_;
  rclcpp::Time last_measurement_;
  std::vector<Eigen::Vector2d> path_points_;
  bool first = true;

  void init()
  {
    last_measurement_ = this->now();
    //initialize kalman filter
    observation_matrix = Eigen::MatrixXd::Zero(2, 3);
    observation_matrix << 1, 0, 0,
                          0, 1, 0;
    prosses_noise = Eigen::MatrixXd::Zero(3, 3);
    prosses_noise << 0.0064, 0, 0,
                     0, 0.0064, 0,
                     0, 0, 0.00045;
    observation_noise = Eigen::MatrixXd::Zero(2, 2);
    observation_noise << 0.24, -0.012,
                         -0.012, 0.24;
    x = Eigen::MatrixXd::Zero(3, 1);
    x <<0,0,0;
    true_x = Eigen::MatrixXd::Zero(3, 1);
    covariance = Eigen::MatrixXd::Zero(3, 3);
    covariance << 0.1, 0, 0,
                  0, 0.1, 0,
                  0, 0, 0.1;
  }

  void predict()
  {
    auto now = this->now();
    rclcpp::Duration elapsed = now - last_measurement_;
    double dt = elapsed.seconds();
    last_measurement_ = now;
    double true_speed = u_speed*RADIUS*5;
    Eigen::MatrixXd F = Eigen::MatrixXd::Zero(3, 3);
    F << 1, 0, -true_speed*sin(x(2))*dt,
         0, 1, true_speed*cos(x(2))*dt,
         0, 0, 1;
    x(0) = x(0) + true_speed*cos(x(2))*dt;
    x(1) = x(1) + true_speed*sin(x(2))*dt;
    x(2) = x(2) + true_speed*tan(u_steering)/WHEELBASE*dt;
    covariance = F*covariance*F.transpose() + prosses_noise;
  }

  void update(Eigen::MatrixXd& z)
  {
    Eigen::MatrixXd heading_correction = Eigen::MatrixXd::Zero(2, 1);
    heading_correction << WHEELBASE/2*cos(x(2)), WHEELBASE/2*sin(x(2));
    Eigen::MatrixXd H(2, 3);
    H << 1 , 0, -WHEELBASE / 2 * sin(x(2)),
    0, 1, WHEELBASE / 2 * cos(x(2));
    Eigen::MatrixXd y = z - (observation_matrix*x + heading_correction);
    Eigen::MatrixXd S = H*covariance*H.transpose() + observation_noise;
    Eigen::MatrixXd K = covariance*H.transpose()*S.inverse();
    RCLCPP_INFO(this->get_logger(), "kalman gain (%f,%f,%f)", (K*y)(0), (K*y)(1),(K*y)(2));
    x = x + K*y;
    covariance = (Eigen::MatrixXd::Identity(3, 3) - K*H)*covariance;
  }

  void kalman_filter(const visualization_msgs::msg::Marker::SharedPtr msg)
  {
    Eigen::MatrixXd z = Eigen::MatrixXd::Zero(2, 1);
    z << msg->pose.position.x, msg->pose.position.y;
    RCLCPP_INFO(this->get_logger(), "True Position: (%f, %f)", true_x(0), true_x(1));
    RCLCPP_INFO(this->get_logger(), "Received Marker: ID = %d, Type = %d, Position = (%f, %f, %f)",msg->id, msg->type, msg->pose.position.x, msg->pose.position.y, msg->pose.position.z);
    if(first)
    {
      x << (msg->pose.position.x-WHEELBASE/2*cos(-PI/2)),
      (msg->pose.position.y-WHEELBASE/2*sin(-PI/2)),
      -PI/2;
      first = false;
    }
    predict();
    update(z);
    RCLCPP_INFO(this->get_logger(), "Estimated Position: (%f, %f, %f)", x(0), x(1), x(2));
    //add error to avreage
    if(x(0)!=0 and true_x(0)!=0){
      n++;
      double error = (x(0) - true_x(0)) * (x(0) - true_x(0)) + (x(1) - true_x(1)) * (x(1) - true_x(1));
      avreage_error = 0.3 * avreage_error + 0.7 * error;
    }
    RCLCPP_INFO(this->get_logger(), "Mean Squared Error: %f avreage: %f", (x(0)-true_x(0))*(x(0)-true_x(0))+(x(1)-true_x(1))*(x(1)-true_x(1)), avreage_error);
    control_bike();
  }
  
  void true_data(const visualization_msgs::msg::Marker::SharedPtr msg){
    true_x << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  }
  
  void post_result()
  {
    auto marker = visualization_msgs::msg::Marker();
    marker.header.frame_id = "map";
    marker.id = 1;
    marker.ns = "state";
    marker.header.stamp = this->now();
    marker.type = visualization_msgs::msg::Marker::ARROW;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.position.x = x(0);
    marker.pose.position.y = x(1);
    marker.pose.position.z = 0.0;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = sin(x(2)/2);
    marker.pose.orientation.w = cos(x(2)/2);
    marker.scale.x = 1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0;
    marker.color.r = 1.0;
    marker.color.g = 0.0;
    marker.color.b = 0.0;
    publisher1_->publish(marker);
  }

  void path_callback(const nav_msgs::msg::Path::SharedPtr msg){
    if(path_points_.empty()){
      path_points_.clear();
      for (const auto& pose : msg->poses) {
		Eigen::Vector2d point(pose.pose.position.x, pose.pose.position.y);
	
		RCLCPP_INFO(this->get_logger(), "loaded path %f %f",pose.pose.position.x,pose.pose.position.y);
		path_points_.push_back(point);
      }
    }
  }
  
  Eigen::VectorXd polyfit(Eigen::VectorXd xvals, Eigen::VectorXd yvals, int order) {
    Eigen::MatrixXd A(xvals.size(), order + 1);
    
    for (int i = 0; i < xvals.size(); i++) {
      A(i, 0) = 1.0;
    }
    
    for (int j = 0; j < xvals.size(); j++) {
      for (int i = 0; i < order; i++) {
	A(j, i + 1) = A(j, i) * xvals(j);
      }
    }

    auto Q = A.householderQr();
    auto result = Q.solve(yvals);
    return result;
  }
  
  Eigen::VectorXd poly_path(){
    Eigen::VectorXd xvals(50);
    Eigen::VectorXd yvals(50);
    for(int i = current_closest_point; i<(current_closest_point + 50) ;i++){
      xvals(i-current_closest_point)=path_points_[i%path_points_.size()](0);
      yvals(i-current_closest_point)=path_points_[i%path_points_.size()](1);
	  //RCLCPP_INFO(this->get_logger(),"x: %f, y: %f\n",xvals(i-current_closest_point),yvals(i-current_closest_point));
    }
    return polyfit(xvals,yvals,4);
  }
  
  void post_point_debug(double* x, double* y, int n)
  {
    auto marker = visualization_msgs::msg::Marker();
    marker.header.frame_id = "map";
    marker.header.stamp = this->now();
    marker.ns = "control";
    marker.id = 2;
    marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.2;
    marker.scale.y = 0.2;
    marker.scale.z = 0.2;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.color.a = 1.0;
    marker.frame_locked = true;
	for(int i = 0;i<n;i++){
	  geometry_msgs::msg::Point p;
	  p.x = x[i];
	  p.y = y[i];
	  p.z = 0;
	  
	  marker.points.push_back(p);
	  
	  std_msgs::msg::ColorRGBA c;
	  c.r = 0.0;
	  c.g = 0.0;
	  c.b = 1.0;
	  c.a = 1.0;
	  marker.colors.push_back(c);
	}
	publisher4_->publish(marker);
  }
  
  control_signal Solve(Eigen::VectorXd& coeff){
    typedef CPPAD_TESTVECTOR(double) Dvector;
    
    double x0 = x(0);
    double y0 = x(1);
    double psi0 = x(2);
    double cte0 = 0;
    double epsi0 = 0;

    int n_vars = 5 * N + 2 * (N-1);
    int n_constraints = 5 * N;

    Dvector vars(n_vars);
    for (int i = 0; i < n_vars; i++) {
      vars[i] = 0;
    }
    
    vars[x_start] = x0;
    vars[y_start] = y0;
    vars[psi_start] = psi0;
    Dvector vars_lowerbound(n_vars);
    Dvector vars_upperbound(n_vars);
    for (int i = 0; i < delta_start; i++) {
      vars_lowerbound[i] = -1.0e19;
      vars_upperbound[i] = 1.0e19;
    }
    
    for (int i = delta_start; i < u_start; i++) {
      vars_lowerbound[i] = -0.2566*0;
      vars_upperbound[i] = 0.2566*0;
    }

    for (int i = u_start; i < n_vars; i++) {
      vars_lowerbound[i] = 0.0;
      vars_upperbound[i] = 1.0*0;
    }

    Dvector constraints_lowerbound(n_constraints);
    Dvector constraints_upperbound(n_constraints);
    for (int i = 0; i < n_constraints; i++) {
      constraints_lowerbound[i] = 0;
      constraints_upperbound[i] = 0;
    }
    constraints_lowerbound[x_start] = x0;
    constraints_lowerbound[y_start] = y0;
    constraints_lowerbound[psi_start] = psi0;
    constraints_lowerbound[cte_start] = 0;
    constraints_lowerbound[epsi_start] = 0;
    
    constraints_upperbound[x_start] = x0;
    constraints_upperbound[y_start] = y0;
    constraints_upperbound[psi_start] = psi0;
    constraints_upperbound[cte_start] = 0;
    constraints_upperbound[epsi_start] = 0;
    Bike_eval bike_eval(coeff);
    
    std::string options;
    options += "Integer print_level  0\n";
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";
    options += "Numeric max_cpu_time          0.5\n";
    CppAD::ipopt::solve_result<Dvector> solution;
    
    CppAD::ipopt::solve<Dvector, Bike_eval>(options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound, constraints_upperbound, bike_eval, solution);
    control_signal c;

    std::cout << statusToString(solution.status) << std::endl;
        
    c.steer = solution.x[delta_start];
    c.speed = solution.x[u_start];
    
    return c;
  }

  void update_current_closest(){
    double min_distance = 1000;
    int closest = current_closest_point;
    for(int i = current_closest_point; i<(current_closest_point + path_points_.size()); i++){
      double distance = sqrt(pow(path_points_[i%(path_points_.size())](0) - x(0),2) + pow(path_points_[i%(path_points_.size())](1) - x(1),2));
	  //print("distace: %f x:%f y:%f px:%f py:%f\n",distance,x(0),x(1),path_points_[i%(path_points_.size())](0),path_points_[i%(path_points_.size())](1));
      if(distance<min_distance){
	    closest = i;
		min_distance=distance;
      }/*
      if((min_distance+1.0)<distance){
		break;
	  }*/
    }
    current_closest_point = closest;
  }

  void control_bike(){
    update_current_closest();
    Eigen::VectorXd coeff = poly_path();

	visualization_msgs::msg::MarkerArray post;
	const int n = 10;
	double y[n];
	double x[n];
	for(double i = 0.0; i<0.1;i+=0.01){
	  RCLCPP_INFO(this->get_logger(),"polynomial %f %f %f %f %f\n",coeff[0],coeff[1],coeff[2],coeff[3],coeff[4]);
	  RCLCPP_INFO(this->get_logger(),"point:%f,%f",i,coeff[4] * CppAD::pow(i, 4) + coeff[3] * CppAD::pow(i, 3) + coeff[2] * CppAD::pow(i, 2) + coeff[1] * i + coeff[0]);
	  double x_val=coeff[4] * pow(i, 4) + coeff[3] * pow(i, 3) + coeff[2] * pow(i, 2) + coeff[1] * i + coeff[0];
	  x[(int)i*100] = x_val;
	  y[(int)i*100] = i;
	}
	post_point_debug(x,y,n);
	control_signal res = Solve(coeff);
    //speed_control(res.speed);
    //steer_control(res.steer);
  }
  
  
  void speed_control(double speed){
    auto msg = std::make_unique<std_msgs::msg::Float32>();
	u_speed = speed;
    msg->data = speed;
    publisher2_->publish(std::move(msg));

  }
  void steer_control(double steer){
    auto msg = std::make_unique<std_msgs::msg::Float32>();
    u_steering = steer;
    msg->data = steer;
    publisher3_->publish(std::move(msg));
  }
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Estimation>());
  rclcpp::shutdown();
  return 0;
}
