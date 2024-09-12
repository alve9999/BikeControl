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
#include <Eigen/Core>
#include <Eigen/QR>
#include "messages/msg/state.hpp"
using std::placeholders::_1;
using namespace std::chrono_literals;


#define print(format, ...) RCLCPP_INFO(this->get_logger(), format, ##__VA_ARGS__)

#include <Eigen/Dense>
#define WHEELBASE 0.8
#define RADIUS 0.425
#define PI 3.14159265359

typedef struct {
  double speed;
  double steer;
} control_signal;

class Estimation : public rclcpp::Node
{
public:
  Estimation() : Node("estimation")
  {
    init();
    subscription1_ = this->create_subscription<visualization_msgs::msg::Marker>("gps/meas", 10, std::bind(&Estimation::kalman_filter, this, _1));
    subscription2_ = this->create_subscription<visualization_msgs::msg::Marker>("ground_truth", 10, std::bind(&Estimation::true_data, this, _1));
	subscription3_ = this->create_subscription<std_msgs::msg::Float32>("/control/steer", 10, std::bind(&Estimation::steer_data, this, _1));
	subscription4_ = this->create_subscription<std_msgs::msg::Float32>("/control/pedal", 10, std::bind(&Estimation::speed_data, this, _1));
	
    publisher1_ = this->create_publisher<visualization_msgs::msg::Marker>("/state/viz", 10);
	publisher2_ = this->create_publisher<messages::msg::State>("/state/est", 10);
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
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subscription3_;
  rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr subscription4_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr publisher1_;
  rclcpp::Publisher<messages::msg::State>::SharedPtr publisher2_;
  rclcpp::TimerBase::SharedPtr timer1_;
  rclcpp::Time last_measurement_;
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
	auto state = messages::msg::State();
	state.x=x(0);
	state.y=x(1);
	state.heading=x(2);
	publisher2_->publish(state);
    RCLCPP_INFO(this->get_logger(), "Estimated Position: (%f, %f, %f)", x(0), x(1), x(2));
    //add error to avreage
    if(x(0)!=0 and true_x(0)!=0){
      n++;
      double error = (x(0) - true_x(0)) * (x(0) - true_x(0)) + (x(1) - true_x(1)) * (x(1) - true_x(1));
      avreage_error = 0.3 * avreage_error + 0.7 * error;
    }
    RCLCPP_INFO(this->get_logger(), "Mean Squared Error: %f avreage: %f", (x(0)-true_x(0))*(x(0)-true_x(0))+(x(1)-true_x(1))*(x(1)-true_x(1)), avreage_error);
  }
  
  void true_data(const visualization_msgs::msg::Marker::SharedPtr msg){
    true_x << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  }

  void speed_data(std_msgs::msg::Float32 speed){
	this->u_speed=speed.data;
  }

  void steer_data(std_msgs::msg::Float32 steer){
	this->u_steering=steer.data;
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
};


int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Estimation>());
  rclcpp::shutdown();
  return 0;
}
