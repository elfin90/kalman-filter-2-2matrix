#include "subs.hpp"

int main(int argc, char **argv) {
    // Initialize the ROS node
    ros::init(argc, argv, "yolo_result1_node");

    // Create a node handle
    ros::NodeHandle nh;

    // Create an instance of SUBS
    subs::SUBS subs_instance(nh);
    
    // Spin and process callbacks
    ros::spin();

    return 0;
}

