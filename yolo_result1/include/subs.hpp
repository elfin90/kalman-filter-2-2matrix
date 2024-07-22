#ifndef SUBS_HPP_
#define SUBS_HPP_

#include <ros/ros.h>
#include "yolo_result1/YoloResult.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>



namespace subs {

// Kalman filtresi sınıfı
class KalmanFilter {
private:
    // Durum vektörü (x, y, vx, vy)(degistirdim)
    Eigen::VectorXd x_hat;

    // Durum geçiş matrisi (A)
    Eigen::MatrixXd A;

    // Ölçüm matrisi (H)
    Eigen::MatrixXd C;

    // Süreç gürültüsü kovaryans matrisi (Q)
    Eigen::MatrixXd Q;

    // Ölçüm gürültüsü kovaryans matrisi (R)
    Eigen::MatrixXd R;

    // Hata kovaryansı (P)
    Eigen::MatrixXd P_old; 
    Eigen::VectorXd x_hat_new;
    Eigen::MatrixXd P_new;
    Eigen::VectorXd x_est;
    Eigen::MatrixXd P_est;

public:
    // Kurucu fonksiyon
    KalmanFilter();
    //Tahmin Fonksiyonu
    void predict();
    // Güncelleme fonksiyonu
    void update(const Eigen::VectorXd& measurement);
    

    // Durum vektörünü almak için get fonksiyonu
    Eigen::VectorXd getState();

    // Hata kovaryans matrisini almak için get fonksiyonu
    Eigen::MatrixXd getCovariance();
};

// SUBS sınıfı
class SUBS {
public:
    // Constructor
    SUBS(ros::NodeHandle& t_node_handle);

    // Callback fonksiyonu
    void customMsgCallback(const yolo_result1::YoloResult::ConstPtr& msg);
    void visualizeState(const Eigen::VectorXd& state);
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    
private:
    ros::NodeHandle nh_;
    ros::Subscriber sub;
    image_transport::Subscriber image_sub_;
    cv::Mat cv_image_;
    // Kalman filtresi örneği
    KalmanFilter kf;
};

} // namespace subs

#endif // SUBS_HPP_
