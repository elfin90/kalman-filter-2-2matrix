#include "subs.hpp"

namespace subs {

KalmanFilter::KalmanFilter() {
    // Başlangıç durumu (x, y)
    x_hat << 0, 0;

    // Durum geçiş matrisi (A)
    A << 1,0,
         0,1;

    // Ölçüm matrisi (C)
    C << 1, 0, 
         0, 1;

    // Süreç gürültüsü kovaryans matrisi (Q)
    Q << 0.1,0,
         0,0.1;

    // Ölçüm gürültüsü kovaryans matrisi (R)
    R << 1, 0,
         0, 1;

    // Başlangıç hata kovaryansı (P)
    P_old = Eigen::Matrix2d::Identity();
}

void KalmanFilter::predict() {
    // prediction part(did not take the control vector(u) and control matrix(B))
    x_hat_new = A * x_hat;
    P_new = A * P_old * A.transpose() + Q;
}

void KalmanFilter::update(const Eigen::Vector2d& measurement) {
    // Kalman kazancı
    Eigen::Matrix<double, 2, 2> K = P_new * C.transpose() * (C * P_new * C.transpose() + R).inverse();

    // Ölçüm hatası (yenilik)
    Eigen::Vector2d y = measurement - C * x_hat_new;

    // estimation values
    x_est = x_hat_new + K * y;
    P_est = (Eigen::Matrix2d::Identity() - K * C) * P_new;
    
    x_hat = x_est;
    P_old = P_est;
}
Eigen::Vector2d KalmanFilter::getState(){
    return x_est;
}

Eigen::Matrix2d KalmanFilter::getCovariance(){
    return P_est;
}

SUBS::SUBS(ros::NodeHandle& nh) : nh_(nh) {
    sub = nh_.subscribe("yolo_result", 100, &SUBS::customMsgCallback, this);
    image_transport::ImageTransport it(nh_);
    image_sub_ = it.subscribe("yolo_image", 1, &SUBS::imageCallback, this);
}

void SUBS::customMsgCallback(const yolo_result1::YoloResult::ConstPtr& msg) {
    if (!msg->detections.detections.empty()) {
        // İlk tespit edilen nesnenin merkez koordinatlarını al
        float x = msg->detections.detections[0].bbox.center.x;
        float y = msg->detections.detections[0].bbox.center.y;
        ROS_INFO("Center: x=%f, y=%f", x, y);

        // Kalman filter tahmini
        kf.predict();

        // Kalman filter güncellemesi
        Eigen::Vector2d measurement;
        measurement << x, y;
        kf.update(measurement);

        // Durum ve kovaryans matrisi bilgilerini yazdır
        Eigen::Vector2d state = kf.getState();
        Eigen::Matrix2d covariance = kf.getCovariance();
        ROS_INFO("State: [%f, %f]", state[0], state[1]);
        ROS_INFO("Covariance: \n[%f, %f]\n[%f, %f]",
            covariance(0,0), covariance(0,1), 
            covariance(1,0), covariance(1,1));
        visualizeState(state);  
    } else {
        kf.predict();

        // Durum ve kovaryans matrisi bilgilerini yazdır
        Eigen::Vector2d state = kf.getState();
        Eigen::Matrix2d covariance = kf.getCovariance();
        ROS_INFO("Predicted state: [%f, %f]", state[0], state[1]);
        ROS_INFO("Covariance: \n[%f, %f]\n[%f, %f]",
            covariance(0,0), covariance(0,1), 
            covariance(1,0), covariance(1,1));
        visualizeState(state);
    }
}

void SUBS::visualizeState(const Eigen::Vector2d& state) {
    cv::Mat img = cv::Mat::zeros(500,500,CV_8UC3);
    cv::circle(img,cv::Point(state[0],state[1]),10,cv::Scalar(0,255,0),-1);
    cv::imshow("Kalman Filter State",img);
    cv::waitKey(1);
}

void SUBS::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        // ROS görüntü mesajını OpenCV görüntüsüne dönüştür
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv_image_ = cv_ptr->image;

        // Kalman filtresi tahminini al
        Eigen::Vector2d state = kf.getState();

        // OpenCV ile tahmin değerlerine göre nokta çizimi
        cv::Point point(state[0], state[1]); // x ve y koordinatlarını kullanın
        cv::circle(cv_image_, point, 5, cv::Scalar(0, 255, 0), -1);

        // Çizilmiş görüntüyü ekranda göster
        cv::imshow("Image with Point", cv_image_);
        cv::waitKey(1); // Ekrandaki görüntünün güncellenmesi için bekleme

    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s",e.what());
        return;
    }
}

} // namespace subs
