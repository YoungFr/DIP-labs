#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using std::cout;
using std::endl;

void global_threshold_binaryzation(Mat &src, Mat &dst);
void log_transform(Mat &src, Mat &dst, double c);
void gamma_transform(Mat &src, Mat &dst, double c, double gamma);
void rgb_comp_transform(Mat &src, Mat &dst);

int main() {
    // 1. 读取图像并显示
    Mat image1 = imread("images/bird.jpg");
    if (image1.empty()) {
        cout << "Could not read the image." << endl;
        return 1;
    }
    imshow("Image", image1);
    waitKey(0);
    destroyWindow("Image");

    // 2. 灰度图像二值化处理
    Mat image2 = imread("images/bird.jpg", IMREAD_GRAYSCALE);
    if (image2.empty()) {
        cout << "Could not read the image." << endl;
        return 1;
    }
    imshow("Image Before Binaryzation", image2);
    global_threshold_binaryzation(image2, image2);
    imshow("Image After Binaryzation",  image2);
    waitKey(0);
    destroyWindow("Image Before Binaryzation");
    destroyWindow("Image After Binaryzation");

    // 3. 对数变换
    Mat image3 = imread("images/log.jpg", IMREAD_GRAYSCALE);
    if (image3.empty()) {
        cout << "Could not read the image." << endl;
        return 1;
    }
    imshow("Image Before Log Transformation", image3);
    log_transform(image3, image3, 27.0);
    imshow("Image After Log Transformation",  image3);
    waitKey(0);
    destroyWindow("Image Before Log Transformation");
    destroyWindow("Image After Log Transformation");

    // 4. 伽马变换
    Mat image4 = imread("images/gamma.jpg", IMREAD_GRAYSCALE);
    if (image4.empty()) {
        cout << "Could not read the image." << endl;
        return 1;
    }
    imshow("Image Before Gamma Transformation", image4);
    gamma_transform(image4, image4, 0.25, 1.2);
    imshow("Image After Gamma Transformation",  image4);
    waitKey(0);
    destroyWindow("Image Before Gamma Transformation");
    destroyWindow("Image After Gamma Transformation");

    // 5. 补色变换
    Mat image5 = imread("images/fruits.jpg");
    if (image5.empty()) {
        cout << "Could not read the image." << endl;
        return 1;
    }
    imshow("Image Before Complementary Transformation", image5);
    rgb_comp_transform(image5, image5);
    imshow("Image After Complementary Transformation", image5);
    waitKey(0);

    return 0;
}

void global_threshold_binaryzation(Mat &src, Mat &dst) {
    dst = src.clone();
    const int GRAY_SCALE = 256;
    const double delta_T = 0.1;
    int histogram[GRAY_SCALE] = {0};
    // 直方图统计
    for (int i = 0; i < src.rows; ++i) {
        uchar *p = src.ptr<uchar>(i);
        for (int j = 0; j < src.cols; ++j)
            ++histogram[p[j]];
    }
    // 构造前缀和数组用于计算平均灰度值
    int cnt[GRAY_SCALE] = {histogram[0]};
    for (int i = 1; i < GRAY_SCALE; ++i)
        cnt[i] = cnt[i-1] + histogram[i];
    double sum[GRAY_SCALE] = {0.0};
    for (int i = 1; i < GRAY_SCALE; ++i)
        sum[i] = sum[i-1] + histogram[i] * i;
    // 迭代计算全局阈值
    double old_threshold = mean(src)[0];
    double new_threshold = 0.0;
    for (;;) {
        int t = static_cast<int>(old_threshold + 0.5);
        double m1 = sum[t] / cnt[t];
        double m2 = (sum[GRAY_SCALE-1] - sum[t]) / 
                    (cnt[GRAY_SCALE-1] - cnt[t]);
        new_threshold = (m1 + m2) / 2;
        if (abs(new_threshold - old_threshold) < delta_T)
            break;
        else
            old_threshold = new_threshold;
    }
    // 根据计算出的阈值对图像进行二值化处理
    int threshold = static_cast<int>(new_threshold + 0.5);
    for (int i = 0; i < dst.rows; ++i) {
        uchar *p = dst.ptr<uchar>(i);
        for (int j = 0; j < dst.cols; ++j)
            p[j] = (p[j] > threshold) ? 255 : 0;
    }
}

void log_transform(Mat &src, Mat &dst, double c) {
    dst.create(src.size(), src.type());
    for (int i = 0; i < dst.rows; ++i)
        for (int j = 0; j < dst.cols; ++j)
            dst.at<uchar>(i, j) = saturate_cast<uchar>(c * log(1.0 + src.at<uchar>(i, j)));
}

void gamma_transform(Mat &src, Mat &dst, double c, double gamma) {
    dst.create(src.size(), src.type());
    for (int i = 0; i < src.rows; ++i)
        for (int j = 0; j < src.cols; ++j)
            dst.at<uchar>(i, j) = saturate_cast<uchar>(c * pow(src.at<uchar>(i, j), gamma));
}

void rgb_comp_transform(Mat &src, Mat &dst) {
    dst.create(src.size(), src.type());
    for (int i = 0; i < dst.rows; ++i) {
        for (int j = 0; j < dst.cols; ++j) {
            dst.at<Vec3b>(i, j)[0] = 255 - src.at<Vec3b>(i, j)[0];
            dst.at<Vec3b>(i, j)[1] = 255 - src.at<Vec3b>(i, j)[1];
            dst.at<Vec3b>(i, j)[2] = 255 - src.at<Vec3b>(i, j)[2];
        }
    }
}
