#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using std::vector;
using std::max_element;

Mat display_histogram(Mat &src) {
    // 区间数目
    const int bins = 256;
    // 统计待处理图像的灰度级分布
    vector<int> gray_scales_counter(bins, 0);
    int rows = src.rows;
    int cols = src.cols;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            ++gray_scales_counter[src.at<uchar>(i, j)];
    // 计算归一化直方图
    int size = rows * cols;
    vector<double> histogram(bins, 0.0);
    for (int i = 0; i < bins; ++i)
        histogram[i] = 1.0 * gray_scales_counter[i] / size;
    // 获得直方图中的最大值
    double max_val = *max_element(histogram.begin(), histogram.end());
    // 绘制直方图
    int hist_h = bins;
    int factor = 2;
    int hist_w = bins * factor;
    Mat hist_image(hist_h, hist_w, CV_8UC1, Scalar(255));
    for (int i = 0; i < bins; ++i) {
        rectangle(hist_image, Point(i * factor, hist_h - 1), 
                  Point((i+1) * factor , hist_h - cvRound((histogram[i]/max_val) * hist_h)), 
                  Scalar(0), FILLED);
    }
    return hist_image;
}

void histogram_equalization(Mat &src, Mat &dst) {
    dst.create(src.size(), src.type());
    const int GRAY_SCALE = 256;
    int rows = src.rows;
    int cols = src.cols;
    // 统计输入图像的灰度级分布
    vector<int> r(GRAY_SCALE, 0);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            ++r[src.at<uchar>(i, j)];
    // 计算每个灰度级的概率
    int size = rows * cols;
    vector<double> pdf(GRAY_SCALE, 0.0);
    for (int i = 0; i < GRAY_SCALE; ++i)
        pdf[i] = 1.0 * r[i] / size;
    // 计算累积概率
    vector<double> cdf(GRAY_SCALE, 0.0);
    for (int i = 1; i < GRAY_SCALE; ++i)
        cdf[i] = cdf[i-1] + pdf[i];
    // 计算输出灰度级
    vector<int> s(GRAY_SCALE, 0);
    for (int i = 0; i < GRAY_SCALE; ++i)
        s[i] = saturate_cast<uchar>(GRAY_SCALE * cdf[i] + 0.5);
    // 输出图像
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst.at<uchar>(i, j) = s[src.at<uchar>(i, j)];
}

void rgb_histogram_equalization(Mat &src, Mat &dst) {
    dst.create(src.size(), src.type());
    const int N_CHANNELS = 3;
    const int GRAY_SCALE = 256;
    int row = src.rows;
    int col = src.cols;
    int size = row * col;
    vector<vector<int>>    r(N_CHANNELS, vector<int>(GRAY_SCALE, 0));
    vector<vector<int>>    s(N_CHANNELS, vector<int>(GRAY_SCALE, 0));
    vector<vector<double>> p(N_CHANNELS, vector<double>(GRAY_SCALE, 0.0));
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; ++j)
            for (int k = 0; k < N_CHANNELS; ++k)
                ++r[k][src.at<Vec3b>(i, j)[k]];
    for (int i = 0; i < N_CHANNELS; ++i)
        for (int j = 0; j < GRAY_SCALE; ++j)
            p[i][j] = 1.0 * r[i][j] / size;
    vector<vector<double>> prefix_sum (N_CHANNELS, vector<double>(GRAY_SCALE, 0.0));
    prefix_sum[0][0] = p[0][0];
    prefix_sum[1][0] = p[1][0];
    prefix_sum[2][0] = p[2][0];
    for (int i = 0; i < N_CHANNELS; ++i)
        for (int j = 1; j < GRAY_SCALE; ++j)
            prefix_sum[i][j] = prefix_sum[i][j-1] + p[i][j];
    for (int i = 0; i < N_CHANNELS; ++i)
        for (int j = 0; j < GRAY_SCALE; ++j)
            s[i][j] = saturate_cast<uchar>(GRAY_SCALE * prefix_sum[i][j] + 0.5);
    for (int i = 0; i < row; ++i)
        for (int j = 0; j < col; j++)
            for (int k = 0; k < N_CHANNELS; ++k)
                dst.at<Vec3b>(i, j)[k] = s[k][src.at<Vec3b>(i, j)[k]];
}

int main(int argc, char **argv) {
    Mat image_rgb = imread("images/taipei101.jpg");
    Mat image_gray;
    cvtColor(image_rgb, image_gray, COLOR_RGB2GRAY);
    // 显示归一化直方图
    imshow("Image", image_gray);
    Mat hist1 = display_histogram(image_gray);
    imshow("hist1", hist1);
    waitKey(0);
    destroyWindow("Image");
    destroyWindow("hist1");
    // 灰度图像直方图均衡化
    Mat image_gray_after_equalization;
    histogram_equalization(image_gray, image_gray_after_equalization);
    imshow("Image After Equalization", image_gray_after_equalization);
    Mat hist2 = display_histogram(image_gray_after_equalization);
    imshow("hist2", hist2);
    waitKey(0);
    destroyWindow("Image After Equalization");
    destroyWindow("hist2");
    // 彩色图像直方图均衡化
    imshow("rgb Image", image_rgb);
    Mat image_rgb_after_equalization;
    rgb_histogram_equalization(image_rgb, image_rgb_after_equalization);
    imshow("rgb Image After Equalization", image_rgb_after_equalization);
    waitKey(0);

    return 0;
}
