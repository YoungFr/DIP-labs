
#include <vector>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using std::vector;
using std::cout;
using std::endl;

void convolution(Mat &src, vector<vector<double>> &dst, 
                 vector<vector<double>> kernel, int border_type);
void convolution(Mat &src, vector<vector<Vec3d>>  &dst, 
                 vector<vector<double>> kernel, int border_type);
vector<vector<double>> gaussian_kernel(int m, double sigma);
void blur(Mat &src, Mat &dst, int m, double sigma);
void laplacian_sharpen(Mat &src, Mat &dst, vector<vector<double>> kernel);
void gradient_sharpen(Mat &src, Mat &dst, 
                      vector<vector<double>> kx, 
                      vector<vector<double>> ky);
void highboost_filter(Mat &src, Mat &dst, int m, double sigma, double k);

int main() {
    Mat image = imread("images/boldt.jpg", IMREAD_GRAYSCALE);

    // blur a gray-scale image using 3*3, 5*5, 9*9 box kernel
    imshow("image", image);
    Mat image_box;
    blur(image, image_box, 3, 0);
    imshow("3*3 box kernel", image_box);
    blur(image, image_box, 5, 0);
    imshow("5*5 box kernel", image_box);
    blur(image, image_box, 9, 0);
    imshow("9*9 box kernel", image_box);
    waitKey(0);
    destroyAllWindows();

    // blur a gray-scale image using 3*3, 5*5, 9*9 gaussian kernel(m == sigma)
    imshow("image", image);
    Mat image_gas;
    blur(image, image_gas, 3, 3);
    imshow("3*3 gaussian kernel", image_gas);
    blur(image, image_gas, 5, 5);
    imshow("5*5 gaussian kernel", image_gas);
    blur(image, image_gas, 9, 9);
    imshow("9*9 gaussian kernel", image_gas);
    waitKey(0);
    destroyAllWindows();

    // blur a rgb image using 3*3, 5*5, 9*9 box kernel
    Mat image_rgb = imread("images/boldt.jpg");

    imshow("image rgb", image_rgb);
    Mat image_rgb_box;
    blur(image_rgb, image_rgb_box, 3, 0);
    imshow("3*3 box kernel rgb", image_rgb_box);
    blur(image_rgb, image_rgb_box, 5, 0);
    imshow("5*5 box kernel rgb", image_rgb_box);
    blur(image_rgb, image_rgb_box, 9, 0);
    imshow("9*9 box kernel rgb", image_rgb_box);
    waitKey(0);
    destroyAllWindows();

    // blur a rgb image using 3*3, 5*5, 9*9 gaussian kernel
    Mat image_rgb_gas;
    imshow("image rgb", image_rgb);
    blur(image_rgb, image_rgb_gas, 3, 3);
    imshow("3*3 gaussian kernel rgb", image_rgb_gas);
    blur(image_rgb, image_rgb_gas, 5, 5);
    imshow("5*5 gaussian kernel rgb", image_rgb_gas);
    blur(image_rgb, image_rgb_gas, 9, 9);
    imshow("9*9 gaussian kernel rgb", image_rgb_gas);
    waitKey(0);
    destroyAllWindows();

    vector<vector<double>> laplacian1 = {{ 0, 1, 0},{ 1,-3, 1},{ 0, 1, 0}};
    vector<vector<double>> laplacian2 = {{ 1, 1, 1},{ 1,-7, 1},{ 1, 1, 1}};
    vector<vector<double>> laplacian3 = {{ 0,-1, 0},{-1, 5,-1},{ 0,-1, 0}};
    vector<vector<double>> laplacian4 = {{-1,-1,-1},{-1, 9,-1},{-1,-1,-1}};

    Mat lap;
    imshow("image", image);
    laplacian_sharpen(image, lap, laplacian1);
    imshow("laplacian 1", lap);
    laplacian_sharpen(image, lap, laplacian2);
    imshow("laplacian 2", lap);
    laplacian_sharpen(image, lap, laplacian3);
    imshow("laplacian 3", lap);
    laplacian_sharpen(image, lap, laplacian4);
    imshow("laplacian 4", lap);
    waitKey(0);
    destroyAllWindows();

    Mat lap_rgb;
    imshow("image rgb", image_rgb);
    laplacian_sharpen(image_rgb, lap_rgb, laplacian1);
    imshow("laplacian 1 rgb", lap_rgb);
    laplacian_sharpen(image_rgb, lap_rgb, laplacian2);
    imshow("laplacian 2 rgb", lap_rgb);
    laplacian_sharpen(image_rgb, lap_rgb, laplacian3);
    imshow("laplacian 3 rgb", lap_rgb);
    laplacian_sharpen(image_rgb, lap_rgb, laplacian4);
    imshow("laplacian 4 rgb", lap_rgb);
    waitKey(0);
    destroyAllWindows();
    
    Mat people = imread("images/ren.jpg", IMREAD_GRAYSCALE);
    Mat imageHigh;
    imshow("people", people);
    highboost_filter(people, imageHigh, 15, 4, 1.8);
    imshow("image highboost", imageHigh);
    waitKey(0);
    destroyAllWindows();

    vector<vector<double>> robertx = {{ 0, 0, 0},{ 0,-1, 0},{ 0, 0, 1}};
    vector<vector<double>> roberty = {{ 0, 0, 0},{ 0, 0,-1},{ 0, 1, 0}};
    vector<vector<double>> sobelx  = {{-1,-2,-1},{ 0, 0, 0},{ 1, 2, 1}};
    vector<vector<double>> sobely  = {{-1, 0, 1},{-2, 0, 2},{-1, 0, 1}};

    Mat image_rob;
    imshow("image", image);
    gradient_sharpen(image, image_rob, robertx, roberty);
    imshow("image robert", image_rob);
    Mat image_rgb_rob;
    imshow("image rgb", image_rgb);
    gradient_sharpen(image_rgb, image_rgb_rob, robertx, roberty);
    imshow("image rgb robert", image_rgb_rob);
    waitKey(0);
    destroyAllWindows();

    Mat image_sob;
    imshow("image", image);
    gradient_sharpen(image, image_sob, sobelx, sobely);
    imshow("image sobel", image_sob);
    Mat image_rgb_sob;
    imshow("image rgb", image_rgb);
    gradient_sharpen(image_rgb, image_rgb_sob, sobelx,sobely);
    imshow("image rgb sobel", image_rgb_sob);
    waitKey(0);
    destroyAllWindows();
    
    return 0;
}

/// @brief convolution operation to a gray-scale image using an opertor
/// @param src input image
/// @param dst the result of convolution, it will be used by the caller
/// @param kernel the operator
/// @param border_type the borderType, using BORDER_CONSTANT or 
///                    BORDER_REPLICATE or BORDER_REFLECT
void convolution(Mat &src, vector<vector<double>> &dst, 
                 vector<vector<double>> kernel, int border_type) {
    CV_Assert(src.type() == CV_8UC1  && 
              src.rows == dst.size() && 
              src.cols == dst[0].size());
    int border = (kernel.size() - 1) / 2;
    int i_dst  = 0;
    int j_dst  = 0;
    double con = 0;
    Mat filled;
    // form the border around the image
    copyMakeBorder(src, filled, border, border, border, border, border_type);
    int rows = src.rows + 2 * border;
    int cols = src.cols + 2 * border;
    // process each pixel
    for (int i = border; i < rows - border; ++i) {
        for (int j = border; j < cols - border; ++j) {
            i_dst = i - border;
            j_dst = j - border;
            // convolution operation
            con = 0;
            for (int ki = i - border; ki <= i + border; ++ki)
                for (int kj = j - border; kj <= j + border; ++kj)
                    con += (kernel[ki-i_dst][kj-j_dst] * 
                            filled.at<uchar>(ki, kj));
            dst[i_dst][j_dst] = con;
        }
    }
}

/// @brief convolution operation to a rgb image using an opertor
/// @param src input image
/// @param dst the result of convolution, it will be used by the caller
/// @param kernel the operator
/// @param border_type the borderType, using BORDER_CONSTANT or 
///                    BORDER_REPLICATE or BORDER_REFLECT
void convolution(Mat &src, vector<vector<Vec3d>> &dst, 
                 vector<vector<double>> kernel, int border_type) {
    CV_Assert(src.type() == CV_8UC3  && 
              src.rows == dst.size() && 
              src.cols == dst[0].size());
    int border = (kernel.size() - 1) / 2;
    int i_dst  = 0;
    int j_dst  = 0;
    Mat filled;
    vector<double> cons(3, 0);
    // form the border around the image
    copyMakeBorder(src, filled, border, border, border, border, border_type);
    int rows = src.rows + 2 * border;
    int cols = src.cols + 2 * border;
    // process each pixel
    for (int i = border; i < rows - border; ++i) {
        for (int j = border; j < cols - border; ++j) {
            i_dst = i - border;
            j_dst = j - border;
            // convolution operation
            cons[0] = cons[1] = cons[2] = 0;
            for (int ki = i - border; ki <= i + border; ++ki)
                for (int kj = j - border; kj <= j + border; ++kj)
                    for (int ch = 0; ch < 3; ++ch)
                        cons[ch] += (kernel[ki-i_dst][kj-j_dst] * 
                                     filled.at<Vec3b>(ki, kj)[ch]);
            dst[i_dst][j_dst] = Vec<double, 3>(cons[0], cons[1], cons[2]);
        }
    }
}

/// @brief get the gaussian kernel used to blur an image.
/// @param m the size of gaussian kernel is m*m,and m is an odd number
/// @param sigma the standard deviation of gaussian function
/// @return a reference to a normalized gaussian kernel used to blur an image
vector<vector<double>> gaussian_kernel(int m, double sigma) {
    vector<vector<double>> kernel(m, vector<double>(m, 0));
    double sum = 0;
    double r_square = 0;
    double center = (m - 1) / 2;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            r_square = pow((i - center), 2) + pow((j - center), 2);
            kernel[i][j] = exp(-(r_square / (2 * sigma * sigma)));
            sum += kernel[i][j];
        }
    }
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < m; ++j)
            kernel[i][j] /= sum;
    return kernel;
}

/// @brief blur an gray-scale or rgb image using a box or gaussian kernel,
///        the in-place processing is supported.
/// @param src input image
/// @param dst output image
/// @param m the size of kernel is m*m,and m is an odd number
/// @param sigma the standard deviation of gaussian function, if sigma == 0,
///              we use a box kernel
void blur(Mat &src, Mat &dst, int m, double sigma) {
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
    // create a box kernel
    const double elem = 1.0 / m / m;
    vector<vector<double>> box_kernel(m, vector<double>(m, elem));
    if (src.type() == CV_8UC1) {
        // gray-scale image
        vector<vector<double>> tmp_dst(src.rows, vector<double>(src.cols, 0));
        if (sigma == 0)
            convolution(src, tmp_dst, box_kernel, BORDER_REFLECT);
        else
            convolution(src, tmp_dst, gaussian_kernel(m, sigma), BORDER_REFLECT);
        dst.create(src.size(), CV_8UC1);
        for (int i = 0; i < dst.rows; ++i)
            for (int j = 0; j < dst.cols; ++j)
                dst.at<uchar>(i, j) = saturate_cast<uchar>(tmp_dst[i][j]);
    } else {
        // rgb image
        vector<vector<Vec3d>> tmp_dst(src.rows, vector<Vec3d>(src.cols, Vec3d(0, 0, 0)));
        if (sigma == 0)
            convolution(src, tmp_dst, box_kernel, BORDER_REFLECT);
        else
            convolution(src, tmp_dst, gaussian_kernel(m, sigma), BORDER_REFLECT);
        dst.create(src.size(), CV_8UC3);
        for (int i = 0; i < dst.rows; ++i)
            for (int j = 0; j < dst.cols; ++j)
                for (int ch = 0; ch < 3; ++ch)
                    dst.at<Vec3b>(i, j)[ch] = saturate_cast<uchar>(tmp_dst[i][j][ch]);
    }
}

/// @brief sharpen an gray-scale or rgb image using a laplacian operator,
///        the in-place processing is supported.
/// @param src input image
/// @param dst output image
/// @param kernel the laplacian operator
void laplacian_sharpen(Mat &src, Mat &dst, vector<vector<double>> kernel) {
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
    if (src.type() == CV_8UC1) {
        // gray-scale image
        vector<vector<double>> tmp_dst(src.rows, vector<double>(src.cols, 0));
        convolution(src, tmp_dst, kernel, BORDER_REFLECT);
        dst.create(src.size(), CV_8UC1);
        for (int i = 0; i < dst.rows; ++i)
            for (int j = 0; j < dst.cols; ++j)
                dst.at<uchar>(i, j) = saturate_cast<uchar>(tmp_dst[i][j]);
    } else {
        // rgb image
        vector<vector<Vec3d>> tmp_dst(src.rows, vector<Vec3d>(src.cols, Vec3d(0, 0, 0)));
        convolution(src, tmp_dst, kernel, BORDER_REFLECT);
        dst.create(src.size(), CV_8UC3);
        for (int i = 0; i < dst.rows; ++i)
            for (int j = 0; j < dst.cols; ++j)
                for (int ch = 0; ch < 3; ++ch)
                    dst.at<Vec3b>(i, j)[ch] = saturate_cast<uchar>(tmp_dst[i][j][ch]);
    }
}

/// @brief sharpen an gray-scale or rgb image using a robert or sobel operator,
///        the in-place processing is supported.
/// @param src input image
/// @param dst output image
/// @param kx derivative x operator
/// @param ky derivative y operator
void gradient_sharpen(Mat &src, Mat &dst, 
                      vector<vector<double>> kx, 
                      vector<vector<double>> ky) {
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
    const int rows = src.rows;
    const int cols = src.cols;
    if (src.type() == CV_8UC1) {
        // gray-scale image
        vector<vector<double>> gx (rows, vector<double>(cols, 0));
        vector<vector<double>> gy (rows, vector<double>(cols, 0));
        vector<vector<double>> Mxy(rows, vector<double>(cols, 0));
        convolution(src, gx, kx, BORDER_REFLECT);
        convolution(src, gy, ky, BORDER_REFLECT);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                Mxy[i][j] = abs(gx[i][j]) + abs(gy[i][j]);
        dst.create(src.size(), CV_8UC1);
        for (int i = 0; i < dst.rows; ++i)
            for (int j = 0; j < dst.cols; ++j)
                dst.at<uchar>(i, j) = saturate_cast<uchar>(Mxy[i][j]);
    } else {
        // rgb image
        vector<vector<Vec3d>> gx (rows, vector<Vec3d>(cols, Vec3d(0, 0, 0)));
        vector<vector<Vec3d>> gy (rows, vector<Vec3d>(cols, Vec3d(0, 0, 0)));
        vector<vector<Vec3d>> Mxy(rows, vector<Vec3d>(cols, Vec3d(0, 0, 0)));
        convolution(src, gx, kx, BORDER_REFLECT);
        convolution(src, gy, ky, BORDER_REFLECT);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j)
                for (int ch = 0; ch < 3; ++ch)
                    Mxy[i][j][ch] = abs(gx[i][j][ch]) + abs(gy[i][j][ch]);
        dst.create(src.size(), CV_8UC3);
        for (int i = 0; i < dst.rows; ++i)
            for (int j = 0; j < dst.cols; ++j)
                for (int ch = 0; ch < 3; ++ch)
                    dst.at<Vec3b>(i, j)[ch] = saturate_cast<uchar>(Mxy[i][j][ch]);
    }
}

void highboost_filter(Mat &src, Mat &dst, int m, double sigma, double k) {
    CV_Assert(src.type() == CV_8UC1);
    const int rows = src.rows;
    const int cols = src.cols;
    Mat blurred;
    blur(src, blurred, m, sigma);
    Mat mask (rows, cols, CV_8UC1);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mask.at<uchar>(i, j) = 
            saturate_cast<uchar>(src.at<uchar>(i, j) - blurred.at<uchar>(i, j));
    imshow("mask", mask);
    Mat temp_src = src.clone();
    dst.create(src.size(), CV_8UC1);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            dst.at<uchar>(i, j) = 
            saturate_cast<uchar>(k * mask.at<uchar>(i, j) + temp_src.at<uchar>(i, j));
}
