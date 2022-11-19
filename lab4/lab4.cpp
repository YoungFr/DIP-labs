#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <random>

using namespace cv;
using std::cout;
using std::endl;
using std::vector;
using std::default_random_engine;
using std::uniform_int_distribution;
using std::normal_distribution;

void add_gaus_noise(const Mat &src, Mat &dst, double sigma);
void add_pesa_noise(const Mat &src, Mat &dst, double p_pepr, double p_salt);
void geometric_mean_filter(const Mat &src, Mat &dst, int m, int n);
void contraharmonic_mean_filter(const Mat &src, Mat &dst, int m, int n, double Q);
void median_filter(const Mat &src, Mat &dst, int m, int n);
void adaptive_mean_filter(const Mat &src, Mat &dst, int m, int n);
void adaptive_median_filter(const Mat &src, Mat &dst, int init_m, int init_n, int max_m, int max_n);
void rgb_arithmetic_mean_filter(const Mat &src, Mat &dst, int m, int n);
void rgb_geometric_mean_filter(const Mat &src, Mat &dst, int m, int n);

int main(int argc, char **argv) {
    Mat m1 = imread("images/lena.jpg", IMREAD_GRAYSCALE);
    imshow("lena", m1);

    // mean filter
    Mat m2, m2f;
    // pepper noise
    add_pesa_noise(m1, m2, 0.05, 0);
    imshow("pepper noise", m2);
    // arithmetic
    contraharmonic_mean_filter(m2, m2f, 5, 5, 0);
    imshow("pepper arithmetic", m2f);
    // contraharmonic, Q = 1.6
    contraharmonic_mean_filter(m2, m2f, 5, 5, 1.6);
    imshow("pepper contraharmonic", m2f);
    waitKey(0);
    destroyAllWindows();

    Mat m3, m3f;
    // salt noise
    add_pesa_noise(m1, m3, 0, 0.05);
    imshow("salt noise", m3);
    // arithmetic
    contraharmonic_mean_filter(m3, m3f, 5, 5, 0);
    imshow("salt arithmetic", m3f);
    // geometric
    geometric_mean_filter(m3, m3f, 5, 5);
    imshow("salt geometric", m3f);
    // harmonic
    contraharmonic_mean_filter(m3, m3f, 5, 5, -1);
    imshow("salt harmonic", m3f);
    // contraharmonic, Q = -1.6
    contraharmonic_mean_filter(m3, m3f, 5, 5, -1.6);
    imshow("salt contraharmonic", m3f);
    waitKey(0);
    destroyAllWindows();

    Mat m4, m4f;
    // salt-and-pepper noise
    add_pesa_noise(m1, m4, 0.025, 0.025);
    imshow("salt-pepper", m4);
    // arithmetic
    contraharmonic_mean_filter(m4, m4f, 5, 5, 0);
    imshow("salt-pepper arithmetic", m4f);
    waitKey(0);
    destroyAllWindows();

    Mat m5, m5f;
    // gaussian noise
    add_gaus_noise(m1, m5, 10);
    imshow("gauss noise", m5);
    // arithmetic
    contraharmonic_mean_filter(m5, m5f, 5, 5, 0);
    imshow("gauss arithmetic", m5f);
    // geometric
    geometric_mean_filter(m5, m5f, 5, 5);
    imshow("gauss geometric", m5f);
    // harmonic
    contraharmonic_mean_filter(m5, m5f, 5, 5, -1);
    imshow("gauss harmonic", m5f);
    waitKey(0);
    destroyAllWindows();


    // median filter
    Mat m6f;
    // pepper noise
    imshow("pepper noise", m2);
    median_filter(m2, m6f, 5, 5);
    imshow("pepper median 5*5", m6f);
    median_filter(m2, m6f, 9, 9);
    imshow("pepper median 9*9", m6f);
    waitKey(0);
    destroyAllWindows();

    // salt noise
    imshow("salt", m3);
    median_filter(m3, m6f, 5, 5);
    imshow("salt median 5*5", m6f);
    median_filter(m3, m6f, 9, 9);
    imshow("salt median 9*9", m6f);
    waitKey(0);
    destroyAllWindows();

    // salt-and-pepper noise
    imshow("salt-pepper", m4);
    median_filter(m4, m6f, 5, 5);
    imshow("salt-pepper median 5*5", m6f);
    median_filter(m4, m6f, 9, 9);
    imshow("salt-pepper median 9*9", m6f);
    waitKey(0);
    destroyAllWindows();


    // adaptive mean filter
    Mat m7, m7f;
    // gauss noise, sigam = 25
    add_gaus_noise(m1, m7, 25);
    imshow("gauss noise, sigma=25", m7);
    // arithmetic mean filter
    contraharmonic_mean_filter(m7, m7f, 7, 7, 0);
    imshow("arithmetic 7*7", m7f);
    adaptive_mean_filter(m7, m7f, 7, 7);
    imshow("adaptive mean 7*7", m7f);
    waitKey(0);
    destroyAllWindows();


    // adaptive median filter
    Mat m8, m8f;
    // salt-and-pepper noise
    add_pesa_noise(m1, m8, 0.25, 0.25);
    imshow("salt-pepper adaptive", m8);
    median_filter(m8, m8f, 7, 7);
    imshow("median 7*7", m8f);
    adaptive_median_filter(m8, m8f, 3, 3, 7, 7);
    imshow("adaptive median 7*7", m8f);
    waitKey(0);
    destroyAllWindows();


    // rgb gaussian noise
    Mat m9 = imread("images/lena.jpg");
    imshow("lena rgb", m9);
    Mat m10, m10f;
    add_gaus_noise(m9, m10, 10);
    imshow("gauss noise rgb", m10);
    rgb_arithmetic_mean_filter(m10, m10f, 5, 5);
    imshow("gauss rgb arithmetic", m10f);
    rgb_geometric_mean_filter(m10, m10f, 5, 5);
    imshow("gauss rgb geometric", m10f);
    waitKey(0);
    destroyAllWindows();

    // rgb salt noise
    add_pesa_noise(m9, m10, 0, 0.05);
    imshow("salt noise rgb", m10);
    rgb_arithmetic_mean_filter(m10, m10f, 5, 5);
    imshow("salt rgb arithmetic", m10f);
    rgb_geometric_mean_filter(m10, m10f, 5, 5);
    imshow("salt rgb geometric", m10f);
    waitKey(0);

    return 0;
}

void add_gaus_noise(const Mat &src, Mat &dst, double sigma) {
    CV_Assert(src.type() == CV_8UC1 || src.type() == CV_8UC3);
    const int rows = src.rows;
    const int cols = src.cols;
    default_random_engine generator;
    normal_distribution<double> noise {0, 1};
    double gaussian_noise;
    dst.create(src.size(), src.type());
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            gaussian_noise = noise(generator);
            if (src.channels() == 1) {
                dst.at<uchar>(i, j) = saturate_cast<uchar>(src.at<uchar>(i, j) + 
                                      static_cast<int>(gaussian_noise * sigma));
            } else {
                for (int ch = 0; ch < 3; ++ch)
                    dst.at<Vec3b>(i, j)[ch] = 
                    saturate_cast<uchar>(src.at<Vec3b>(i, j)[ch] + 
                    static_cast<int>(gaussian_noise * sigma));
            }
        }
    }
}

void add_pesa_noise(const Mat &src, Mat &dst, double p_pepr, double p_salt) {
    CV_Assert((src.type() == CV_8UC1 || src.type() == CV_8UC3) && 
              p_pepr >= 0 && p_salt >= 0 && (p_pepr + p_salt) <= 1);
    dst = src.clone();
    const int rows = src.rows;
    const int cols = src.cols;
    int i = 0;
    int j = 0;
    default_random_engine generator;
    uniform_int_distribution<int> random_row(0, rows - 1);
    uniform_int_distribution<int> random_col(0, cols - 1);
    int pepr_polluted = static_cast<int>(p_pepr * rows * cols);
    int salt_polluted = static_cast<int>(p_salt * rows * cols);
    int kp = 0, ks = 0;
    // add noise
    while (kp++ < pepr_polluted) {
        i = random_row(generator);
        j = random_col(generator);
        if (src.channels() == 1) {
            dst.at<uchar>(i, j) = 0;
        } else {
            dst.at<Vec3b>(i, j)[0] = 0;
            dst.at<Vec3b>(i, j)[1] = 0;
            dst.at<Vec3b>(i, j)[2] = 0;
        }
    }
    while (ks++ < salt_polluted) {
        i = random_row(generator);
        j = random_col(generator);
        if (src.channels() == 1) {
            dst.at<uchar>(i, j) = 255;
        } else {
            dst.at<Vec3b>(i, j)[0] = 255;
            dst.at<Vec3b>(i, j)[1] = 255;
            dst.at<Vec3b>(i, j)[2] = 255;
        }
    }
}

void geometric_mean_filter(const Mat &src, Mat &dst, int m, int n) {
    Mat filled;
    dst.create(src.size(), src.type());
    int row_border = (m - 1) / 2;
    int col_border = (n - 1) / 2;
    copyMakeBorder(src, filled, row_border, row_border, 
                                col_border, col_border, BORDER_REFLECT);
    int i_dst = 0;
    int j_dst = 0;
    double pd = 1;
    const double exponent = 1.0 / m / n;
    for (int i = row_border; i < filled.rows - row_border; ++i) {
        for (int j = col_border; j < filled.cols - col_border; ++j) {
            i_dst = i - row_border;
            j_dst = j - col_border;
            pd = 1;
            for (int x = i - row_border; x <= i + row_border; ++x)
                for (int y = j - col_border; y <= j + col_border; ++y)
                    pd *= filled.at<uchar>(x, y);
            dst.at<uchar>(i_dst, j_dst) = saturate_cast<uchar>(pow(pd, exponent));
        }
    }
}

/*
 * Q > 0, eliminates pepper noise
 * Q < 0, eliminates salt noise
 * Q = 0 --> arithmetic mean filter, eliminates noise
 * Q =-1 -->  harmonic_mean_filter,  eliminates gauss and salt noise
 */
void contraharmonic_mean_filter(const Mat &src, Mat &dst, int m, int n, double Q) {
    Mat filled;
    dst.create(src.size(), src.type());
    int row_border = (m - 1) / 2;
    int col_border = (n - 1) / 2;
    copyMakeBorder(src, filled, row_border, row_border, 
                                col_border, col_border, BORDER_REFLECT);
    int i_dst = 0;
    int j_dst = 0;
    double s1 = 0;
    double s2 = 0;
    for (int i = row_border; i < filled.rows - row_border; ++i) {
        for (int j = col_border; j < filled.cols - col_border; ++j) {
            s1 = s2 = 0;
            i_dst = i - row_border;
            j_dst = j - col_border;
            for (int x = i - row_border; x <= i + row_border; ++x) {
                for (int y = j - col_border; y <= j + col_border; ++y) {
                    s1 += pow(filled.at<uchar>(x, y), Q + 1);
                    s2 += pow(filled.at<uchar>(x, y), Q);
                }
            }
            dst.at<uchar>(i_dst, j_dst) = saturate_cast<uchar>(s1 / s2);
        }
    }
}

void median_filter(const Mat &src, Mat &dst, int m, int n) {
    Mat filled;
    dst.create(src.size(), src.type());
    int row_border = (m - 1) / 2;
    int col_border = (n - 1) / 2;
    copyMakeBorder(src, filled, row_border, row_border, 
                                col_border, col_border, BORDER_REFLECT);
    int i_dst = 0;
    int j_dst = 0;
    vector<uchar> Sxy;
    int median_index = m * n / 2;
    for (int i = row_border; i < filled.rows - row_border; ++i) {
        for (int j = col_border; j < filled.cols - col_border; ++j) {
            i_dst = i - row_border;
            j_dst = j - col_border;
            Sxy.clear();
            for (int x = i - row_border; x <= i + row_border; ++x)
                for (int y = j - col_border; y <= j + col_border; ++y)
                    Sxy.push_back(filled.at<uchar>(x, y));
            sort(Sxy.begin(), Sxy.end());
            dst.at<uchar>(i_dst, j_dst) = Sxy[median_index];
        }
    }
}

void adaptive_mean_filter(const Mat &src, Mat &dst, int m, int n) {
    // int rows = src.rows;
    // int cols = src.cols;
    // // 计算噪声图像的标准差
    // double sum      = 0;
    // double mean_src = 0;
    // double sdev_src = 0;
    // for (int i = 0; i < rows; ++i)
    //     for (int j = 0; j < cols; ++j)
    //         sum += src.at<uchar>(i, j);
    // mean_src = sum / (rows * cols);
    // sum = 0;
    // for (int i = 0; i < rows; ++i)
    //     for (int j = 0; j < cols; ++j)
    //         sum += pow(1.0 * src.at<uchar>(i, j) - mean_src, 2);
    // sdev_src = sqrt(sum / (rows * cols));
    double sdev_src = 25;

    Mat filled;
    dst.create(src.size(), src.type());
    int row_border = (m - 1) / 2;
    int col_border = (n - 1) / 2;
    copyMakeBorder(src, filled, row_border, row_border, 
                                col_border, col_border, BORDER_REFLECT);
    int i_dst = 0;
    int j_dst = 0;
    Mat Sxy;
    Mat mean_Sxy;
    Mat sdev_Sxy;
    double k = 0;
    for (int i = row_border; i < filled.rows - row_border; ++i) {
        for (int j = col_border; j < filled.cols - col_border; ++j) {
            i_dst = i - row_border;
            j_dst = j - col_border;
            Sxy = Mat(filled, Rect(j - col_border, i - row_border, n, m));
            meanStdDev(Sxy, mean_Sxy, sdev_Sxy);
            k = (pow(sdev_src, 2)) / (pow(sdev_Sxy.at<double>(0, 0), 2) + 1e-6);
            if (k < 1)
                dst.at<uchar>(i_dst, j_dst) = 
                saturate_cast<uchar>(filled.at<uchar>(i, j) - 
                k * (filled.at<uchar>(i, j) - mean_Sxy.at<double>(0, 0)));
            else
                dst.at<uchar>(i_dst, j_dst) = mean_Sxy.at<double>(0, 0);
        }
    }
}

/// @brief the implementation of an adaptive median filter,
///        the in-place processing is not supported.
/// @param src input image
/// @param dst output image
/// @param init_m the rows of the initial neighborhood S_xy
/// @param init_n the cols of the initial neighborhood S_xy
/// @param max_m the rows of the maximal neighborhood S_max
/// @param max_n the cols of the maximal neighborhood S_max
void adaptive_median_filter(const Mat &src, Mat &dst, int init_m, int init_n, int max_m, int max_n) {
    CV_Assert(src.type() == CV_8UC1 && init_m <= max_m && init_n <= max_n);
    dst.create(src.size(), CV_8UC1);
    int max_row_border = (max_m - 1) / 2;
    int max_col_border = (max_n - 1) / 2;
    Mat filled;
    // 按照S_xy的最大允许尺寸S_max进行边界填充，filled为填充结果，填充方式为镜像填充
    copyMakeBorder(src, filled, max_row_border, max_row_border,
                                max_col_border, max_col_border, BORDER_REFLECT);
    int x, y;
    int r_bias, c_bias;
    int curr_m = init_m;
    int curr_n = init_n;
    vector<uchar> S_xy;
    uchar z_xy, z_min, z_med, z_max;
    bool is_Sxy_greater_than_Smax = false;

    for (int i = 0; i < dst.rows; ++i) {
        for (int j = 0; j < dst.cols; ++j) {
            // 计算z_xy
            x = i + max_row_border; // dst图像中的像素在filled中所处的行
            y = j + max_col_border; // dst图像中的像素在filled中所处的列
            z_xy = filled.at<uchar>(x, y);

            // 计算z_min、z_med和z_max
            S_xy.clear();
            r_bias = (curr_m - 1) / 2;
            c_bias = (curr_n - 1) / 2;
            for (int si = x - r_bias; si <= x + r_bias; ++si)
                for (int sj = y - c_bias; sj <= y + c_bias; ++sj)
                    S_xy.push_back(filled.at<uchar>(si, sj));
            sort(S_xy.begin(), S_xy.end());
            z_min = S_xy[0];
            z_med = S_xy[curr_m * curr_n / 2];
            z_max = S_xy[curr_m * curr_n - 1];

            // 当不满足z_min < z_med < z_max时，扩大S_xy
            is_Sxy_greater_than_Smax = false;
            while (!(z_min < z_med && z_med < z_max)) {
                curr_m += 2;
                curr_n += 2;
                // 如果S_xy > S_max，转移到Sxy_greater_than_Smax语句
                if (curr_m > max_m || curr_n > max_n) {
                    is_Sxy_greater_than_Smax = true;
                    goto Sxy_greater_than_Smax;
                }
                // 扩大S_xy，即将当前S_xy中外边一圈的像素包括进来
                ++r_bias;
                ++c_bias;
                for (int c = y - c_bias; c <= y + c_bias; ++c)
                    S_xy.push_back(filled.at<uchar>(x - r_bias, c)); //  top row
                for (int c = y - c_bias; c <= y + c_bias; ++c)
                    S_xy.push_back(filled.at<uchar>(x + r_bias, c)); // bottom row
                for (int r = x - r_bias + 1; r <= x + r_bias - 1; ++r)
                    S_xy.push_back(filled.at<uchar>(r, y - c_bias)); // left column
                for (int r = x - r_bias + 1; r <= x + r_bias - 1; ++r)
                    S_xy.push_back(filled.at<uchar>(r, y + c_bias)); // right column

                // 计算新的z_min、z_med和z_max
                sort(S_xy.begin(), S_xy.end());
                z_min = S_xy[0];
                z_med = S_xy[curr_m * curr_n / 2];
                z_max = S_xy[curr_m * curr_n - 1];
            }

            // S_xy > S_max时输出z_med
            Sxy_greater_than_Smax:
            dst.at<uchar>(i, j) = z_med;
            // 满足z_min < z_med < z_max且S_xy <= S_max时输出z_xy或z_med
            if (!is_Sxy_greater_than_Smax)
                dst.at<uchar>(i, j) = (z_min < z_xy && z_xy < z_max) ? z_xy : z_med;
        }
    }
}

void rgb_arithmetic_mean_filter(const Mat &src, Mat &dst, int m, int n) {
    Mat filled;
    dst.create(src.size(), src.type());
    int row_border = (m - 1) / 2;
    int col_border = (n - 1) / 2;
    copyMakeBorder(src, filled, row_border, row_border, 
                                col_border, col_border, BORDER_REFLECT);
    int i_dst = 0;
    int j_dst = 0;
    vector<double> sum(3, 0);
    for (int i = row_border; i < filled.rows - row_border; ++i) {
        for (int j = col_border; j < filled.cols - col_border; ++j) {
            i_dst = i - row_border;
            j_dst = j - col_border;
            sum[0] = sum[1] = sum[2] = 0;
            for (int x = i - row_border; x <= i + row_border; ++x)
                for (int y = j - col_border; y <= j + col_border; ++y)
                    for (int ch = 0; ch < 3; ++ch)
                        sum[ch] += filled.at<Vec3b>(x, y)[ch];
            for (int ch = 0; ch < 3; ++ch)
                dst.at<Vec3b>(i_dst, j_dst)[ch] = saturate_cast<uchar>(sum[ch] / (m * n));
        }
    }
}

void rgb_geometric_mean_filter(const Mat &src, Mat &dst, int m, int n) {
    Mat filled;
    dst.create(src.size(), src.type());
    int row_border = (m - 1) / 2;
    int col_border = (n - 1) / 2;
    copyMakeBorder(src, filled, row_border, row_border, 
                                col_border, col_border, BORDER_REFLECT);
    int i_dst = 0;
    int j_dst = 0;
    vector<double> pd(3, 1);
    const double exponent = 1.0 / m / n;
    for (int i = row_border; i < filled.rows - row_border; ++i) {
        for (int j = col_border; j < filled.cols - col_border; ++j) {
            i_dst = i - row_border;
            j_dst = j - col_border;
            pd[0] = pd[1] = pd[2] = 1;
            for (int x = i - row_border; x <= i + row_border; ++x)
                for (int y = j - col_border; y <= j + col_border; ++y)
                    for (int ch = 0; ch < 3; ++ch)
                        pd[ch] *= filled.at<Vec3b>(x, y)[ch];
            for (int ch = 0; ch < 3; ++ch)
                dst.at<Vec3b>(i_dst, j_dst)[ch] = saturate_cast<uchar>(pow(pd[ch], exponent));
        }
    }
}
