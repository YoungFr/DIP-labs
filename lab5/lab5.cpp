#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>
#include <iostream>

using namespace cv;
using std::vector;
using std::cout;
using std::endl;

void frequency_domain_filter(Mat& f, const int type, const int D0, const int n);
Mat transfer_func(const int P, const int Q, int type, const int D0, const int n);

int main() {
    const char * filename = "images/lena.jpg";
    Mat I = imread(samples::findFile(filename), IMREAD_GRAYSCALE);
    if(I.empty()){
        cout << "Error opening image" << endl;
        return 1;
    }
	imshow("Lena", I);
    frequency_domain_filter(I, 0, 50,-1);
	waitKey(0);
	imshow("Lena", I);
    frequency_domain_filter(I, 0, 50, 2);
	waitKey(0);
	imshow("Lena", I);
    frequency_domain_filter(I, 1, 50,-1);
	waitKey(0);
	imshow("Lena", I);
    frequency_domain_filter(I, 1, 50, 2);
    waitKey(0);
    return 0;
}

/**
 * @brief 
 * 使用《数字图像处理(第四版)》第182-183页所描述的算法实现的频率域滤波函数
 * @param    f 输入图像
 * @param type 0表示低通滤波器，非0表示高通滤波器
 * @param   D0 截止频率
 * @param    n 布特沃斯滤波器的阶数，如果要使用理想高通/低通滤波器，请将该值设为-1
 */
void frequency_domain_filter(Mat& f, const int type, const int D0, const int n) {
    const int M = f.rows;
    const int N = f.cols;
	// 1. 计算填充后的图像的尺寸
    const int P = 2 * M;
    const int Q = 2 * N;
	// 2. 使用镜像填充形成大小为P * Q的填充后的图像fp(x,y)
    Mat fp;
    copyMakeBorder(f, fp, 0, M, 0, N, BORDER_REFLECT);
	// 3. 将fp(x,y)乘以(-1)^(x+y)，使傅里叶变换位于P * Q大小的频率矩形的中心
    for (int i = 0; i < P; i++)
        for (int j = 0; j < Q; j++)
            if ((i + j) & 1)
                fp.at<uchar>(i, j) = -fp.at<uchar>(i, j);
	// 4. 计算步骤3得到的图像的DFT
    Mat real_imag[] = {Mat_<float>(fp), Mat::zeros(fp.size(), CV_32F)};
	Mat F;
	merge(real_imag, 2, F);
	dft(F, F);
	split(F, real_imag);
	// 5. 构建滤波器传递函数
	Mat H = transfer_func(P, Q, type, D0, n);
	// 6. 采用对应像素相乘得到G
	for (int i = 0; i < P; i++) {
        for (int j = 0; j < Q; j++) {
			real_imag[0].at<float>(i, j) *= H.at<float>(i, j);
			real_imag[1].at<float>(i, j) *= H.at<float>(i, j);
		}
	}
	// 7. 计算G的IDFT
	Mat G;
	merge(real_imag, 2, G);
	idft(G, G);
	split(G, real_imag);
	// 8. 提取G的左上角部分的实部，去除寄生复数项
	Mat g(Size(M, N), CV_32F);
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
			g.at<float>(i, j) = real_imag[0].at<float>(i, j) * pow(-1, i+j);
	// 9. 归一化处理
	normalize(g, g, 0, 1, NORM_MINMAX);
	imshow("g", g);
}

// type == 0 -> lowpass  -> ILPF if n == -1 else BLPF(n > 0)
// type == 1 -> highpass -> IHPF if n == -1 else BHPF(n > 0)
Mat transfer_func(const int P, const int Q, int type, const int D0, const int n) {
	const int M = P / 2;
	const int N = Q / 2;
	const double D0_2 = D0 * D0;
	Mat H(Size(P, Q), CV_32F);
	if (type == 0) { // LP
		for (int i = 0; i < P; i++)
        	for (int j = 0; j < Q; j++)
				H.at<float>(i, j) = 
				(n == -1) ?
				(((i-M)*(i-M) + (j-N)*(j-N) <= D0_2) ? 1 : 0) :       // ILPF
				1.0 / (1 + pow(((i-M)*(i-M)+(j-N)*(j-N))/D0_2, 2*n)); // BLPF
	} else {         // HP
		for (int i = 0; i < P; i++)
        	for (int j = 0; j < Q; j++) {
				H.at<float>(i, j) = 
				(n == -1) ?
				(((i-M)*(i-M) + (j-N)*(j-N) <= D0_2) ? 0 : 1) :       // IHPF
				1.0 / (1 + pow(D0_2/((i-M)*(i-M)+(j-N)*(j-N)), 2*n)); // BHPF
			}
	}
	return H;
}