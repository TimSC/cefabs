#ifndef _IMG_PROC_H
#define _IMG_PROC_H

#include <opencv2/opencv.hpp>

void cef_opencv_sobel(cv::InputArray mimg, cv::Mat st, float threshold2, cv::OutputArray out);

// https://stackoverflow.com/a/13301755/4288232
template<typename T, int interpolate, int D> T getColorSubpix(const cv::Mat& img, cv::Point2f pt)
{
	if(img.empty())
		throw std::invalid_argument("Image is empty");
	T out;
	if(img.channels() != out.channels)
		throw std::invalid_argument("Image has wrong number of channels");
	if(D < CV_8U or D > CV_64F)
		throw std::invalid_argument("Colour depth could be either CV_8U, CV_8S, CV_16U, CV_16S, CV_32S, CV_32F or CV_64F");
	if(img.type() % 8 != D)
		throw std::invalid_argument("Input has unexpected colour depth");

	int x=0, y=0;
	if(interpolate == 0) //Nearest pixel
	{
		x = (int)(pt.x+0.5);
		y = (int)(pt.y+0.5);
	}
	else if(interpolate == 1) //Bilinear
	{
		x = (int)pt.x;
		y = (int)pt.y;
	}

	int x0 = cv::borderInterpolate(x,   img.cols, cv::BORDER_REFLECT_101);
	int y0 = cv::borderInterpolate(y,   img.rows, cv::BORDER_REFLECT_101);

	if(interpolate == 0)
	{
		for(size_t i=0; i<out.channels; i++)
			out[i] = img.at<T>(y0, x0)[i];
	}
	else if(interpolate == 1)
	{
		int x1 = cv::borderInterpolate(x+1, img.cols, cv::BORDER_REFLECT_101);
		int y1 = cv::borderInterpolate(y+1, img.rows, cv::BORDER_REFLECT_101);

		float a = pt.x - (float)x;
		float c = pt.y - (float)y;

		for(size_t i=0; i<out.channels; i++)
		{
			out[i] = cvRound((img.at<T>(y0, x0)[i] * (1.f - a) + img.at<T>(y0, x1)[i] * a) * (1.f - c)
						   + (img.at<T>(y1, x0)[i] * (1.f - a) + img.at<T>(y1, x1)[i] * a) * c);
		}
	}
	return out;
}

template<typename T> 
using Sampler = T(*)(const cv::Mat&, cv::Point2f);

template<typename T>
inline float st2A(T g) 
{
    float a = 0.5f * (g[1] + g[0]); 
    float b = 0.5f * sqrtf(fmaxf(0.0f, g[1]*g[1] - 2*g[0]*g[1] + g[0]*g[0] + 4*g[2]*g[2]));
    float lambda1 = a + b;
    float lambda2 = a - b;
    return (lambda1 + lambda2 > 0)?
        (lambda1 - lambda2) / (lambda1 + lambda2) : 0;
}

template<typename SrcT, typename StT, int order> 
void cef_opencv_stgauss2_filter_imp(int row, int col, const cv::Mat& src, const cv::Mat& st,
	cv::Mat& dst, Sampler<SrcT> srcSamp, Sampler<StT> stSamp, float sigma, float cos_max, 
    bool adaptive, float step_size ) 
{
    if (adaptive) {
        float A = st2A(stSamp(st, cv::Point2f(row, col)));
        sigma *= 0.25f * (1.0f + A)*(1.0f + A);
    }

	//cv::Mat src2;
	//cv::GaussianBlur(src, src2, cv::Size(0, 0), sigma);	
	
	/*stgauss2_filter<T,SRC> f(src, sigma);
    if (order == 1) st_integrate_euler(p0, st, f, cos_max, dst.w, dst.h, step_size);
    if (order == 2) st_integrate_rk2(p0, st, f, cos_max, dst.w, dst.h, step_size);
    if (order == 4) st_integrate_rk4(p0, st, f, cos_max, dst.w, dst.h, step_size);
    dst(ix, iy) = f.c_ / f.w_;*/
}

template<typename SrcT, int srcInterpolate, int srcD, typename StT, int stInterpolate, int stD, int order> 
void cef_opencv_stgauss2_filter( cv::InputArray src, cv::InputArray st, 
									  float sigma, float max_angle, bool adaptive,
									  float step_size,
									  cv::OutputArray out )
{
	if (sigma <= 0.0f)
	{
		src.copyTo(out);
		return;
	}
	cv::Mat dst;
	src.copyTo(dst);

	Sampler<SrcT> srcSamp = getColorSubpix<SrcT, srcInterpolate, srcD>;
	float cos_max = cos(max_angle*M_PI/180.0);

	cv::Mat srcMat = src.getMat();
	cv::Mat stMat = st.getMat();
	for(size_t r=0; r<srcMat.rows; r++)
		for(size_t c=0; c<srcMat.cols; c++)
		{
			if (src.size() == st.size()) {
				Sampler<StT> stSamp = getColorSubpix<StT, stInterpolate, stD>;
				cef_opencv_stgauss2_filter_imp<SrcT, StT, order>(r, c, srcMat, stMat, dst, srcSamp, stSamp, sigma, cos_max, adaptive, step_size);
			} else {
			/*	float2 s = make_float2((float)st.w() / src.w(), (float)st.h() / src.h());
				gpu_resampler_ST<float4> st_sampler(st, s, st_linear? cudaFilterModeLinear : cudaFilterModePoint);
				if (order == 1) imp_stgauss2_filter<1,float><<<dst.blocks(), dst.threads()>>>(dst, src_sampler, st_sampler, sigma, cos_max, adaptive, step_size);
				else if (order == 2) imp_stgauss2_filter<2,float><<<dst.blocks(), dst.threads()>>>(dst, src_sampler, st_sampler, sigma, cos_max, adaptive, step_size);
				else if (order == 4) imp_stgauss2_filter<4,float><<<dst.blocks(), dst.threads()>>>(dst, src_sampler, st_sampler, sigma, cos_max, adaptive, 					step_size);*/
			}
		}
	src.copyTo(out);
}


#endif //_IMG_PROC_H

