#include <vector>
#include "imgproc.h"
using namespace cv;
using namespace std;

void cef_opencv_sobel(cv::InputArray mimg, cv::Mat stm, float threshold2, cv::OutputArray out)
{
	if(mimg.channels() != 4)
		throw invalid_argument("Expected 4 channels in cef_opencv_sobel input");
	if(!stm.empty() && (stm.channels() != 4))
		throw invalid_argument("Expected 4 channels in cef_opencv_sobel input");
	UMat mimg2 = mimg.getUMat();
	UMat st2h, st2v, st2;
	Mat hk = Mat(3, 3, CV_32F, Scalar(0.0f));
	hk.at<float>(0,0) = -0.183f;
	hk.at<float>(1,0) = -0.634f;
	hk.at<float>(2,0) = -0.183f;
	hk.at<float>(0,2) = 0.183f;
	hk.at<float>(1,2) = 0.634f;
	hk.at<float>(2,2) = 0.183f;
	hk *= 0.5f;
	
	filter2D(mimg2, st2h, CV_32F, hk);
	filter2D(mimg2, st2v, CV_32F, hk.t());
	
	vector<UMat> st2hch, st2vch;
	split (st2h, st2hch);
	split (st2v, st2vch);

	UMat add1, uum, vvm, uvm, uvm2sc, mag;
	add(st2hch[0].mul(st2hch[0]), st2hch[1].mul(st2hch[1]), add1);
	add(add1, st2hch[2].mul(st2hch[2]), uum);

	add(st2vch[0].mul(st2vch[0]), st2vch[1].mul(st2vch[1]), add1);
	add(add1, st2vch[2].mul(st2vch[2]), vvm);

	add(st2hch[0].mul(st2vch[0]), st2hch[1].mul(st2vch[1]), add1);
	add(add1, st2hch[2].mul(st2vch[2]), uvm);

	add(uum.mul(uum), vvm.mul(vvm), add1);
	uvm.mul(uvm).convertTo(uvm2sc, CV_32F, 2.0);
	add(add1, uvm2sc, mag);

	Mat magm = mag.getMat(ACCESS_READ);
	Mat uumm = uum.getMat(ACCESS_READ);
	Mat vvmm = vvm.getMat(ACCESS_READ);
	Mat uvmm = uvm.getMat(ACCESS_READ);
	Mat outTmp;
	mimg.copyTo(outTmp);
	
	for(int r=0; r<magm.rows; r++)
		for(int c=0; c<magm.cols; c++)
		{
			float magval = magm.at<float>(r, c);
			Vec4f &outvals = outTmp.at<Vec4f>(r, c);
			if(magval < threshold2)
			{
				if(!stm.empty())
					outvals = stm.at<Vec4f>(r, c);
				else
				{
					outvals[0] = uumm.at<float>(r, c);
					outvals[1] = vvmm.at<float>(r, c);
					outvals[2] = uvmm.at<float>(r, c);
					outvals[3] = 0.0f;
				}
			}
			else
			{
				outvals[0] = uumm.at<float>(r, c);
				outvals[1] = vvmm.at<float>(r, c);
				outvals[2] = uvmm.at<float>(r, c);
				outvals[3] = magval;
			}
		}

	outTmp.copyTo(out);
}

