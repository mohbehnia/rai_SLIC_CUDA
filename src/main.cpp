#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "SlicCudaHost.h"

using namespace std;
using namespace cv;

int main() {
 
	// VideoCapture cap("/home/jimmy/ece508/data/waterfall.avi");


	// Parameters
	int wc = 35;
	int nIteration = 5;
	int num_segments = 10000;
	SlicCuda::InitType initType = SlicCuda::SLIC_SIZE;

	//start segmentation

	auto t30 = std::chrono::high_resolution_clock::now();

	Mat frame;
	frame = imread("/src/coco.jpeg", IMREAD_COLOR);
	int diamSpx = sqrt(frame.rows*frame.cols/10000); //want about 10
	SlicCuda oSlicCuda;
	oSlicCuda.initialize(frame, diamSpx, initType, wc, nIteration);
	Mat labels;
	auto t00 = std::chrono::high_resolution_clock::now();
	oSlicCuda.segment(frame);
	auto t01 = std::chrono::high_resolution_clock::now();
	double time0 = std::chrono::duration<double>(t01-t00).count() ;
	cout << "Frame " << frame.size() << "Segment Time: "<< time0 <<"s"<<endl;

	oSlicCuda.enforceConnectivity();

	labels = oSlicCuda.getLabels();
	auto data = labels.data;

	// SlicCuda::displayBound(frame, (float*)labels.data, Scalar(0, 0, 0));
	auto t20 = std::chrono::high_resolution_clock::now();
	SlicCuda::displayPoint1(frame, (float*)labels.data, Scalar(0, 0, 0));
	auto t21 = std::chrono::high_resolution_clock::now();
	double time2 = std::chrono::duration<double>(t21-t20).count() ;

	cout << "Triangle Time: "<< time2 <<"s"<<endl;

	bool success = imwrite("/build/out.jpg", frame);

	cout << "WRITE " << success << endl;

	auto t31 = std::chrono::high_resolution_clock::now();
	double time3 = std::chrono::duration<double>(t31-t30).count() ;
	cout << "Total Time: "<< time3 << "s"<< endl;

    return 0;
}
