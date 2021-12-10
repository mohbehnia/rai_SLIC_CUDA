#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include "SlicCudaHost.h"

using namespace std;
using namespace cv;

int main() {
 
	// VideoCapture cap("/home/jimmy/ece508/data/waterfall.avi");
	cout << "FIRST0" << endl;

	// Parameters
	int wc = 35;
	int nIteration = 5;
	int num_segments = 10000;
	SlicCuda::InitType initType = SlicCuda::SLIC_SIZE;

	//start segmentation
	Mat frame;
	frame = imread("/src/temple.jpg", IMREAD_COLOR);
	int diamSpx = sqrt(frame.rows*frame.cols/10000); //want about 10
	SlicCuda oSlicCuda;
	oSlicCuda.initialize(frame, diamSpx, initType, wc, nIteration);
	Mat labels;
	auto t0 = std::chrono::high_resolution_clock::now();
	oSlicCuda.segment(frame);
	auto t1 = std::chrono::high_resolution_clock::now();
	double time = std::chrono::duration<double>(t1-t0).count() ;
	cout << "Frame " << frame.size() << "Segment Time: "<< time <<"s"<<endl;

	oSlicCuda.enforceConnectivity();
	cout << "FIRST6" << endl;

	labels = oSlicCuda.getLabels();
	auto data = labels.data;
	cout << "FIRST7" << endl;
	// SlicCuda::displayBound(frame, (float*)labels.data, Scalar(0, 0, 0));
	SlicCuda::displayPoint1(frame, (float*)labels.data, Scalar(0, 0, 0));
	cout << "FIRST8" << endl;

	bool success = imwrite("/build/out.jpg", frame);

	cout << "WRITE " << success << endl;

    return 0;
}
