# SLIC_CUDA

### How to run:
Ensure "rai" binary is in parent directory (same directory as rai_SLIC_CUDA). (Or modify run script)
./run.sh
Observe output and build folder for results.
./del.sh to delete previous run and get ready for the next


Superpixel Segmentation SLIC of Achanta et al. [PAMI 2012, vol. 34, num. 11, pp. 2274-2282]. GPU version with CUDA.

### Library required :
* opencv 3.0 min
* cuda compute capability 3.0 min

### Example
  Cuda : 30 fps
  
<p align="center">
  <img src="https://cloud.githubusercontent.com/assets/10605043/17391166/fceb718c-59e1-11e6-8251-7ca9b90287e5.gif" width="700"/>
</p>

 Comparison with cpu version from OpenCv : 4 fps
  <p align="center">
  <img src="https://cloud.githubusercontent.com/assets/10605043/17391176/247636ec-59e2-11e6-9d3d-df4e16c75218.gif" width="700"/>
  </p>
