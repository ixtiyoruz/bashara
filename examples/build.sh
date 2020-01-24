mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="$HOME/freenect2/lib/cmake/freenect2;/home/ixtiyor/Downloads/caffe/" -DBOOST_ROOT=/home/ixtiyor/boost ..
make -j12

