rm -r build
mkdir build
cd build
cmake ..
make
mv bin/main ..
mv bin/features_compute ..
cd ..
