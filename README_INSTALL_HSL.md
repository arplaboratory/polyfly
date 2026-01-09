git clone the repo https://github.com/coin-or-tools/ThirdParty-HSL

get the hsl libraries 
gunzip coinhsl-x.y.z.tar.gz
tar xf coinhsl-x.y.z.tar
rename the extracted coinhsl-x.y.z to coinhsl 
move coinhsl/ into the ThirdParty repo 


# 2. Configure for a loadable shared library. In https://github.com/coin-or-tools/ThirdParty-HSL
./configure --prefix=/usr/local LIBS="-llapack" --with-blas="-L/usr/lib -lblas" CXXFLAGS="-g -O2 -fopenmp" FCFLAGS="-g -O2 -fopenmp" CFLAGS="-g -O2 -fopenmp" --enable-shared --enable-static --enable-loadable-library

# 3. Build and install
make
sudo make install

cd /usr/local/lib
sudo ln -s libcoinhsl.so libhsl.so
sudo ldconfig

add /usr/local/lib to LD_LIBRARY_PATH