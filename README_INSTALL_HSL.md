1. Get the coinhsl files at https://licences.stfc.ac.uk/product/coin-hsl (academic users get these solvers for free)
2. git clone the repo https://github.com/coin-or-tools/ThirdParty-HSL
3. unzip through `tar xf coinhsl-x.y.z.tar`
4. rename the extracted coinhsl-x.y.z to coinhsl 
5. move `coinhsl` into the ThirdParty-HSL repo 
6. Configure for a loadable shared library. In https://github.com/coin-or-tools/ThirdParty-HSL
$PREFIX generally points to a /usr/local
`./configure --prefix="$PREFIX"`
7. Build and install
make
make install
8. Set env variable that points to library. Add the following to your bashrc
export LIBHSL_DIR=$PREFIX
