# matrix
Sample class Matrix with basics functions and fast multiplication using __GPU, AVX2 instructions or openmp__.

_Methods of class :_

* __inverse(using gauss)__

* __decomposition PLU__

* __decomposition QR__ using gram schmidt or householder algorithm

* __Fast Fourier Transform__

* __Kernel of the matrix__

* __inverse modular__

* __Eigen values__ using QR algorithm and __Eigen vectors__



Example of basics functions :

```C++
#include <iostream>
#include "matrix.hpp"

int main() {
  Matrix<int> ma(std::vector<std::vector<int> > { {1, 2, 3, 4}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}} );
  Matrix<int> mb(std::vector<std::vector<int> > { {1, 2, 3, 4}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15} } );
  
  ma.display();
  mb.display();
  
//basic matrix product
  auto mc = ma.dot(mb);
  mc.display();
  
//faster matrix product
  auto md = ma.strassen(mb);
  md.display();

//if you have NVDIA GPU
  auto m = ma.dotGPU(mb);
  m.display();
  
  return 0;
}

```` 
You have to compile with: -fopenmp -mavx2 -O3  to obtain maximum performance.

If you want to use your GPU compile with nvcc and compile the kernel.cu file
