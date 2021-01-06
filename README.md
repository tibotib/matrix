# matrix
Simple class Matrix with basics functions and fast multiplication.

It can use GPU or AVX2 instructions.

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

//or if you have NVDIA GPU
  auto m = ma.dotGPU(mb);
  m.display();
  
  
  return 0;
}

```` 
You have to compile with: -fopenmp -mavx2 -O3  to obtain maximum performance.

If you want to use your GPU compile with nvcc and compile the kernel.cu file
