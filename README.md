# matrix
Simple class Matrix with basics functions and fast multiplication using GPU, AVX2 instructions or openmp.

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
  
//QR decomposition
  auto qr = ma.decompositionQR();
  
  std::cout << "Q : " << std::endl;
  qr.first.display();
  
  std::cout << "R : " << std::endl;
  qr.second.display();
  std::cout << std::endl;
 
//PLU decomposition
  auto plu = ma.decompositionPLU();
  
  std::cout << "P : " << std::endl;
  std::get<0>(plu).display();
  
  std::cout << "L : " << std::endl;
  std::get<1>(plu).display();
  
  std::cout << "U : " << std::endl;
  std::get<2>(plu).display();
  
  
  return 0;
}

```` 
You have to compile with: -fopenmp -mavx2 -O3  to obtain maximum performance.

If you want to use your GPU compile with nvcc and compile the kernel.cu file
