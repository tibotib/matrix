#include "matrix.hpp"
#include "identity.hpp"
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <functional>
#include <chrono>
#include <boost/rational.hpp>
#include <unistd.h>
#define N 2048*2

template <typename T>
void display(const std::vector<T> &a) {
        for(int i = 0; i < a.size(); i++)
                std::cout << a[i] << " ";
        std::cout << std::endl;
}

int main() {
        using value_type = int;
        srand(getpid());
        std::vector<std::vector<value_type>>cpy;
        //std::vector<std::vector<boost::rational<long int>> > cpy;
        for(size_t i = 0; i < N; i++) {
                // std::vector<boost::rational<long int>>v;
                std::vector<value_type> v;
                 for(size_t j = 0u; j < N; j++) {
                         //v.push_back(j);
                           v.push_back((rand()));
                        // v.push_back(boost::rational<long int>(rand() % 100, rand() % 100));
                 }
                 cpy.push_back(v);
                // cpy.push_back(generate_data<int>(N));
         }


//         Matrix<boost::rational<long int>> ma(cpy);

//         Matrix<boost::rational<long int>> mb(cpy);

        //Matrix<value_type> ma(cpy);
        //Matrix<value_type> mb(cpy);

        Matrix<value_type> ma(std::vector<std::vector<value_type> > { {1, 2, 3, 4}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}} );
        Matrix<value_type> mb(std::vector<std::vector<value_type> > { {1, 2, 3, 4}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15} } );
        Matrix<value_type> mc(Matrix<value_type>(std::vector<std::vector<value_type>> { {5, 2, 6}, {9, 4, 12} , {1, 2, 3} }));

         std::cout << "begin" << std::endl << std::endl;
         auto start = std::chrono::high_resolution_clock::now();

        // std::cout << ma.norm_vector(1) << std::endl;
/*
triangulaire
*/

        //mc.display();
        //std::cout << std::endl;
        //mc.triangular_inferior();
        //mc.display();
        //std::cout << std::endl;
        //mc.triangular_superior();
        //mc.display();

/*
Identity
*/
        //Matrix<int>a = Identity<int>(5, 5);
        //a.display();
        //ma.display();
/*
strassen
*/

        //auto at = ma.strassen(mb);
        //mc.echelon_form();
        //mc.display();
/*
LU decomposition
*/

        //Matrix<value_type> d(std::vector<std::vector<value_type>>({{2,3,1,5}, {6,13,5,19}, {2,19,10,23}, {4,10,11,31}}));
        //auto PLU = d.decompositionPLU();
        //std::get<0>(PLU).display();
        //std::get<1>(PLU).display();
        //std::get<2>(PLU).display();
/*
pow
*/
        //auto r = mc.pow(100);
        //r.display();
/*
multiplication en chaines
*/
        //Matrix<double>a1(Matrix<double>(std::vector<std::vector<double>>{ {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1} }));
        //Matrix<double>a2(Matrix<double>(std::vector<std::vector<double>>{ {1, 1}, {1, 1}, {1, 1}, {1, 1}, {1, 1} }));
        //Matrix<double>a3(Matrix<double>(std::vector<std::vector<double>>{ {1, 1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1} }));
        //Matrix<double>a4(Matrix<double>(std::vector<std::vector<double>>{ {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1} }));

        //std::vector<Matrix<double>> vec_matrix{a1, a2, a3, a4};
        //auto ret = mc.dot(vec_matrix);
        //ret.display();
/*
decomposition QR using gram schmidt or householder
*/
        //Matrix<double> q = mc.gram_schmidt();
        //auto qt = q;
        //qt.transpose();
        //(qt.dot(q)).display();

        //auto m  = Matrix<double>(std::vector<std::vector<double>>{{12,-51,4},{6,167,-68},{-4,24,-41}});
        //auto qr = ma.decompositionQR();
        //qr.first.display();
        //qr.second.display();
/*
gaussian inverse modular or not
*/

        //Matrix <value_type> inv = mc.inverse_mod(113);
        //inv.display();

/*
fft
*/
        //Matrix<int>poly_a(std::vector<std::vector<int>> {{1,2,3,4,5,6,7,8,9,10,11,12,13,14}});
        //auto res = poly_a.fft();

/*
kernel
*/
        //auto m = Matrix<double>(std::vector<std::vector<double>>{{1,0,-3,0,2,-8}, {0,1,5,0,-1,4},{0,0,0,1,7,-9},{0,0,0,0,0,0}});
        //auto r = ma.kernel();
        //r.display();

/*
eigen values
*/
        //Matrix<double>(std::vector<std::vector<double>>{{0,1,0},{1,-1,1},{0,1,0}});
        //auto a = Matrix<double>(std::vector<std::vector<double>>{{12,-51,4},{6,167,-68},{-4,24,-41}});
        //auto r = a.eigen();
        //for(auto e : r.first)
        //        std::cout << e << " ";
        //std::cout << std::endl << std::endl;
        //r.second.display();

/*
svd decomposition
*/
        //Matrix<double> a = Matrix<double>(std::vector<std::vector<double>>{{3,1,1},{-1,3,1}});
        //auto svd = a.decompositionSVD();
        //std::get<0>(svd).display();
        //std::get<1>(svd).display();
        //std::get<2>(svd).display();

/*
polynom multiplication
*/
        //Matrix<int> p1(std::vector<std::vector<int>>{{3,2,1}});
        //Matrix<int> p2(std::vector<std::vector<int>>{{4,8,9}});
        //auto res = p1.polynom_multiplication(p2);
        //res.display();

/*
SNF
*/
        Matrix<int> a = Matrix<int>(std::vector<std::vector<int>>{{2,4,4},{-6,6,1},{0,0,0}});
        a.display();
        a.SNF().display();


        std::cout <<std::endl << "end" << std::endl;
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        // To get the value of duration use the count()
        // member function on the duration object
        std::cout << duration.count() << std::endl;




        return 0;
}
