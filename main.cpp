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
#define N 8192

template <typename T>
static std::vector<T> generate_data(size_t size)
{
    using value_type = T;
    // We use static in order to instantiate the random engine
    // and the distribution once only.
    // It may provoke some thread-safety issues.
    static std::uniform_int_distribution<value_type> distribution(
        std::numeric_limits<value_type>::min(),
        std::numeric_limits<value_type>::max());
    static std::default_random_engine generator;

    std::vector<value_type> data(size);
    std::generate(data.begin(), data.end(), []() { return distribution(generator); });
    return data;
}

int main() {
        using value_type = float;
        auto vec = std::vector<std::vector<int> > { {1, 2, 3}, {1, 2, 3}, {1, 2, 3} };
        //Matrix<double> mt = Matrix<double>(std::vector<std::vector<double> > { {1, 2, 3}, {1, 2, 3}, {1, 2, 3} });
        //Matrix<double> ma( Matrix<double>(std::vector<std::vector<double> > { {1, 2, 3}, {1, 2, 3}, {1, 2, 3} }));
        srand(getpid());
        std::vector<std::vector<value_type>>cpy;
        //std::vector<std::vector<boost::rational<long int>> > cpy;
        for(size_t i = 0; i < N; i++) {
                // std::vector<boost::rational<long int>>v;
                std::vector<value_type> v;
                 for(size_t j = 0u; j < N; j++) {
                         //v.push_back(j);
                           v.push_back(static_cast<float>(rand()));
                        // v.push_back(boost::rational<long int>(rand() % 100, rand() % 100));
                 }
                 cpy.push_back(v);
                // cpy.push_back(generate_data<int>(N));
         }


//         Matrix<boost::rational<long int>> ma(cpy);

//         Matrix<boost::rational<long int>> mb(cpy);

        Matrix<value_type> ma(cpy);
        Matrix<value_type> mb(cpy);

        //Matrix<int> ma( Matrix<int>(std::vector<std::vector<int> > { {1, 2, 3, 4}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}} ));
        //Matrix<int> mb( Matrix<int>(std::vector<std::vector<int> > { {1, 2, 3, 4}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15} } ));
        //Matrix<double> mc(Matrix<double>(std::vector<std::vector<double>> { {5, 2, 6}, {9, 4, 12} , {1, 2, 3} }));

         std::cout << "begin" << std::endl << std::endl;
         auto start = std::chrono::high_resolution_clock::now();

/*
triangulaire
*/

        //mc.triangular_inferior();
        //mc.display();

        //mc.triangular_inferior();
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
        auto at = ma.strassen(mb);
        //at.display();
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


        std::cout <<std::endl << "end" << std::endl;
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        // To get the value of duration use the count()
        // member function on the duration object
        std::cout << duration.count() << std::endl;




        return 0;
}