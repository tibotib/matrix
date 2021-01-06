#ifndef __MATRIX__
#define __MATRIX__
#include <iostream>
#include <vector>
#include <thread>
#include <future>
#include <iomanip>
#include <stdio.h>
#include <stdarg.h>
#include <iterator>
#include <algorithm>
#include <cstring>
#include "kernel.cuh"
#include <immintrin.h>
#include <fstream>
#include <limits>

template <typename T>
class Matrix {
        template <typename U>
        friend U det_recursiv(const Matrix<U> &a);

        //template <typename U>
        //friend Matrix<U> strassen_recursiv(const Matrix<U> &a, const Matrix<T> &b);

        protected :
                std::vector<std::vector<T>> m_tab;
                size_t m_length;
                size_t m_width;
                void multiply_line(T factor, size_t line);
                void add_line(size_t res_line, size_t other_line);
                void multiply_and_add(size_t res_line, T factor, size_t other_line);

        public :
                Matrix(const std::vector<std::vector<T> > &tab);
                Matrix(std::vector<std::vector<T> > &&tab);
                Matrix(const Matrix<T> &mt);
                Matrix(Matrix<T> &&mt);
                Matrix(std::ifstream file);
                Matrix(size_t nb_col = 0u, size_t nb_line = 0u, T def = 0);
                Matrix<T>& operator=(const Matrix<T> &);
                ~Matrix();

                T getElement(size_t length, size_t width)const;
                void setElement(size_t length, size_t width, T value);
                void display()const;
                T& operator()(size_t i, size_t j);

                size_t getLength() const;
                size_t getWidth() const;
                const T *getBuffer(size_t length) const;
                const T *getData() const;
                Matrix<T> subMatrix(size_t i, size_t j, size_t i_end, size_t j_end)const;

                Matrix<T> add(const Matrix<T> &a)const;
                Matrix<T> sub(const Matrix<T> &a)const;

                Matrix<T> dot(const Matrix<T> &a)const;
                Matrix<T> dot(const std::vector<Matrix<T>> &vec);
                Matrix<T> dot(const T scalair) const;

                Matrix<T> strassen(Matrix<T> a);
                Matrix<T> pow(int n);

                Matrix<T> operator+(const Matrix<T> &a)const;
                Matrix<T> operator+=(const Matrix<T> &a) const;

                Matrix<T> operator-(const Matrix<T> &a) const;
                Matrix<T> operator-=(const Matrix<T> &a) const;

                Matrix<T> operator*(const Matrix<T> &a) const;
                Matrix<T> operator*=(const Matrix<T> &a)const;

                Matrix<T> operator*(const T &a)const;
                Matrix<T> operator*=(const T &a)const;

                bool operator==(const Matrix<T> &)const;
                bool operator!=(const Matrix<T> &)const;

                bool isSquare()const;
                T det()const;
                size_t determinant() const;

                void transpose();
                void triangular_inferior();
                void triangular_superior();
                Matrix<T> inverse_const()const;
                Matrix<T> inverse();
                T trace()const;
                void resize(size_t n);
                void resize(size_t length, size_t width);

                Matrix<T> addGPU(const Matrix<T> &a);
                Matrix<T> dotGPU(const Matrix<T> &a);
                Matrix<T> transposeGPU();

                Matrix<T> dot512(const Matrix<T> &b)const;
};

#include "identity.hpp"

template <typename T>
Matrix<T>::Matrix(const std::vector<std::vector<T> > &tab) : m_tab(tab), m_length(tab.size()){
        if(tab.size() > 0)
                this->m_width = tab[0].size();
}

template <typename T>
Matrix<T>::Matrix(std::vector<std::vector<T> > &&tab) : m_tab(std::move((std::vector<std::vector<T> > &&) tab)), m_length(m_tab.size())
{
        if(m_tab.size() > 0)
                this->m_width = m_tab[0].size();
        tab.clear();
}

template<typename T>
Matrix<T>::Matrix(const Matrix<T> &mt) : m_tab(mt.m_length, std::vector<T>(mt.m_width)), m_length(mt.m_length), m_width(mt.m_width) {
        for(size_t i = 0u; i < this->m_length; i++)
                for(size_t j = 0u; j < this->m_width; j++)
                        this->m_tab[i][j] = mt.m_tab[i][j];
}

template <typename T>
Matrix<T>::Matrix(Matrix<T> &&mt) : m_tab(std::move((std::vector<std::vector<T> > &&) mt.m_tab)), m_length(mt.m_length), m_width(mt.m_width) {
        mt.m_tab.clear();
}

template <typename T>
Matrix<T>::Matrix(size_t length, size_t width, T def) : m_tab(length, std::vector<T>(width)), m_length(length), m_width(width) {
        if(def != 0) {
                for(size_t i = 0u; i < this->m_length; i++) {
                        for(size_t j = 0u; j < this->m_width; j++) {
                                this->m_tab[i][j] = def;
                        }
                }
        }
}

template <typename T>
Matrix<T>::Matrix(std::ifstream file) {
        if(!file)
                throw std::invalid_argument("File is incorrect");
        char* tok = (char*) alloca(1);
        *tok = ',';

        std::string str;
        std::getline(file, str);
        int length = std::stoi(str);

        std::getline(file, str);
        int width = std::stoi(str);
        std::vector<std::vector<T>>tab(length, std::vector<T>(width));
        size_t i = 0u;
        while(std::getline(file, str)) {
                size_t j = 0u;
                char *ch  = (char*) str.c_str();
                char *ele = strtok(ch, tok);
                while(ele != nullptr) {
                        tab[i][j] = atoi(ele);
                        ele       = strtok(nullptr, tok);
                        ++j;
                }
                ++i;
        }
        this->m_length = length;
        this->m_width  = width;
        this->m_tab    = tab;
}

template <typename T>
Matrix<T>::~Matrix() {
        this->m_tab.clear();
}

template <typename T>
Matrix<T>& Matrix<T>::operator=(const Matrix<T> &a) {
        this->m_length = a.m_length;
        this->m_width  = a.m_width;
        this->m_tab.clear();
        this->m_tab = std::vector<std::vector<T>>(this->m_length, std::vector<T>(a.m_width));
        for(size_t i = 0u; i < a.m_length; i++)
                for(size_t j = 0u; j < a.m_width; j++)
                        this->m_tab[i][j] = a.m_tab[i][j];
        return *this;

}

template <typename T>
T Matrix<T>::getElement(size_t length, size_t width)const{
        if(length > this->m_length - 1 || width > this->m_width - 1)
                throw std::invalid_argument("exceed matrix size\n");

        return m_tab[length][width];
}

template <typename T>
void Matrix<T>::setElement(size_t length, size_t width, T value) {
        if(length > this->m_length - 1 || width > this->m_width - 1)
                throw std::invalid_argument("exceed matrix size\n");

        this->m_tab[length][width] = value;
}

template <typename T>
void Matrix<T>::display()const {
        for(size_t i = 0; i < this->m_length; i++) {
                for(size_t j = 0; j < this->m_width; j++) {
                        std::cout <<  this->m_tab[i][j] << " ";
                }
                std::cout << std::endl;
        }
}

template <typename T>
size_t Matrix<T>::getLength() const {
        return this->m_length;
}

template<typename T>
size_t Matrix<T>::getWidth()const {
        return this->m_width;
}

template <typename T>
const T *Matrix<T>::getBuffer(size_t length) const{
        return this->m_tab[length].data();
}

template <typename T>
const T *Matrix<T>::getData() const {
        return this->m_tab[0].data();
}

template <typename T>
Matrix<T> Matrix<T>::subMatrix(size_t i, size_t j, size_t i_end, size_t j_end)const {
        if(i > i_end || i_end >= this->m_length || j > j_end || j_end >= this->m_width)
                throw std::invalid_argument("Bad index for submatrix\n");

        Matrix<T>ret(i_end - i, j_end - j);
        for(; i < i_end; i++) {
                for(size_t k = j; k < j_end; k++){
                        ret.m_tab[i][k] = this->m_tab[i][k];
                }
        }
        return ret;
}

template <typename T>
Matrix<T> Matrix<T>::add(const Matrix<T> &a)const {
        if(a.m_length != this->m_length || a.m_width != this->m_width)
                return Matrix<T>();
        Matrix<T> ret(this->m_length, this->m_width);

        for(size_t i = 0; i < this->m_length; i++) {
                for(size_t j = 0u; j < this->m_width; j++) {
                        ret.m_tab[i][j] = this->m_tab[i][j] + a.m_tab[i][j];
                }
        }
        return ret;
}

template <typename T>
Matrix<T> Matrix<T>::sub(const Matrix<T> &a)const {
        if(a.m_length != this->m_length || a.m_width != this->m_width)
                return Matrix<T>();

        Matrix<T> ret(this->m_length, this->m_width);
        for(size_t i = 0; i < this->m_length; i++) {
                for(size_t j = 0u; j < this->m_width; j++) {
                        ret.m_tab[i][j] = this->m_tab[i][j] - a.m_tab[i][j];
                }
        }
        return ret;
}

template <typename T>
Matrix<T> Matrix<T>::operator+(const Matrix<T> &a)const {
        return this->add(a);
}

template <typename T>
Matrix<T> Matrix<T>::operator-(const Matrix<T> &a) const {
        return this->sub(a);
}

template<typename T>
Matrix<T> Matrix<T>::operator+=(const Matrix<T> &a) const {
        *this = *this + a;
        return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator-=(const Matrix<T> &a) const {
        *this = *this - a;
        return *this;
}
/*        if(sizeof(T) == 32) {
                int length = this->m_length >> 3;
                int width  = this->m_width >> 3;

                for(int i = 0; i < length; i++) {
                        for(int j = 0; j < width; j+=8) {
                                __m256i f = _mm256_set_epi32(this->m_tab[i][j], this->m_tab[i][j+1], this->m_tab[i][j+2], this->m_tab[i][j+3], this->m_tab[i][j+4], this->m_tab[i][j+5], this->m_tab[i][j+6], this->m_tab[i][j+7]);
                                __m256i s = _mm256_set_epi32(a.m_tab[i][j], a.m_tab[i][j+1], a.m_tab[i][j+2], a.m_tab[i][j+3], a.m_tab[i][j+4], a.m_tab[i][j+5], a.m_tab[i][j+6], a.m_tab[i][j+7]);
                                __m256i r = _mm256_add_epi32(f, s);
                                T *res    = (T*) (&r);
                                ret.m_tab[i][j]   = res[7];
                                ret.m_tab[i][j+1] = res[6];
                                ret.m_tab[i][j+2] = res[5];
                                ret.m_tab[i][j+3] = res[4];
                                ret.m_tab[i][j+4] = res[3];
                                ret.m_tab[i][j+5] = res[2];
                                ret.m_tab[i][j+6] = res[1];
                                ret.m_tab[i][j+7] = res[0];
                        }
                }
                for(int i = 0; i < this->m_length % 8; i++) {
                        for(int j = 0; j < this->m_width % 8; j++) {
                                ret.m_tab[i][j] = this->m_tab[i][j] + a.m_tab[i][j];
                        }
                }
        }*/
template <typename T>
Matrix<T> Matrix<T>::dot(const Matrix<T> &a)const {
        if(this->m_width != a.m_length)
                throw std::invalid_argument("Bad size for multiplication\n");

        Matrix<T> ret(this->m_length, a.m_width);
        std::vector<T> colA(this->m_width);
        for(size_t i = 0u; i < a.m_width; i++) {
                for(int j = 0; j < this->m_width; j++)
                        colA[j] = a.m_tab[j][i];
                for(size_t j = 0u; j < this->m_length; j++) {
                        T tmp = 0;
                        for(size_t k = 0u; k < this->m_width; k++) {
                                tmp += this->m_tab[j][k] * colA[k];
                        }

                        ret.m_tab[j][i] = tmp;
                }
        }
        return ret;
}

template <typename T>
static std::pair<int, int> get_dim(const std::vector<Matrix<T>> &vec, size_t i, size_t j) {
        if(i >= vec.size() || j >= vec.size())
                throw std::invalid_argument("Can\'t compute matrix product bad sizes\n");
        if(i == j)
                return std::pair<int, int>(vec[i].getLength(), vec[i].getWidth());
        return std::pair<int, int>(vec[i].getLength(), vec[j].getWidth());
}

template <typename T>
static std::vector<int> get_length(const std::vector<Matrix<T>> &vec) {
        std::vector<int> ret(vec.size() + 1);
        for(size_t i = 0; i < vec.size(); i++)
                ret[i] = vec[i].getLength();
        ret[vec.size()] = vec.back().getWidth();
        return ret;
}

template <typename T>
static Matrix<T>dot_static(const std::vector<std::vector<int>>  &s, const std::vector<Matrix<T>> &grid, int i, int j) {
        //calcule la multiplication de i a j
        std::cout << "ok" << std::endl;
        if(i == j) {
                grid[i].display();
                return grid[i];
        }
        auto a = dot_static(s, grid, i, s[i][j]-1);
        auto b = dot_static(s, grid, s[i][j], j);
        return a.strassen(b);

        //return
        //return dot(s, grid, s[i][j], )
}

template <typename T>
Matrix<T> Matrix<T>::dot(const std::vector<Matrix<T>> &vec) {

        auto p = get_length(vec);
        std::vector<std::vector<unsigned int>> tab(p.size(), std::vector<unsigned int>(p.size()));
        std::vector<std::vector<int>> s(p.size(), std::vector<int>(p.size()));

        for(size_t l = 1; l < p.size(); l++) {
                for(int j = l, i = 1; j < p.size(); j++, i++) {
                        tab[i-1][j] = std::numeric_limits<unsigned int>::max() ;
                        for(int k = i-1; k < j; k++) {
                                auto q = tab[i-1][k] + tab[k + 1][j] + p[i-1]*p[k+1]*p[j+1];
                                if(q < tab[i-1][j]) {
                                        tab[i-1][j] = q;
                                        s[i-1][j]   = k+1;
                                }

                        }
                }
        }

        return dot_static(s, vec, 0, vec.size() - 1);
}

template <typename T>
Matrix<T> Matrix<T>::dot(const T scal) const {
        auto ret(*this);

        for(size_t i = 0u; i < this->m_length; i++)
                for(size_t j = 0u; j < this->m_width; j++)
                        ret.m_tab[i][j] *= scal;
        return ret;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const Matrix<T> &a)const {
        return this->dot(a);
}

template <typename T>
Matrix<T> Matrix<T>::operator*=(const Matrix<T> &a)const {
        *this = *this * a;
        return *this;
}

template <typename T>
Matrix<T> Matrix<T>::operator*(const T &a)const {
        return this->dot(a);
}

template <typename T>
Matrix<T> Matrix<T>::operator*=(const T &a)const {
        *this = *this * a;
        return *this;
}

template <typename T>
bool Matrix<T>::operator==(const Matrix<T> &a)const {
        if(a.m_length == this->m_length && a.m_width == this->m_width) {
                for(size_t i = 0; i < this->m_length; i++)
                        for(size_t j = 0; j < this->m_width; j++)
                                if(this->m_tab[i][j] != a.m_tab[i][j])
                                        return false;
        }
        return true;
}

template <typename T>
bool Matrix<T>::operator!=(const Matrix<T> &a)const {
        if(a.m_length == this->m_length && a.m_width == this->m_width) {
                for(size_t i = 0; i < this->m_length; i++)
                        for(size_t j = 0; j < this->m_width; j++)
                                if(this->m_tab[i][j] != a.m_tab[i][j])
                                        return true;
        }
        return false;
}

template <typename T>
T& Matrix<T>::operator()(size_t i, size_t j) {
        if(i >= this->m_length || j >= this->m_width)
                throw std::invalid_argument("bad index\n");
        return this->m_tab[i][j];
}

template <typename T>
bool Matrix<T>::isSquare()const {
        return this->m_length == this->m_width;
}

template <typename T>
T det_recursiv(const Matrix<T> &a) {//tres long
        if(a.m_length == 2)
                return a.m_tab[0][0] * a.m_tab[1][1] - a.m_tab[1][0] * a.m_tab[0][1];

        T ret = 0;
        for(size_t j = 0u; j < a.m_width; j++) {
                Matrix<T> tmp(a.m_width - 1, a.m_width - 1);
                for(size_t i = 1u; i < a.m_width; i++) {
                        for(size_t k = 0u; k < a.m_width; k++) {
                                if(k == j)
                                        continue;
                                if(k > j) {
                                        if(i > 0 && k > 0)
                                                tmp.m_tab[i - 1][k - 1] = a.m_tab[i][k];
                                        else if(i == 0 && k > 0)
                                                tmp.m_tab[i][k - 1] = a.m_tab[i][k];
                                        else if(i > 0 && k == 0)
                                                tmp.m_tab[i - 1][k] = a.m_tab[i][k];
                                        else
                                                tmp.m_tab[i][k] = a.m_tab[i][k];
                                }else {
                                        if(i > 0)
                                                tmp.m_tab[i - 1][k] = a.m_tab[i][k];
                                        else if(i == 0)
                                                tmp.m_tab[i][k] = a.m_tab[i][k];
                                }

                        }
                }
                if(j % 2 == 0)
                        ret += a.m_tab[0][j] * det_recursiv(tmp);
                else
                        ret -= a.m_tab[0][j] * det_recursiv(tmp);
        }
        return ret;
}

template <typename T>
T Matrix<T>::det()const {
        //very long
        if(!this->isSquare())
                throw std::invalid_argument("Can't get determinant : the matrix is not square");
        return det_recursiv(*this);
}

template <typename T>
size_t Matrix<T>::determinant() const{
        if(!this->isSquare())
                throw std::invalid_argument("Can't get determinant : the matrix is not square");

        auto tmp = (*this);
        tmp.triangular_inferior();
        T ret = tmp.m_tab[0][0];
        for(size_t i = 1; i < this->m_length; i++)
                ret *= tmp.m_tab[i][i];
        return (size_t) ret;
}

template <typename T>
void Matrix<T>::transpose() {
        auto tmp_mat(*this);
        this->m_tab = std::vector<std::vector<T>> (this->m_width, std::vector<T>(this->m_length));
        auto tmp       = this->m_length;
        this->m_length = this->m_width;
        this->m_width  = tmp;

        for(size_t i = 0u; i < this->m_length; i++) {
                for(size_t j = 0u; j < this->m_width; j++) {
                        this->m_tab[i][j] = tmp_mat.m_tab[j][i];
                }
        }
}

static float sum8(__m256 x) {
    // hiQuad = ( x7, x6, x5, x4 )
    const __m128 hiQuad = _mm256_extractf128_ps(x, 1);
    // loQuad = ( x3, x2, x1, x0 )
    const __m128 loQuad = _mm256_castps256_ps128(x);
    // sumQuad = ( x3 + x7, x2 + x6, x1 + x5, x0 + x4 )
    const __m128 sumQuad = _mm_add_ps(loQuad, hiQuad);
    // loDual = ( -, -, x1 + x5, x0 + x4 )
    const __m128 loDual = sumQuad;
    // hiDual = ( -, -, x3 + x7, x2 + x6 )
    const __m128 hiDual = _mm_movehl_ps(sumQuad, sumQuad);
    // sumDual = ( -, -, x1 + x3 + x5 + x7, x0 + x2 + x4 + x6 )
    const __m128 sumDual = _mm_add_ps(loDual, hiDual);
    // lo = ( -, -, -, x0 + x2 + x4 + x6 )
    const __m128 lo = sumDual;
    // hi = ( -, -, -, x1 + x3 + x5 + x7 )
    const __m128 hi = _mm_shuffle_ps(sumDual, sumDual, 0x1);
    // sum = ( -, -, -, x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 )
    const __m128 sum = _mm_add_ss(lo, hi);
    return _mm_cvtss_f32(sum);
}

namespace Type {
        template< class T >
        struct TypeIsInt
        {
            static const bool value = false;
        };

        template<>
        struct TypeIsInt< int >
        {
            static const bool value = true;
        };

        template< class T >
        struct TypeIsFloat
        {
            static const bool value = false;
        };

        template<>
        struct TypeIsFloat< float >
        {
            static const bool value = true;
        };

        template< class T >
        struct TypeIsDouble
        {
            static const bool value = false;
        };

        template<>
        struct TypeIsDouble< double >
        {
            static const bool value = true;
        };
};
/*if(Type::TypeIsInt<T>::value) {
        //int tmp_tab[sizeof(__m256i)/sizeof(int)];
        for(size_t k = 0; k < this->m_width; k+=8) {
                __m256i f  = _mm256_set_epi32(m_tab[j][k], m_tab[j][k+1], m_tab[j][k+2], m_tab[j][k+3], m_tab[j][k+4], m_tab[j][k+5], m_tab[j][k+6], m_tab[j][k+7]);
                __m256i s = _mm256_set_epi32(colA[k], colA[k+1], colA[k+2], colA[k+3],colA[k+4], colA[k+5], colA[k+6], colA[k+7]);
                __m256i r = _mm256_mullo_epi32(f, s);

                //int *tmp_tab = (int*)&r;

                //_mm256_store_si256((__m256i *)tmp_tab, r);
                //tmp += tmp_tab[0] + tmp_tab[1]+ tmp_tab[2]+ tmp_tab[3]+ tmp_tab[4]+ tmp_tab[5]+ tmp_tab[6]+ tmp_tab[7];
                s = _mm256_permute2f128_si256(r, r, 1);
                r = _mm256_add_epi32(r, s);
                r = _mm256_hadd_epi32(r, r);
                r = _mm256_hadd_epi32(r, r);
                tmp += _mm256_extract_epi32(r, 0);
        }
}*/
template <typename T>
Matrix<T> Matrix<T>::dot512(const Matrix<T> &a)const {
        if(this->m_width != a.m_length)
                throw std::invalid_argument("Bad size for multiplication\n");

        Matrix<T> ret(this->m_length, a.m_width);
        std::vector<T> colA(this->m_width);
        for(size_t i = 0u; i < a.m_width; i++) {
                for(size_t j = 0u; j < this->m_width; j++)
                        colA[j] = a.m_tab[j][i];

                for(size_t j = 0u; j < this->m_length; j++) {
                        T tmp = 0;
#ifdef __AVX2__
                        if(sizeof(T) == 4) {

                                if(Type::TypeIsFloat<T>::value) {
                                        for(size_t k = 0; k < this->m_width; k+=8) {
                                                __m256 vecA = _mm256_set_ps(m_tab[j][k], m_tab[j][k+1], m_tab[j][k+2], m_tab[j][k+3], m_tab[j][k+4], m_tab[j][k+5], m_tab[j][k+6], m_tab[j][k+7]);
                                                __m256 vecB = _mm256_set_ps(colA[k], colA[k+1], colA[k+2], colA[k+3],colA[k+4], colA[k+5], colA[k+6], colA[k+7]);
                                                __m256 r    = _mm256_mul_ps(vecA, vecB);
                                                tmp += sum8(r);
                                                goto end;
                                        }
                                }
                        }

                        else if(sizeof(T) == 8) {

                                if(Type::TypeIsDouble<T>::value) {
                                        for(size_t k = 0; k < this->m_width; k+=4) {
                                                __m256d vecA = _mm256_set_pd(m_tab[j][k], m_tab[j][k+1], m_tab[j][k+2], m_tab[j][k+3]);
                                                __m256d vecB = _mm256_set_pd(colA[k], colA[k+1], colA[k+2], colA[k+3]);
                                                __m256d r    = _mm256_mul_pd(vecA, vecB);
                                                double *tmp_tab = (double*)&r;
                                                tmp += tmp_tab[0] + tmp_tab[1]+ tmp_tab[2]+ tmp_tab[3];
                                        }
                                        goto end;
                                }
                        }
#endif
                        for(size_t k = 0; k < this->m_width; k++) {
                                tmp += this->m_tab[j][k] * colA[k];
                        }
                        end :
                                ret.m_tab[j][i] = tmp;
                }
        }
        return ret;
}

template <typename T>
Matrix<T> strassen_recursiv(const Matrix<T> &a, const Matrix<T> &b) {
        if(a.getLength() <= 512)
                return a.dot512(b);

        size_t k = a.getLength() / 2;
        Matrix<T> a11(k, k);
        Matrix<T> a12(k, k);
        Matrix<T> a21(k, k);
        Matrix<T> a22(k, k);

        Matrix<T> b11(k, k);
        Matrix<T> b12(k, k);
        Matrix<T> b21(k, k);
        Matrix<T> b22(k, k);

//#pragma omp parallel for
        for(size_t i = 0u; i < k; i++) {
//#pragma omp parallel for
            for(size_t j = 0u; j < k; j++) {
                a11.setElement(i, j, a.getElement(i, j));
                a12.setElement(i, j, a.getElement(i, k + j));
                a21.setElement(i, j, a.getElement(k + i, j));
                a22.setElement(i, j, a.getElement(k + i, k + j));
                b11.setElement(i, j, b.getElement(i, j));
                b12.setElement(i, j, b.getElement(i, k + j));
                b21.setElement(i, j, b.getElement(k + i, j));
                b22.setElement(i, j, b.getElement(k + i, k + j));
            }
        }
        Matrix<T> m1, m2, m3, m4, m5, m6, m7;
        #pragma omp parallel
        {
            #pragma omp single
                    {

                 #pragma omp task shared(m1)
                        m1  = strassen_recursiv((a11 + a22) , (b11 + b22));
                 #pragma omp task shared(m2)
                        m2  = strassen_recursiv((a21 + a22) , b11);
                 #pragma omp task shared(m3)
                        m3  = strassen_recursiv(a11, (b12 - b22));
                 #pragma omp task shared(m4)
                        m4  = strassen_recursiv(a22, (b21 - b11));
                 #pragma omp task shared(m5)
                        m5  = strassen_recursiv((a11 + a12) , b22);
                 #pragma omp task shared(m6)
                        m6  = strassen_recursiv((a21 - a11) , (b11 + b12));
                 #pragma omp task shared(m7)
                        m7  = strassen_recursiv((a12 - a22) , (b21 + b22));
                }
        }

 #pragma omp taskwait
        auto c11  = m1 + m4 - m5 + m7;
        auto c12  = m3 + m5;
        auto c21  = m2 + m4;
        auto c22  = m1 - m2 + m3 + m6;


        Matrix<T> ret = Matrix<T>(a.getLength(), a.getLength());

        for(size_t i = 0u; i < k; i++) {
            for(size_t j = 0u; j < k; j++) {
                ret.setElement(i, j, c11.getElement(i, j));
                ret.setElement(i, j + k, c12.getElement(i, j));
                ret.setElement(k + i, j, c21.getElement(i, j));
                ret.setElement(k + i, k + j, c22.getElement(i, j));
            }
        }

        return ret;
}

template <typename T>
static Matrix<T> pow_recursiv(Matrix<T> a, Matrix<T> b, int n) {
        if(n == 0)
                return a.strassen(b);
        if(n % 2 == 0)
                return pow_recursiv<T>(a, b.strassen(b), n / 2);
        return pow_recursiv<T>(a.strassen(b), b, (n - 1) / 2);

}

template <typename T>
Matrix<T> Matrix<T>::pow(int n) {
        if(!this->isSquare())
                throw std::invalid_argument("Can\'t compute pow, the matrix is not square\n");
        return pow_recursiv(Identity<T>(this->m_length), *this, n);
}

template <typename T>
void Matrix<T>::resize(size_t s) {
        if(s == this->m_length && s == this->m_width)
                return ;
        if(s > this->m_length || s > this->m_width) {
                for(size_t i = this->m_length; i < s; i++) {
                        std::vector<T> vec(s, 0);
                        this->m_tab.push_back(vec);
                }

                for(size_t i = 0u; i < this->m_length; i++) {
                        for(size_t j = this->m_width; j < s; j++) {
                                this->m_tab[i].push_back(0);
                        }
                }
                this->m_length = s;
                this->m_width  = s;
        }
}

template <typename T>
void Matrix<T>::resize(size_t length, size_t width) {
        if(length < this->m_length) {
                auto i = length;
                while(i != this->m_length) {
                        this->m_tab.pop_back();
                        ++i;
                }
        }
        this->m_length = length;
        if(width < this->m_width) {
                for(size_t i = 0u; i < this->m_length; i++) {
                        auto j = width;
                        while(j != this->m_width) {
                                this->m_tab[i].pop_back();
                                ++j;
                        }
                }
        }
        this->m_width = width;
}

template <typename T = int>
T max(size_t num_arg, ...) {
        va_list vl;
        va_start(vl, num_arg);

        T ret = 0;
        for(size_t i = 0u; i < num_arg; i++) {
                T ele = va_arg(vl, T);
                ret   = (ret > ele) ? ret : ele;
        }
        va_end(vl);
        return ret;
}

size_t get_power(int n) {
        if((n & (n - 1)) == 0)
                return n;

        int length        = n;
        int l             = 1;
        while(length > 0) {
                length = length >> 1;
                l = l << 1;
        }
        return l;

}

template <typename T>
Matrix<T> Matrix<T>::strassen(Matrix<T> a) {
        if(this->m_width != a.m_length)
                throw std::invalid_argument("Bad size for multiplication\n");
        if(this->m_width <= 512 && a.m_width <= 512 && this->m_length <= 512)
                return this->dot(a);
        auto l = this->m_length;
        auto w = a.m_width;
        size_t size = get_power(max(3, this->m_width, a.m_width, this->m_length));
        this->resize(size);
        a.resize(size);
        auto ret = strassen_recursiv(*this, a);
        ret.resize(l, w);
        return ret;
}

template <typename T>
void Matrix<T>::multiply_line(T factor, size_t line) {
        for(size_t j = 0u; j < this->m_width; j++) {
                this->m_tab[line][j] *= factor;
        }
}

template <typename T>
void Matrix<T>::add_line(size_t res_line, size_t other_line) {
        for(size_t j = 0u; j < this->m_width; j++) {
                this->m_tab[res_line][j] += this->m_tab[other_line][j];
        }
}

template <typename T>
void Matrix<T>::multiply_and_add(size_t res_line, T factor, size_t other_line) {
        for(size_t j = 0u; j < this->m_width; j++) {
                this->m_tab[res_line][j] += this->m_tab[other_line][j] * factor;
        }
}

template <typename T>
void Matrix<T>::triangular_inferior() {
        //ne fonctionne qu'avec des types rationnels
        for(size_t k = 0u; k < this->m_width && k < this->m_length; ++k) {
                T ref = this->m_tab[k][k];
                for(size_t i = k + 1; i < this->m_length; ++i) {
                        auto factor = (this->m_tab[i][k] / ref);
                        multiply_and_add(i, -factor, k);
                }
        }
}

template <typename T>
T min(T a, T b) {
        return (a > b) ? b : a;
}

template <typename T>
void Matrix<T>::triangular_superior() {
        //ne fonctionne qu'avec des types rationnels
        size_t compt = 1u;
        for(int k = this->m_width - 1; k >= 0; --k) {
                T ref = this->m_tab[this->m_length - compt][k];
                for(int i = this->m_length - compt - 1; i >= 0; --i) {
                        auto factor = (this->m_tab[i][k] / ref);
                        multiply_and_add(i, -factor, this->m_length - compt);
                }
                ++compt;
                if(compt > this->m_length)
                        break;
        }
}

template <typename T>
Matrix<T> Matrix<T>::inverse() {
        if(this->m_length != this->m_width)
                throw std::invalid_argument("Can\' t compute inverse : the matrix is not square");

        Matrix<T> id = Identity<T>(this->m_length);
//triangulaire superieure
        for(size_t k = 0u; k < this->m_width && k < this->m_length; ++k) {
                T ref = this->m_tab[k][k];
                if(ref == 0)
                        throw std::invalid_argument("matrix isn\'t inversible\n");
                for(size_t i = k + 1; i < this->m_length; ++i) {
                        auto factor = (this->m_tab[i][k] / ref);
                        for(size_t j = 0u; j < this->m_width; j++) {
                                id.m_tab[i][j]    -= id.m_tab[k][j] * factor;
                                this->m_tab[i][j] -= this->m_tab[k][j] * factor;
                        }

                }
        }

//matrice diagonale
        size_t compt = 1u;
        for(int k = this->m_width - 1; k >= 0; --k) {
                T ref = this->m_tab[this->m_length - compt][k];
                if(ref == 0)
                        throw std::invalid_argument("matrix isn\'t inversible\n");
                for(int i = this->m_length - compt - 1; i >= 0; --i) {
                        auto factor = (this->m_tab[i][k] / ref);
                        for(size_t j = 0u; j < this->m_width; j++) {
                                id.m_tab[i][j]    -= id.m_tab[this->m_length - compt][j] * factor;
                                this->m_tab[i][j] = this->m_tab[i][j] - this->m_tab[this->m_length - compt][j] * factor;
                        }
                }
                ++compt;
                if(compt > this->m_length)
                        break;
        }

        for(size_t i = 0; i < id.m_length; i++) {
                auto coef = this->m_tab[i][i];
                for(size_t j = 0; j < id.m_width; j++) {
                        id.m_tab[i][j] /= coef;
                }
        }

        return id;

}

template <typename T>
Matrix<T> Matrix<T>::inverse_const() const{
        if(this->m_length != this->m_width)
                throw std::invalid_argument("Can\' t compute inverse : the matrix is not square");

        auto tmp     = *this;
        Matrix<T> id = Identity<T>(tmp.m_length);

        for(size_t k = 0u; k < tmp.m_width && k < tmp.m_length; ++k) {
                T ref = tmp.m_tab[k][k];
                if(ref == 0)
                        throw std::invalid_argument("matrix isn\'t inversible\n");
                for(size_t i = k + 1; i < tmp.m_length; ++i) {
                        auto factor = (tmp.m_tab[i][k] / ref);
                        for(size_t j = 0u; j < tmp.m_width; j++) {
                                id.m_tab[i][j]    -= id.m_tab[k][j] * factor;
                                tmp.m_tab[i][j] -= tmp.m_tab[k][j] * factor;
                        }

                }
        }

        size_t compt = 1u;
        for(int k = tmp.m_width - 1; k >= 0; --k) {
                T ref = tmp.m_tab[tmp.m_length - compt][k];
                for(int i = tmp.m_length - compt - 1; i >= 0; --i) {
                        auto factor = (tmp.m_tab[i][k] / ref);
                        if(ref == 0)
                                throw std::invalid_argument("matrix isn\'t inversible\n");
                        for(size_t j = 0u; j < tmp.m_width; j++) {
                                id.m_tab[i][j]    -= id.m_tab[tmp.m_length - compt][j] * factor;
                                tmp.m_tab[i][j] = tmp.m_tab[i][j] - tmp.m_tab[tmp.m_length - compt][j] * factor;
                        }
                }
                ++compt;
                if(compt > tmp.m_length)
                        break;
        }

        for(size_t i = 0; i < id.m_length; i++) {
                auto coef = tmp.m_tab[i][i];
                for(size_t j = 0; j < id.m_width; j++) {
                        id.m_tab[i][j] /= coef;
                }
        }

        return id;

}

template <typename T>
T Matrix<T>::trace() const{
        if(this->m_length != this->m_width)
                throw std::invalid_argument("Can\' t compute trace : the matrix is not square");
        T sum = 0;
        for(size_t i = 0u; i < this->m_length; i++)
                sum += this->m_tab[i][i];
        return sum;
}

template <typename T>
Matrix<T> Matrix<T>::addGPU(const Matrix<T> &a) {
        if(a.m_length != this->m_length || a.m_width != this->m_width)
                throw std::invalid_argument("Bad size for addition\n");

        T *buffer_this = new T[this->m_tab.size() * this->m_tab[0].size()];
        T *buffer_a    = new T[this->m_tab.size() * this->m_tab[0].size()];
        T *buffer_res  = new T[this->m_tab.size() * this->m_tab[0].size()];
        for(size_t i = 0; i < this->m_tab.size(); i++) {
                memcpy(buffer_this + i * this->m_tab[0].size(), this->m_tab[i].data(), this->m_tab[0].size() * sizeof(T));
                memcpy(buffer_a + i * a.m_tab[0].size(), a.m_tab[i].data(), a.m_tab[0].size() * sizeof(T));
        }

        GPU::add(buffer_this, buffer_a, buffer_res, this->m_tab.size() * this->m_tab[0].size());
        std::vector<std::vector<T>> buff(this->m_tab.size(), std::vector<T>(this->m_tab[0].size()));
        for(size_t i = 0u; i < this->m_tab.size(); i++) {
                for(size_t j = 0u; j < this->m_tab[0].size(); j++) {
                        buff[i][j] = buffer_res[i * this->m_tab[0].size() + j];
                }
        }

        delete []buffer_this;
        delete []buffer_a;
        delete []buffer_res;
        return Matrix<T>(buff);
}

template <typename T>
Matrix<T> Matrix<T>::dotGPU(const Matrix<T> &a) {
        if(this->m_width != a.m_length)
                throw std::invalid_argument("Bad size for multiplication\n");

        T *buffer_this = new T[this->m_tab.size() * this->m_tab[0].size()];
        T *buffer_a    = new T[a.m_tab.size() * a.m_tab[0].size()];
        T *buffer_res  = new T[this->m_tab.size() * a.m_tab[0].size()];
        for(size_t i = 0; i < this->m_tab.size(); i++) {
                memcpy(buffer_this + i * this->m_tab[0].size(), this->m_tab[i].data(), this->m_tab[0].size() * sizeof(T));
        }
        for(size_t i = 0; i < a.m_tab.size(); i++) {
                memcpy(buffer_a + i * a.m_tab[0].size(), a.m_tab[i].data(), a.m_tab[0].size() * sizeof(T));
        }

        GPU::dot(buffer_this, buffer_a, buffer_res, this->m_tab[0].size(), this->m_tab.size(), a.m_tab[0].size());
        std::vector<std::vector<T>> buff(this->m_tab.size(), std::vector<T>(a.m_tab[0].size()));
        for(size_t i = 0u; i < this->getLength(); i++) {
                for(size_t j = 0u; j < a.getWidth(); j++) {
                        buff[i][j] = buffer_res[i * a.m_tab[0].size() + j];
                }
        }

        delete []buffer_this;
        delete []buffer_a;
        delete []buffer_res;
        return Matrix<T>(buff);
}

template <typename T>
Matrix<T> Matrix<T>::transposeGPU() {
        T *buffer_this = new T[this->m_tab.size() * this->m_tab[0].size()];
        for(size_t i = 0; i < this->m_tab.size(); i++) {
                memcpy(buffer_this + i * this->m_tab[0].size(), this->m_tab[i].data(), this->m_tab[0].size() * sizeof(T));
        }

        GPU::transpose(buffer_this, this->getLength(), this->getWidth());
        std::vector<std::vector<T>> buff(this->m_tab[0].size(), std::vector<T>(this->m_tab.size()));
        for(size_t i = 0u; i < this->getWidth(); i++) {
                for(size_t j = 0u; j < this->getLength(); j++) {
                        buff[i][j] = buffer_this[i * this->m_tab.size() + j];
                }
        }

        delete []buffer_this;
        return Matrix<T>(buff);
}


#endif
