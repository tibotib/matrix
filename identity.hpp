#ifndef __IDENTITY_H__
#define __IDENTITY_H__
#include "matrix.hpp"
//extern template class Matrix<int>;

template <typename T>
class Identity : public Matrix<T> {
public :
        Identity(size_t length = 0u);
        Matrix<T> operator*(const Matrix<T> &mt)const;
        Matrix<T> operator*(const T)const;
};

template <typename T>
Identity<T>::Identity(size_t length) : Matrix<T>(length, length) {
        for(size_t i = 0u; i < length; i++) {
                this->m_tab[i][i] = 1;
        }
}

template <typename T>
Matrix<T> Identity<T>::operator*(const T ele)const {
        Matrix<T>ret(this->m_length, this->m_length);
        for(size_t i = 0u; i < this->m_length; i++) {
                ret.m_tab[i][i] = ele;
        }
        return ret;
}

template <typename T>
Matrix<T> Identity<T>::operator*(const Matrix<T> &mt)const {
        return mt;
}

#endif
