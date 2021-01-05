#include "matrix.hpp"

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
Matrix<T>::Matrix(const Matrix<T> &mt) : m_tab(mt.m_tab), m_length(mt.m_length), m_width(mt.m_width) {}

template <typename T>
Matrix<T>::Matrix(Matrix<T> &&mt) : m_tab(std::move((std::vector<std::vector<T> > &&) mt.m_tab)), m_length(mt.m_length), m_width(mt.m_width) {
        mt.m_tab.clear();
}

template <typename T>
Matrix<T>::Matrix(size_t length, size_t width) : m_tab(length), m_length(length), m_width(width) {}

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
                        std::cout << this->m_tab[i][j] << " ";
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
T *Matrix<T>::getBuffer(size_t length) const{
        return this->m_tab[length].buffer();
}

//template class Matrix<double>;
