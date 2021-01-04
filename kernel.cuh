namespace GPU {
        template <typename T>
        void add(T *a, T *b, T *res, size_t size);

        template <typename T>
        void dot(T *a, T *b, T *c, size_t same_for_both, size_t length, size_t width);

        template <typename T>
        void transpose(T *a, size_t length, size_t width);
};
