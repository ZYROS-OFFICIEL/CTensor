This The documentation for CTensor
I/ Tesnor declaration:
Tensor a = Tensor::empty((dims),grad)
size_t dtype_size(DType dt)

Returns size in bytes of each element.

const char* dtype_to_cstr(DType d)

Converts dtype to a string ("Float32", "Int32", "Double64").

read_scalar_at(const void* data, size_t idx, DType dt)

Reads a scalar value from raw memory at a given index, returning it as double.

write_scalar_at(void* data, size_t idx, DType dt, double val)

Writes a scalar to raw memory at a given index, converting from double to the tensor’s dtype.