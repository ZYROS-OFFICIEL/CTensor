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

    Struct: Tensor
Members
Member	Type	Description
void* data	Pointer to tensor data.	
void* grad	Pointer to gradient data (optional).	
size_t ndim	Number of dimensions.	
size_t* shape	Array of dimension sizes.	
size_t* strides	Strides (steps to move to next element along each dimension).	
bool requires_grad	Whether tensor participates in gradient computations.	
DType dtype	Data type.

    Constructors :   

Tensor(const std::vector<size_t>& shape, DType dtype, bool requires_grad)

Creates a tensor with given shape, dtype, and gradient flag.
Allocates memory for data, shape, strides, and optionally grad.

Tensor()
Default constructor. Creates an empty tensor.

Tensor(const Tensor& other)
Deep copy constructor. Allocates new memory and copies content.

Tensor(Tensor&& other) noexcept
Move constructor — transfers ownership of memory.

Tensor& operator=(Tensor&& other) noexcept
Move assignment operator.

~Tensor()
Destructor — frees all allocated memory.

    Methods:

size_t numel_() const
Returns total number of elements = product of all shape dimensions.

std::vector<size_t> shape_() const
Returns a vector copy of the tensor shape.

void print_shape() const
Prints shape like (3, 2, 4).

     Indexing via Proxies
Tensor::Proxy and Tensor::ConstProxy

Provide Python-like indexing:

Tensor t({2, 3});
t[0][1] = 5.0;
double val = t[0][1];


Proxy → read/write

ConstProxy → read-only

     Basic Methods
Classes 	Description
Tensor::ones(shape, dtype)	Returns tensor filled with ones.
Tensor::zeros(shape, dtype)	Returns tensor filled with zeros.
Tensor::full(shape, value, dtype)	Returns tensor filled with value.
Tensor::rand(shape, dtype)	Fills tensor with random values in [0, 1].
Tensor::empty(shape, dtype)	Allocates memory without initialization.

    Type Utilities
Method	Description
_dtype()	Returns tensor dtype.
dtype_name()	Returns dtype as string.
dtype_bytes()	Returns element size in bytes.

    Type Conversion
Tensor astype(DType new_dtype) const
Returns a new tensor converted to another dtype.

void to_(DType new_dtype)
In-place dtype conversion.

    Shape Manipulation
Tensor t() const

Returns a new tensor with last two dimensions swapped (matrix transpose).

Tensor& t_()

In-place transpose — swaps shape and strides of the last two dimensions.

Tensor permute(const std::vector<size_t>& dims) const

Reorders dimensions according to dims.

Example:
Tensor x({2, 3, 4});
Tensor y = x.permute({1, 0, 2});

    Printing :
print_t(const Tensor& t)

Flat print of tensor values.

print_(const Tensor& t)

Recursive print with {} braces, preserving shape.