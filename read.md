This The documentation for CTensor
I/ Tensor declaration:
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

    Broadcasting :
linear_index_from_padded(const Tensor&, const std::vector<size_t>&)
Computes linear memory index for a broadcasted index vector.

Tensor pad_to_ndim(const Tensor& t, size_t target_ndim)
Pads a tensor to higher dimensions with broadcasting (like PyTorch).

broadcast_batch_shape_from_vectors(a, b)
Computes output shape when broadcasting two tensors.
normalize_
void normalize_(const std::vector<float>& mean, const std::vector<float>& stdv);


Description: Adds a normalization transformation to the pipeline. Normalizes each channel:

output=
stdv
input−mean
	​


Parameters:

mean → Vector of mean values per channel (or length 1 for all channels).

stdv → Vector of standard deviations per channel (or length 1 for all channels).

Exceptions:

std::invalid_argument if mean or stdv length does not match channels.

Usage Example:

Transforme tf;
tf.normalize_({0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});
void resize_(size_t H, size_t W);
Description: Adds a resize transformation to the pipeline. Currently a placeholder; creates a tensor of shape {C, H, W}.

Parameters:

H → Target height.

W → Target width.

Usage Example:
tf.resize_(224, 224);
Tensor astype(DType new_dtype) const;
Description: Returns a new tensor with elements converted to the specified DType. Original tensor is unchanged.

Parameters:

new_dtype → Target data type.

Return: Tensor → New tensor of type new_dtype.

Usage Example:
Tensor y = x.astype(DType::Float32);
void to_(DType new_dtype);
Description: Converts the tensor in-place to the given DType. Updates data and gradient memory.

Parameters:

new_dtype → Target data type.

Exceptions:

std::bad_alloc if memory allocation fails.

Usage Example:
x.to_(DType::Float32);
Tensor operator()(const Tensor& input) const;
Description: Applies all transformations in the pipeline sequentially to the input tensor.

Parameters:

input → Tensor to transform.

Return: Tensor → Result after applying all transformations.

Usage Example:
Tensor output = tf(input_tensor);

Usage Pattern
Transforme tf;
tf.normalize_({0.5f, 0.5f, 0.5f}, {0.5f, 0.5f, 0.5f});
tf.resize_(128, 128);

Tensor input = load_tensor("image_data");
Tensor output = tf(input); // Applies normalization then resize