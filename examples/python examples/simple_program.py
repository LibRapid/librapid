import librapid as libr

# Create a new vector with 5 elements, all of which are 0

# Note: at the time of writing, libr.extent(5) (note, not a list)
# doesn't work, but I've just fixed it, so it'll be updated
# in the next version (0.0.11)
my_vector = libr.ndarray(libr.extent([4]), 0)
print(my_vector)  # Print it out!
# [0. 0. 0. 0.]

# Fill it with values:
for i in range(my_vector.extent[0]):
    my_vector[i] = i + 1

print(my_vector)
# [1. 2. 3. 4.]

# Create a 2x2 matrix from my_vector
my_matrix = my_vector.reshaped(2, 2)

print(my_matrix)
# [[1. 2.]
#  [3. 4.]]

# Transpose the matrix
my_matrix_T = my_matrix.transposed()
print(my_matrix_T)
# [[1. 3.]
#  [2. 4.]]

# Reshape the transposed matrix to get a new vector
my_vector_2 = my_matrix_T.reshaped(4)
print(my_vector_2)
# [1. 3. 2. 4.]

# Add the two together
my_sum = my_vector + my_vector_2
print(my_sum)
# [2. 5. 5. 8.]
