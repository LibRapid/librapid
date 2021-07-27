#include <iostream>
#include <chrono>
#include <map>

// #define LIBRAPID_CBLAS

#include <librapid/librapid.hpp>

struct datatype
{
	const char *name;
	lr_int bytes;
	void *data;
	typename void *dtype;
};

struct datatype_int32
{
	const char *name = "int32";
	lr_int bytes = 4;
	int *data;
	typename int *dtype;
};

struct datatype_int64
{
	const char *name = "int64";
	lr_int bytes = 8;
	long long *data;
	typename long long *dtype;
};

class test
{
public:
	test(lr_int size, const char *dtype_name)
	{
		m_size = size;

		if (dtype_name == "int32")
			m_dtype = (datatype *) new datatype_int32{dtype_name, 4,
			(int *) malloc(sizeof(int) * size)};
		if (dtype_name == "int64")
			m_dtype = (datatype *) new datatype_int64{dtype_name, 8,
			(long long *) malloc(sizeof(long long) * size)};
	}

	test(const test &other)
	{
		free(m_dtype->data);
		delete m_dtype;

		m_size = other.m_size;

		if (other.m_dtype->name == "int32")
			m_dtype = (datatype *) new datatype_int32{other.m_dtype->name, 4,
			(int *) malloc(sizeof(int) * m_size)};
		if (other.m_dtype->name == "int64")
			m_dtype = (datatype *) new datatype_int64{other.m_dtype->name, 8,
			(long long *) malloc(sizeof(long long) * m_size)};
	}

	~test()
	{
		free(m_dtype->data);
		delete m_dtype;
	}

	void print_data(void *data, lr_int len) const
	{
		std::cout << "Cannot print data. Data was void\n";
	}

	void print_data(int *data, lr_int len) const
	{
		for (lr_int i = 0; i < len; i++)
			std::cout << data[i] << ", ";
		std::cout << "\n";
	}

	void print_data(long long *data, lr_int len) const
	{
		for (lr_int i = 0; i < len; i++)
			std::cout << data[i] << ", ";
		std::cout << "\n";
	}

	void thingy()
	{
		std::cout << "type name: " << m_dtype->name << "\n";
		print_data((m_dtype->dtype) m_dtype->data, m_size);
	}

private:

	lr_int m_size;
	datatype *m_dtype;
};

int main()
{
	test my_thing(8, "int32");
	my_thing.thingy();

	// 	auto top = librapid::range(1., 10.).reshaped({3, 3});
	// 	auto bottom = librapid::range(10., 19.).reshaped({3, 3});
	// 
	// 	std::cout << top << "\n\n\n";
	// 	std::cout << bottom << "\n\n\n";
	// 
	// 	try
	// 	{
	// 		std::cout << librapid::concatenate({top, bottom}, 0) << "\n\n\n\n";
	// 	}
	// 	catch (std::exception &e)
	// 	{
	// 		std::cout << "Failed\n";
	// 	}
	// 
	// 	try
	// 	{
	// 		std::cout << librapid::concatenate({top, bottom}, 1) << "\n\n\n\n";
	// 	}
	// 	catch (std::exception &e)
	// 	{
	// 		std::cout << "Failed\n";
	// 	}
	// 
	// 	try
	// 	{
	// 		std::cout << librapid::concatenate({top, bottom}, 2) << "\n\n\n\n";
	// 	}
	// 	catch (std::exception &e)
	// 	{
	// 		std::cout << "Failed\n";
	// 	}
	// 
	// 	try
	// 	{
	// 		std::cout << librapid::stack({top, bottom}, 2).str(0, true) << "\n";
	// 	}
	// 	catch (std::exception &e)
	// 	{
	// 		std::cout << "Error: " << e.what() << "\n";
	// 		return 1;
	// 	}
	// 
	// 	double start, end;
	// 
	// 	librapid::network_config<float> config = {
	// 	{"input", D_{{"x", 1}, {"y", 1}}},
	// 	{"hidden", {3}},
	// 	{"output", D_{{"o", 1}}},
	// 	{"learning rate", {0.05, 0.05}},
	// 	{"activation", {"leaky relu", "sigmoid"}},
	// 	{"optimizer", "adam"}
	// 	};
	// 
	// 	auto test_network = librapid::network(config);
	// 
	// 	auto test_network2 = librapid::network(config);
	// 
	// 	test_network = test_network2;
	// 
	// 	// // The training data input for XOR
	// 	// auto train_input = librapid::from_data(
	// 	// 	VEC<VEC<float>>{
	// 	// 			{0, 0},
	// 	// 			{0, 1},
	// 	// 			{1, 0},
	// 	// 			{1, 1}
	// 	// });
	// 
	// 	// Input is a vector << All inputs
	// 	// of vectors		 << Values for input on neural network
	// 	// of dictionaries	 << KEY ( name , value )
	// 
	// 	using input_type = std::unordered_map<std::string, librapid::basic_ndarray<float>>;
	// 
	// 	auto train_input = std::vector<input_type>{
	// 	{{"x", librapid::from_data(0.f)},
	// 	{"y", librapid::from_data(0.f)}},
	// 
	// 	{{"x", librapid::from_data(1.f)},
	// 	{"y", librapid::from_data(0.f)}},
	// 
	// 	{{"x", librapid::from_data(0.f)},
	// 	{"y", librapid::from_data(1.f)}},
	// 
	// 	{{"x", librapid::from_data(1.f)},
	// 	{"y", librapid::from_data(1.f)}}
	// 	};
	// 
	// 	// The labeled data (x XOR y)
	// 	auto train_output = std::vector<input_type>{
	// 		{{"o", librapid::from_data(0.f)}},
	// 
	// 		{{"o", librapid::from_data(1.f)}},
	// 
	// 		{{"o", librapid::from_data(1.f)}},
	// 
	// 		{{"o", librapid::from_data(0.f)}}
	// 	};
	// 
	// 	test_network.compile();
	// 
	// 	start = TIME;
	// 	for (lr_int i = 0; i < 3000 * 4; i++) // 3000 * 4 because there are 4 bits of data
	// 	{
	// 		lr_int index = librapid::math::random(0, 3);
	// 		test_network.backpropagate(train_input[index], train_output[index]);
	// 	}
	// 	end = TIME;
	// 	std::cout << "Time: " << end - start << "\n";
	// 
	// 	for (lr_int i = 0; i < 4; i++)
	// 	{
	// 		std::cout << "Input: " << train_input[i]["x"] << " ^ " << train_input[i]["y"] << "\n";
	// 		std::cout << "Output: " << test_network.forward(train_input[i]) << "\n";
	// 	}
	// 
	// 	return 0;
}
