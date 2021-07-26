#ifndef LIBRAPID_NETWORK_CORE
#define LIBRAPID_NETWORK_CORE

#include <map>
#include <unordered_map>

#include <librapid/config.hpp>
#include <librapid/ndarray/ndarray.hpp>

using SV_ = std::vector<std::string>;
using D_ = std::unordered_map<std::string, lr_int>;

namespace librapid
{
	template<typename T = double, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
	struct config_container
	{
		bool is_real = false;
		bool is_string = false;
		bool is_real_vector = false;
		bool is_str_vector = false;
		bool is_dict = false;
		bool is_array = false;

		T real = 0;
		std::string str;
		std::unordered_map<std::string, lr_int> dict;
		std::vector<T> real_vec;
		std::vector<std::string> str_vec;
		basic_ndarray<T> arr;

		config_container() = default;

		// 		config_container(const std::string &title, T val)
		// 			: name(title), real(val), is_real(true)
		// 		{}
		//
		// 		config_container(const std::string &title, const std::string &val)
		// 			: name(title), str(val), is_string(true)
		// 		{}
		//
		// 		config_container(const std::string &title, const std::vector<T> &val)
		// 			: name(title), real_vec(std::vector<T>(val.begin(), val.end())), is_real_vector(true)
		// 		{}
		//
		// 		config_container(const std::string &title, const std::initializer_list<T> &val)
		// 			: name(title), real_vec(std::vector<T>(val.begin(), val.end())), is_real_vector(true)
		// 		{}
		//
		// 		config_container(const std::string &title, const std::vector<std::string> &val)
		// 			: name(title), str_vec(std::vector<std::string>(val.begin(), val.end())), is_str_vector(true)
		// 		{}
		//
		// 		config_container(const std::string &title, const std::initializer_list<std::string> &val)
		// 			: name(title), str_vec(std::vector<std::string>(val.begin(), val.end())), is_str_vector(true)
		// 		{}
		//
		// 		config_container(const std::string &title, const std::unordered_map<std::string, lr_int> &val)
		// 			: name(title), dict(val), is_dict(true)
		// 		{}
		//
		// 		config_container(const std::string &title, const basic_ndarray<T> &val)
		// 			: name(title), arr(val), is_array(true)
		// 		{}

		config_container(T val)
			: real(val), is_real(true)
		{}

		config_container(const std::string &val)
			: str(val), is_string(true)
		{}

		config_container(const char *val)
			: str(val), is_string(true)
		{}

		config_container(const std::vector<T> &val)
			: real_vec(std::vector<T>(val.begin(), val.end())), is_real_vector(true)
		{}

		config_container(const std::initializer_list<T> &val)
			: real_vec(std::vector<T>(val.begin(), val.end())), is_real_vector(true)
		{}

		config_container(const std::vector<std::string> &val)
			: str_vec(std::vector<std::string>(val.begin(), val.end())), is_str_vector(true)
		{}

		config_container(const std::initializer_list<std::string> &val)
			: str_vec(std::vector<std::string>(val.begin(), val.end())), is_str_vector(true)
		{}

		config_container(const std::unordered_map<std::string, lr_int> &val)
			: dict(val), is_dict(true)
		{}

		config_container(const basic_ndarray<T> &val)
			: arr(val), is_array(true)
		{}
	};

	// template<typename T = double>
	// using network_config = std::vector<config_container<T>>;
	template<typename T = double>
	using network_config = std::unordered_map<std::string, config_container<T>>;

	template<typename T = double>
	using named_param = std::unordered_map<std::string, basic_ndarray<T>>;

	template<typename T = double,
		typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
		class network
	{
	public:
		network()
		{
			m_reference_count = new std::atomic<lr_int>(1);
		}

		/**
		 * \rst
		 *
		 * Create a new neural network from a given ``network_config``.
		 * The ``network_config`` object should contain all of the
		 * information needed to construct a neural network, such as the
		 * numbers and sizes of layers, for example.
		 *
		 * .. Attention::
		 *		When using the C++ library, it may be necessary to specify the
		 *		type of some values, otherwise an error might be raised at runtime.
		 *		The datatypes this impacts are listed below:
		 *
		 *		- `std::vector<std::string>`
		 *		- `std::unordered_map<std::string, integer>`
		 *
		 *		To shorten the naming of datatypes, the following aliases are
		 *		provided:
		 *
		 *		- `std::vector<std::string> = SV_`
		 *		- `std::unordered_map<std::string, lr_int>` = `D_`
		 *
		 * Parameters
		 * ----------
		 *
		 * input: integer, unordered_map<string, integer>, dict
		 *		Represents the inputs to the neural network.
		 *
		 *		If only an integer is passed in as input, the value is
		 *		treated as the number of nodes in the input layer of the
		 *		network.
		 *
		 *		If the input is a ``map<string, integer>``, the inputs
		 *		are assumed to be named. For every element in the map, the
		 *		structure :math:`\textbf{must}` be ``name of input, nodes of input``, and
		 *		there can be any number of these pairs of values. The name
		 *		of the input can be used to simplify neural network
		 *		predictions and training by providing human-readable inputs
		 *		with their corresponding values. The number of nodes each
		 *		input represents can be adjusted to support individual
		 *		values, one-hot vectors, or any other form of input.
		 *
		 *		When making predictions with the neural network or training
		 *		it, the input must be either a single array of values or
		 *		a ``map<string, array>``, where the first element of each
		 *		value passed is the name of the input it represents, and the
		 *		second is the input itself, which should be a vector that can
		 *		be broadcast to the number of nodes specified earlier. If a
		 *		single array of values is passed, the number of elements must
		 *		equal the total number of nodes in the input layer, and the
		 *		values will be assigned to inputs in the order they were
		 *		initially specified.
		 *
		 * output: integer, unordered_map<string, integer>, dict
		 *		Represents the outputs of the neural network. The functionality
		 *		of the output parameter is nearly identical to that of the
		 *		input parameter in terms of functionality, except it refers
		 *		to the outputs of the network rather than the inputs (obviously...)
		 *
		 * hidden: vector<integer>
		 *		A list of values containing the number of nodes for each of the
		 *		hidden layers of the neural network.
		 *
		 *		For example, passing in ``hidden = {3, 4, 2}`` will create a
		 *		neural network with three hidden layers, with 3, 4 and 2 nodes
		 *		respectively.
		 *
		 * optimizer: string, vector<string> (optional)
		 *		A list of optimizers to use for the neural network. If no optimizers
		 *		are passed, the default ``sgd`` optimizer will be selected for the
		 *		entire network. If a single optimizer is provided, the entire network
		 *		will use the specified optimizer. If more are provided, there must be
		 *		an optimizer for every hidden layer and the output layer, otherwise an
		 *		error will be thrown. Valid optimizers are:
		 *
		 *		- ``sgd`` (Stochastic Gradient Descent)
		 *		- ``sgd momentum`` (Stochastic Gradient Descent *with Momentum*)
		 *		- ``rmsprop`` (Root Mean Square Propagation)
		 *		- ``adam`` (Adaptive Moment Estimation)
		 *
		 * activation: string, vector<string> (optional)
		 *		A list of activations for the neural network. If no activations
		 *		are passed, the neural network will default to using the ``sigmoid``
		 *		activation. If only one activation is passed, then that activation
		 *		will be used on all of the layers of the network. If more than one
		 *		activation is passed, the number of activations must equal
		 *		``the number of hidden layers + 1``, otherwise an error will be thrown.
		 *		(This is to provide an activation for each hidden layer and the output
		 *		layer).
		 *
		 *		Valid activations are:
		 *
		 *		- ``sigmoid``
		 *		- ``tanh``
		 *		- ``relu``
		 *		- ``leaky_relu``
		 *
		 * learning rate: vector<real> (optional)
		 *		A list of learning rates for the neural network. If no inputs are given,
		 *		each layer will set it's learning rate to the default for the optimizer
		 *		on that layer. The defaults are shown below. If only one input is given,
		 *		it will represent the learning rate for all layers of the network. If
		 *		more values are provided, there must be one value per hidden layer, and
		 *		another for the output layer, otherwise an error will occur.
		 *
		 *		Default Learning Rates:
		 *		-----------------------
		 *		- ``sgd = 0.01``
		 *		- ``sgd momentum = 0.01``
		 *		- ``rmsprop = 0.01``
		 *		- ``adam = 0.001``
		 *
		 *
		 * \endrst
		 */
		network(const network_config<T> &config)
		{
			m_reference_count = new std::atomic<lr_int>(1);

			int found_input = 0;
			int found_output = 0;
			int found_hidden = 0;
			int found_activations = 0;
			int found_optimizers = 0;
			int found_learning_rates = 0;

			bool use_named_inputs = false;
			bool use_named_outputs = false;
			lr_int input_nodes = 0;
			std::vector<lr_int> hidden_nodes;
			lr_int output_nodes = 0;
			std::unordered_map<std::string, lr_int> input_names;
			std::unordered_map<std::string, lr_int> output_names;

			std::vector<std::string> activations;
			std::vector<std::string> optimizers;
			std::vector<T> learning_rates;

			// Parse the input information and store it
			for (const auto &param : config)
			{
				const std::string &key = param.first;
				const config_container<T> &value = param.second;

				if (key == "input")
				{
					found_input++; // Increment (not bool to allow for error checking)

					if (value.is_real)
					{
						// No need to use named inputs
						use_named_inputs = false;
						input_nodes = value.real; // Convert from real to integer
					}
					else if (value.is_dict)
					{
						// Use named inputs
						use_named_inputs = true;

						input_nodes = 0;
						for (const auto &io_pair : value.dict)
							input_nodes += io_pair.second;

						m_config["input_names"] = value.dict;
					}
					else
					{
						throw std::invalid_argument("The 'input' parameter requires "
													"an integer or an unordered_map/dict");
					}

					// Set "input_nodes" in the config
					m_config["input_nodes"] = input_nodes;
				}
				else if (key == "output")
				{
					found_output++; // Increment (not bool to allow for error checking)

					if (value.is_real)
					{
						// No need to use named outputs
						use_named_outputs = false;
						output_nodes = value.real; // Convert from real to integer
					}
					else if (value.is_dict)
					{
						// Use named outputs
						use_named_outputs = true;

						output_nodes = 0;
						for (const auto &io_pair : value.dict)
							output_nodes += io_pair.second;

						m_config["output_names"] = value.dict;
					}
					else
					{
						throw std::invalid_argument("The 'output' parameter requires "
													"an integer or an unordered_map/dict");
					}

					m_config["output_nodes"] = output_nodes;
				}
				else if (key == "hidden")
				{
					found_hidden++;

					if (value.is_real_vector)
						for (const auto &val : value.real_vec)
							hidden_nodes.emplace_back(val);
					else
						throw std::invalid_argument("The 'hidden' parameter requires a vector/list of integers");

					m_config["hidden_nodes"] = std::vector<T>(hidden_nodes.begin(),
															  hidden_nodes.end());
				}
				else if (key == "activation")
				{
					found_activations++;

					if (value.is_string)
						activations = std::vector<std::string>({value.str});
					else if (value.is_str_vector)
						activations = value.str_vec;
					else
						throw std::invalid_argument("The 'activations' parameter requires a string or a vector/list of strings");

					m_config["activation_names"] = activations;
				}
				else if (key == "optimizer")
				{
					found_optimizers++;

					if (value.is_string)
						optimizers = std::vector<std::string>({value.str});
					else if (value.is_str_vector)
						optimizers = value.str_vec;
					else
						throw std::invalid_argument("The 'optimizers' parameter requires a string or a vector/list of strings");

					m_config["optimizer_names"] = optimizers;
				}
				else if (key == "learning rate")
				{
					found_learning_rates++;

					if (value.is_real)
						learning_rates = std::vector<T>({value.real});
					else if (value.is_real_vector)
						learning_rates = value.real_vec;
					else
						throw std::invalid_argument("The 'learning rates' parameter requires a real or list of reals");

					m_config["learning_rates"] = learning_rates;
				}
				else
				{
					throw std::invalid_argument("Parameter '" + key + "' is invalid");
				}
			}

			// Deal with the parsed information

			std::vector<lr_int> shape;
			shape.emplace_back(input_nodes);
			for (const lr_int val : hidden_nodes) shape.emplace_back(val);
			shape.emplace_back(output_nodes);

			if (found_input != 1)
			{
				throw std::invalid_argument("Only 1 'input' parameter is allowed, but "
											+ std::to_string(found_input) + " were found");
			}
			else
			{
				// Add the input layer
				add_layer(new layers::input<T>(input_nodes));

				if (use_named_inputs)
					m_input = input_names;
			}

			if (found_hidden != 1)
			{
				throw std::invalid_argument("Only 1 'hidden' parameter is allowed, but "
											+ std::to_string(found_hidden) + " were found");
			}
			else
			{
				for (size_t i = 1; i < shape.size() - 1; i++)
				{
					// Add the hidden layer

					layers::basic_layer<T> *layer = generate_layer(i,
																   shape,
																   learning_rates,
																   activations,
																   optimizers);

					add_layer(layer);
				}
			}

			if (found_output != 1)
			{
				throw std::invalid_argument("Only 1 'output' parameter is allowed, but "
											+ std::to_string(found_output) + " were found");
			}
			else
			{
				layers::basic_layer<T> *layer = generate_layer(shape.size() - 1,
															   shape,
															   learning_rates,
															   activations,
															   optimizers);

				add_layer(layer);
			}
		}

	#if LIBRAPID_BUILD == 1
		LR_INLINE network(py::dict args)
		{
			network_config<python_dtype> config;
			for (auto arg : args)
			{
				std::string key_name;
				config_container<T> container;

				auto key = arg.first;
				auto value = arg.second;
				std::string key_type = py::repr(key.get_type());
				std::string value_type = py::repr(value.get_type());

				if (key_type != "<class 'str'>")
					throw std::invalid_argument("Parameter '" + std::string(py::repr(key))
												+ "' is invalid. Expected type 'str', received '"
												+ key_type + "'");

				if (value_type == "<class 'int'>" || value_type == "<class 'float'>")
				{
					container = py::cast<T>(value);
				}
				else if (value_type == "<class 'str'>")
				{
					container = py::cast<std::string>(value);
				}
				else if (value_type == "<class 'list'>")
				{
					// Convert to std::vector with correct datatype
					py::list new_val = py::cast<py::list>(value);
					if (new_val.size() == 0)
						throw std::invalid_argument("LibRapid cannot process an empty list");

					if (std::string(py::repr(new_val.get_type())) == "<class 'str'>")
						container = py::cast<std::vector<std::string>>(value);
					else
						container = py::cast<std::vector<python_dtype>>(value);
				}
				else if (value_type == "<class 'dict'>")
				{
					container = py::cast<std::unordered_map<std::string, lr_int>>(value);
				}
				else if (value_type == "<class 'librapid_.ndarray'>")
				{
					container = py::cast<basic_ndarray<python_dtype>>(value);
				}
				else
				{
					throw std::invalid_argument("Corresponding value of parameter '"
												+ std::string(py::repr(key))
												+ "' is invalid. Type '"
												+ value_type + "' is invalid");
				}

				key_name = py::cast<std::string>(key);
				config[key_name] = container;
			}

			m_reference_count = new std::atomic<lr_int>(1);
			*this = network<T>(config);
		}
	#endif

		network(const network<T> &other)
		{
			decrement();
			m_layers.clear();

			m_input = other.m_input;
			m_output = other.m_output;

			m_is_compiled = other.m_is_compiled;
			m_has_config = other.m_has_config;

			m_config = other.m_config;
			m_train_config = other.m_train_config;

			m_has_named_inputs = other.m_has_named_inputs;
			m_has_named_outputs = other.m_has_named_outputs;

			m_reference_count = other.m_reference_count;

			increment();
		}

		LR_INLINE network<T> &operator=(const network<T> &other)
		{
			decrement();

			m_input = other.m_input;
			m_output = other.m_output;

			m_layers = other.m_layers;

			m_is_compiled = other.m_is_compiled;
			m_has_config = other.m_has_config;

			m_config = other.m_config;
			m_train_config = other.m_train_config;

			m_has_named_inputs = other.m_has_named_inputs;
			m_has_named_outputs = other.m_has_named_outputs;

			m_reference_count = other.m_reference_count;

			increment();

			return *this;
		}

		~network()
		{
			decrement();
		}

		LR_INLINE void add_layer(layers::basic_layer<T> *layer)
		{
			if (m_is_compiled)
				throw std::runtime_error("Cannot add layers to a compiled network");

			m_layers.emplace_back(layer);
		}

		LR_INLINE void add_layers(const std::vector<layers::basic_layer<T> *> &layers)
		{
			if (m_is_compiled)
				throw std::runtime_error("Cannot add layers to a compiled network");

			for (auto *layer : layers)
				m_layers.emplace_back(layer);
		}

		LR_INLINE void compile()
		{
			// Check the layers are all unique
			for (size_t i = 0; i < m_layers.size(); i++)
			{
				for (size_t j = 0; j < m_layers.size(); j++)
				{
					if (i != j && m_layers[i] == m_layers[j])
					{
						throw std::logic_error("Layers " + std::to_string(i) + " and "
											   + std::to_string(j) + " share the same memory"
											   " location and therefore the same data, so training"
											   " the neural network will result in incorrect"
											   "results. All layers must be unique");
					}
				}
			}

			m_layers[0]->compile(nullptr);
			for (size_t i = 1; i < m_layers.size(); i++)
			{
				m_layers[i]->compile(m_layers[i - 1]);
			}

			m_is_compiled = true;
		}

		LR_INLINE basic_ndarray<T> forward(const basic_ndarray<T> &input)
		{
			if (!m_is_compiled)
				throw std::runtime_error("Cannot run forward feed on a network "
										 "that has not yet been compiled. Please "
										 "see the documentation for more information.");

			// Check that the input is a valid shape
			auto fixed = fix_array(input, m_config["input_nodes"].real);

			return internal_forward_feed(fixed);
		}

		// LR_INLINE named_param<T> forward(const named_param<T> &input)
		LR_INLINE basic_ndarray<T> forward(const named_param<T> &input)
		{
			if (!m_is_compiled)
				throw std::runtime_error("Cannot run forward feed on a network "
										 "that has not yet been compiled. Please "
										 "see the documentation for more information.");

			auto input_array = array_from_named(input, m_config["input_names"]);
			return internal_forward_feed(input_array);

			// TODO: NEEDS ARRAY SLICING
			// auto res = internal_forward_feed(input_array);
			// return named_from_array(res, m_config["output_names"]);
		}

		LR_INLINE basic_ndarray<T> backpropagate(const basic_ndarray<T> &input,
												 const basic_ndarray<T> &target)
		{
			if (!m_is_compiled)
				throw std::runtime_error("Cannot backpropagate on a network "
										 "that has not yet been compiled. Please "
										 "see the documentation for more information.");

			// Check that the input is a valid shape
			auto fixed_input = fix_array(input, m_config["input_nodes"].real);
			auto fixed_target = fix_array(target, m_config["output_nodes"].real);

			return internal_backpropagate(fixed_input, fixed_target);
		}

		LR_INLINE basic_ndarray<T> backpropagate(const named_param<T> &input,
												 const named_param<T> &target)
		{
			if (!m_is_compiled)
				throw std::runtime_error("Cannot backpropagate on a network "
										 "that has not yet been compiled. Please "
										 "see the documentation for more information.");

			auto input_array = array_from_named(input, m_config["input_names"]);
			auto target_array = array_from_named(target, m_config["output_names"]);

			return internal_backpropagate(input_array, target_array);
		}

	private:
		LR_INLINE void increment() const
		{
			(*m_reference_count)++;
		}

		LR_INLINE void decrement()
		{
			(*m_reference_count)--;

			if ((*m_reference_count) == 0)
			{
				for (auto &layer : m_layers)
					delete layer;
				delete m_reference_count;
			}
		}

		LR_INLINE layers::basic_layer<T> *generate_layer(lr_int index,
														 const std::vector<lr_int> &shape,
														 const std::vector<T> &learning_rates,
														 const std::vector<std::string> &activations,
														 const std::vector<std::string> &optimizers) const
		{
			// Process the learning rate
			T lr = -1;

			if (learning_rates.size() == 1)
				lr = learning_rates[0];
			else if (learning_rates.size() > 1)
				if (learning_rates.size() != shape.size() - 1)
					throw std::invalid_argument("Expected " + std::to_string(shape.size() - 1)
												+ " learning rate parameters, but received "
												+ std::to_string(learning_rates.size()));
				else
					lr = learning_rates[index - 1];

			// Process the activation
			std::string activation_name = "sigmoid";

			if (activations.size() == 1)
				activation_name = activations[0];
			else if (activations.size() > 1)
				activation_name = activations[index - 1];

			activations::basic_activation<T> *activation = nullptr;

			if (activation_name == "sigmoid") activation = new activations::sigmoid<T>();
			else if (activation_name == "tanh") activation = new activations::tanh<T>();
			else if (activation_name == "relu") activation = new activations::relu<T>();
			else if (activation_name == "leaky relu") activation = new activations::leaky_relu<T>();
			else throw std::invalid_argument("Activation '" + activation_name + "' is invalid. "
											 "See documentation for valid arguments");

			// Process the optimizer
			std::string optimizer_name = "sgd";
			optimizers::basic_optimizer<T> *optimizer;

			if (optimizers.size() == 1)
				optimizer_name = optimizers[0];
			else if (optimizers.size() > 1)
				if (optimizers.size() != shape.size() - 1)
					throw std::invalid_argument("Expected " + std::to_string(shape.size() - 1)
												+ " optimizer parameters, but received "
												+ std::to_string(optimizers.size()));
				else
					optimizer_name = optimizers[index - 1];

			if (lr == -1)
			{
				if (optimizer_name == "sgd") optimizer = new optimizers::sgd<T>();
				else if (optimizer_name == "sgd momentum") optimizer = new optimizers::sgd_momentum<T>();
				else if (optimizer_name == "rmsprop") optimizer = new optimizers::rmsprop<T>();
				else if (optimizer_name == "adam") optimizer = new optimizers::adam<T>();
				else throw std::invalid_argument("Optimizer '" + optimizer_name + "' is invalid. "
												 "See documentation for valid arguments");
			}
			else
			{
				if (optimizer_name == "sgd") optimizer = new optimizers::sgd<T>(lr);
				else if (optimizer_name == "sgd momentum") optimizer = new optimizers::sgd_momentum<T>(lr);
				else if (optimizer_name == "rmsprop") optimizer = new optimizers::rmsprop<T>(lr);
				else if (optimizer_name == "adam") optimizer = new optimizers::adam<T>(lr);
				else throw std::invalid_argument("Optimizer '" + optimizer_name + "' is invalid. "
												 "See documentation for valid arguments");
			}

			// Create the final layer
			layers::basic_layer<T> *layer = nullptr;

			// TODO: Other layer types???
			layer = new layers::affine<T>(shape[index], activation, optimizer);

			return layer;
		}

		LR_INLINE basic_ndarray<T> fix_array(const basic_ndarray<T> &arr, lr_int target_nodes) const
		{
			if (arr.ndim() == 1)
			{
				if (arr.get_extent()[0] != target_nodes)
					goto invalid;

				return arr.reshaped({AUTO, 1ll});
			}

			if (arr.ndim() == 2)
			{
				if (arr.get_extent()[0] == target_nodes && arr.get_extent()[1] == 1)
					return arr;

				if (arr.get_extent()[0] == 1 && arr.get_extent()[1] == target_nodes)
					return arr.transposed();

				goto invalid;
			}

		invalid:
			throw std::domain_error("An array with " + arr.get_extent().str()
									+ " cannot be broadcast to an array with extent("
									+ std::to_string(target_nodes) + ", 1).");
		}

		LR_INLINE basic_ndarray<T> array_from_named(const named_param<T> &data,
													const config_container<T> &names) const
		{
			// Convert a list of named data into a single array with the
			// correct extent

			// Calculate the number of nodes needed
			std::vector<basic_ndarray<T>> fixed;
			for (const auto &input : names.dict)
			{
				if (data.find(input.first) != data.end())
				{
					// Found the key
					fixed.emplace_back(fix_array(data.at(input.first), input.second));
				}
				else
				{
					// Did not find the key
					throw std::invalid_argument("Could not find key '" + input.first +
												" in input parameters. All keys are required");
				}
			}

			if (data.size() != names.dict.size())
				throw std::invalid_argument("Additional keys were found in the input array that "
											"do not exist in the neural network");

			return concatenate(fixed, 0);
		}

		// TODO: NEEDS ARRAY SLICING
		// LR_INLINE basic_ndarray<T> named_from_array(const basic_ndarray<T> &arr,
		// 											const config_container<T> &names) const
		// {
		// 	// Convert an array into a dictionary of named values and the corresponding output
		// 	// values.
		// 
		// 	// Calculate the number of nodes needed
		// 	std::vector<basic_ndarray<T>> fixed;
		// 	for (const auto &input : names.dict)
		// 	{
		// 		if (data.find(input.first) != data.end())
		// 		{
		// 			// Found the key
		// 			fixed.emplace_back(fix_array(data.at(input.first), input.second));
		// 		}
		// 		else
		// 		{
		// 			// Did not find the key
		// 			throw std::invalid_argument("Could not find key '" + input.first +
		// 										" in input parameters. All keys are required");
		// 		}
		// 	}
		// 
		// 	if (data.size() != names.dict.size())
		// 		throw std::invalid_argument("Additional keys were found in the input array that "
		// 									"do not exist in the neural network");
		// 
		// 	return concatenate(fixed, 0);
		// }

		LR_INLINE basic_ndarray<T> internal_forward_feed(const basic_ndarray<T> &input) const
		{
			m_layers[0]->forward(input);
			for (size_t i = 1; i < m_layers.size(); i++)
				m_layers[i]->forward(m_layers[i - 1]->get_prev_output());
			return m_layers[m_layers.size() - 1]->get_prev_output();
		}

		LR_INLINE basic_ndarray<T> internal_backpropagate(const basic_ndarray<T> &input,
														  const basic_ndarray<T> &target)
		{
			auto output = internal_forward_feed(input);
			auto loss = target - output;

			for (lr_int i = (lr_int) m_layers.size() - 1; i >= 0; i--)
				loss.set_to(m_layers[i]->backpropagate(loss));

			return target - output;
		}

	private:
		std::atomic<lr_int> *m_reference_count = nullptr;

		bool m_is_compiled = false;
		bool m_has_config = false;

		network_config<T> m_config;
		network_config<T> m_train_config;

		std::vector<layers::basic_layer<T> *> m_layers;

		// Named input information
		bool m_has_named_inputs = false;
		bool m_has_named_outputs = false;

		std::unordered_map<std::string, lr_int> m_input;
		std::unordered_map<std::string, lr_int> m_output;

		std::mt19937 m_random_generator = std::mt19937();
	};
}

#endif // LIBRAPID_NETWORK_CORE