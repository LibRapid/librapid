#ifndef LIBRAPID_NETWORK_CORE
#define LIBRAPID_NETWORK_CORE

#include <map>

#include <librapid/config.hpp>
#include <librapid/ndarray/ndarray.hpp>

namespace librapid
{
	template<typename T = double, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
	struct config_container
	{
		bool is_real = false;
		bool is_string = false;
		bool is_vector = false;
		bool is_dict = false;
		bool is_array = false;

		std::string name;

		T real = 0;
		std::string str;
		std::map<std::string, lr_int> dict;
		std::vector<double> vec;
		basic_ndarray<T> arr;

		config_container(const std::string &title, T val)
			: name(title), real(val), is_real(true)
		{}

		config_container(const std::string &title, const std::string &val)
			: name(title), str(val), is_string(true)
		{}

		config_container(const std::string &title, const std::vector<double> &val)
			: name(title), vec(std::vector<double>(val.begin(), val.end())), is_vector(true)
		{}
		
		config_container(const std::string &title, const std::initializer_list<double> &val)
			: name(title), vec(std::vector<double>(val.begin(), val.end())), is_vector(true)
		{}

		config_container(const std::string &title, const std::map<std::string, lr_int> &val)
			: name(title), dict(val), is_dict(true)
		{}

		config_container(const std::string &title, const basic_ndarray<T> &val)
			: name(title), arr(val), is_array(true)
		{}
	};

	template<typename T = double>
	using network_config = std::vector<config_container<T>>;

	template<typename T = double,
		typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
		class network
	{
	public:
		network() = default;

		/**
		 * \rst
		 *
		 * Create a new neural network from a given ``network_config``.
		 * The ``network_config`` object should contain all of the
		 * information needed to construct a neural network, such as the
		 * numbers and sizes of layers, for example.
		 *
		 * Parameters
		 * ----------
		 *
		 * input: integer, map<string, integer>
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
		 * output: integer, map<string, integer>
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
		 * \endrst
		 */
		network(const network_config<T> &config)
		{
			bool use_named_inputs = false;
			bool use_named_outputs = false;
			std::vector<lr_int> shape;
			std::map<std::string, lr_int> input_names;
			std::map<std::string, lr_int> output_names;

			if (config.find("input") != config.end())
			{
				// Input parameter
				auto input = config["input"];

				if (input.is_real)
				{
					// No need to use named inputs
					lr_int nodes = input.real; // Convert from real to integer
					shape.emplace_back(nodes);
				}
				else if (input.is_dict)
				{
					// Use named inputs
					use_named_inputs = true;
					input_names = input.dict;
				}
			}
		}

		~network()
		{
			for (auto &layer : m_layers)
				delete layer;
		}

		LR_INLINE void add_layer(layers::basic_layer<T> *layer)
		{
			m_layers.emplace_back(layer);
		}

		LR_INLINE void add_layers(const std::vector<layers::basic_layer<T> *> &layers)
		{
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
			return internal_forward_feed(input);
		}

		LR_INLINE basic_ndarray<T> backpropagate(const basic_ndarray<T> &input,
												 const basic_ndarray<T> &target)
		{
			return internal_backpropagate(input, target);
		}

	private:
		LR_INLINE basic_ndarray<T> fix_array(const basic_ndarray<T> &arr, bool is_input) const
		{
			lr_int target_nodes = is_input ? m_config.inputs : m_config.outputs;

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
		bool m_is_compiled = false;
		bool m_has_config = false;

		network_config<T> m_config;
		network_config<T> m_train_config;

		std::vector<layers::basic_layer<T> *> m_layers;

		std::mt19937 m_random_generator = std::mt19937();
	};
}

#endif // LIBRAPID_NETWORK_CORE