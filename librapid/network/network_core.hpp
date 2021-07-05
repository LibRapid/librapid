#ifndef LIBRAPID_NETWORK_CORE
#define LIBRAPID_NETWORK_CORE

#include <librapid/config.hpp>
#include <librapid/ndarray/ndarray.hpp>

namespace librapid
{
	template<typename T>
	struct network_config
	{
		lr_int inputs = 0;
		std::vector<lr_int> hidden;
		lr_int outputs = 0;

		std::vector<std::string> activations;
		std::vector<std::string> optimizers;
		std::vector<T> learning_rates;
	};

	struct train_config
	{
		lr_int batch_size;
		lr_int epochs;

		// A trivial constructor with default values
		train_config(lr_int batch = -1, lr_int epoch = -1) : batch_size(batch), epochs(epoch)
		{}
	};

	template<typename T = double,
		typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
		class network
	{
	public:
		network() = default;

		network(const network_config<T> &config)
		{
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
		train_config m_train_config;

		std::vector<layers::basic_layer<T> *> m_layers;

		std::mt19937 m_random_generator = std::mt19937();
	};
}

#endif // LIBRAPID_NETWORK_CORE