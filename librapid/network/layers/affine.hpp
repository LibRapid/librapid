#ifndef LIBRAPID_AFFINE
#define LIBRAPID_AFFINE

#include <librapid/config.hpp>
#include <librapid/ndarray/ndarray.hpp>
#include <librapid/network/activations.hpp>
#include <librapid/network/optimizers.hpp>
#include <librapid/network/layers/layer_base.hpp>

namespace librapid
{
	namespace layers
	{
		template<typename T = double>
		class affine : public basic_layer<T>
		{
		public:
			affine(lr_int nodes, activations::basic_activation<T> *activation,
				   optimizers::basic_optimizer<T> *optimizer)
				: m_nodes(nodes), m_activation(activation), m_optimizer(optimizer),
				m_type("affine")
			{}

			~affine()
			{
				// Free the optimizer and the activation
				// Don't free the previous layer, as that is freed by the network class
				delete m_optimizer;
				delete m_activation;
			}

			inline void compile(basic_layer<T> *prev_layer) override
			{
				m_prev_layer = prev_layer;

				// Construct the activation so we can use the correct weightings
				m_activation->construct(m_prev_layer->get_nodes());

				// Construct the network to be the correct shape
				m_weight = m_activation->weight(extent({m_nodes,
												m_prev_layer->get_nodes()}));
				m_bias = m_activation->weight(extent({m_nodes, 1ll}));
				m_prev_output = basic_ndarray<T>(extent({m_nodes, 1ll}));
			}

			inline bool check(basic_layer<T> *other) override
			{
				if (this == other)
					return true;

				if (m_optimizer == other->get_optimizer())
					return true;

				if (m_activation == other->get_activation())
					return true;

				return false;
			}

			inline lr_int get_nodes() const override
			{
				return m_nodes;
			}

			inline optimizers::basic_optimizer<T> *get_optimizer() const
			{
				return m_optimizer;
			}

			inline basic_ndarray<T> get_prev_output() const override
			{
				return m_prev_output;
			}

			inline activations::basic_activation<T> *get_activation() const override
			{
				return m_activation;
			}

			inline basic_ndarray<T> forward(const basic_ndarray<T> &x) override
			{
				if (x.get_extent()[0] != m_weight.get_extent()[1])
					throw std::domain_error("Cannot compute forward feed on data with "
											+ std::to_string(x.get_extent()[0])
											+ " nodes. Expected "
											+ std::to_string(m_weight.get_extent()[0])
											+ " nodes.");

				m_prev_output = m_activation->f(m_weight.dot(x) + m_bias);
				return m_prev_output;
			}

			inline basic_ndarray<T> backpropagate(const basic_ndarray<T> &error) override
			{
				// Calculate the weight gradient and adjust the weight
				// and bias for the layer accordingly. The weight update
				// is controlled by the optimizer, while the bias is
				// updated by adding the gradients

				auto gradient = m_activation->df(m_prev_output) * error;
				auto transposed = m_prev_layer->get_prev_output().transposed();
				auto dx = gradient.dot(transposed);

				m_weight = m_optimizer->apply(m_weight, dx);
				m_bias += gradient * m_optimizer->get_param("learning rate");

				// Return the error to be used by earlier layers
				return m_weight.transposed().dot(error);
			}

			// private:
		public:
			std::string m_type;
			lr_int m_nodes;

			basic_ndarray<T> m_weight;
			basic_ndarray<T> m_bias;
			basic_ndarray<T> m_prev_output;

			basic_layer<T> *m_prev_layer = nullptr;

			optimizers::basic_optimizer<T> *m_optimizer = nullptr;
			activations::basic_activation<T> *m_activation = nullptr;
		};
	}
}

#endif // LIBRAPID_AFFINE