#ifndef LIBRAPID_LAYER_BASE
#define LIBRAPID_LAYER_BASE

#include <librapid/config.hpp>
#include <librapid/ndarray/ndarray.hpp>
#include <librapid/network/activations.hpp>
#include <librapid/network/optimizers.hpp>

namespace librapid
{
	namespace layers
	{
		template<typename T = double>
		class basic_layer
		{
		public:
			virtual ~basic_layer()
			{}

			inline virtual void compile(basic_layer<T> *prevLayer) = 0;
			inline virtual bool check(basic_layer<T> *other) = 0;

			inline virtual lr_int get_nodes() const = 0;
			inline virtual optimizers::basic_optimizer<T> *get_optimizer() const = 0;
			inline virtual activations::basic_activation<T> *get_activation() const = 0;

			inline virtual basic_ndarray<T> forward(const basic_ndarray<T> &x) = 0;
			inline virtual basic_ndarray<T> backpropagate(const basic_ndarray<T> &error) = 0;

			inline virtual basic_ndarray<T> get_prev_output() const = 0;

		private:
			std::string m_type = "none";
			basic_ndarray<T> m_prev_output;
		};

		template<typename T = double>
		class input : public basic_layer<T>
		{
		public:
			input(lr_int nodes) : m_nodes(nodes), m_type("input")
			{}

			~input() = default;

			inline void compile(basic_layer<T> *prev_layer) override
			{
				m_prev_output = basic_ndarray<T>(extent({m_nodes, 1ll}));
			}

			inline bool check(basic_layer<T> *other) override
			{
				return this == other;
			}

			inline basic_ndarray<T> forward(const basic_ndarray<T> &x) override
			{
				m_prev_output = x;
				return m_prev_output;
			}

			inline basic_ndarray<T> backpropagate(const basic_ndarray<T> &error) override
			{
				return error;
			}

			inline lr_int get_nodes() const override
			{
				return m_nodes;
			}

			inline optimizers::basic_optimizer<T> *get_optimizer() const override
			{
				return nullptr;
			}

			inline basic_ndarray<T> get_prev_output() const override
			{
				return m_prev_output;
			}

			inline activations::basic_activation<T> *get_activation() const override
			{
				return nullptr;
			}

		private:
			std::string m_type;
			lr_int m_nodes;
			basic_ndarray<T> m_prev_output;
		};
	}
}

#endif // LIBRAPID_LAYER_BASE