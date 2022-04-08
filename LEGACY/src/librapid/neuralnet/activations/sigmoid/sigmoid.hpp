#ifndef LIBRAPID_ACTIVATION_SIGMOID
#define LIBRAPID_ACTIVATION_SIGMOID

#include <librapid/config.hpp>
#include <librapid/array/ops.hpp>
#include <librapid/neuralnet/activations/activation.hpp>
#include <librapid/array/multiarray.hpp>

namespace librapid {
	/**
	 * \rst
	 *
	 * The sigmoid activation function
	 *
	 * .. Math::
	 *		\text{Activation: }
	 *		f(x)=\frac{1}{1 + e^{-x}}
	 *
	 *		\text{Derivative: }
	 *		\frac{df}{dx}f(x)=x\times(1-x)
	 *
	 *		\text{Weights: }
	 *		\text{Random distribution in range }\frac{\pm
	 *1}{\sqrt{\text{previous_nodes}}}
	 *
	 * \endrst
	 */
	class Sigmoid : public Activation {
	public:
		void construct(int64_t prevNodes) override;
		Array f(const Array &input) override;
		Array df(const Array &input) override;
		[[nodiscard]] Array weight(const Extent &e) const override;

	private:
		int64_t m_prevNodes;
	};
} // namespace librapid

#endif // LIBRAPID_ACTIVATION_SIGMOID