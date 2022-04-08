#include <librapid/config.hpp>
#include <librapid/array/multiarray.hpp>
#include <librapid/array/math_utils.hpp>
#include <librapid/neuralnet/activations/sigmoid/sigmoid.hpp>

namespace librapid {
	void Sigmoid::construct(int64_t prevNodes) { m_prevNodes = prevNodes; }

	Array Sigmoid::f(const Array &input) { return 1 / (1 + exp(-input)); }
	Array Sigmoid::df(const Array &input) { return 0; }
	Array Sigmoid::weight(const Extent &input) const { return 0; }
} // namespace librapid
