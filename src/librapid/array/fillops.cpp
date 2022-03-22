#include <librapid/config.hpp>
#include <librapid/autocast/autocast.hpp>
#include <librapid/array/multiarray.hpp>

#include <cmath>

namespace librapid {
	Array linear(double start, double end, int64_t num, const Datatype &dtype,
				 const Accelerator &locn) {
		// Extract the correct datatype for the final array
		Datatype resType;
		if (dtype == Datatype::VALIDNONE)
			resType = Datatype::FLOAT64;
		else
			resType = dtype;

		Array res(librapid::Extent({num}), resType, locn);
		ops::FillLinear<double> op(start, (double)(end - start) / (double)num);
		Array::applyBinaryOp(res, res, res, op, false, false);
		return res;
	}

	Array range(double start, double end, double inc, const Datatype &dtype,
				const Accelerator &locn) {
		Array res(librapid::Extent({(int64_t) floor((end - start) / inc)}));
		ops::FillLinear<double> op(start, inc);
		Array::applyBinaryOp(res, res, res, op, false, false);
		return res;
	}
} // namespace librapid
