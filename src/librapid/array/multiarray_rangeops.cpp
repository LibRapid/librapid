#include <librapid/config.hpp>
#include <librapid/autocast/autocast.hpp>
#include <librapid/array/ops.hpp>
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
		if (end == std::numeric_limits<double>::infinity()) {
			end	  = start;
			start = 0;
		}

		Array res(
		  librapid::Extent({(int64_t)ceil((end - start) / inc)}), dtype, locn);
		ops::FillLinear<double> op(start, inc);
		Array::applyBinaryOp(res, res, res, op, false, false);
		return res;
	}

	Array linear(double start, double end, int64_t num,
				 const std::string &dtype, const Accelerator &locn) {
		return linear(start, end, num, stringToDatatype(dtype), locn);
	}

	Array linear(double start, double end, int64_t num, const Datatype &dtype,
				 const std::string &locn) {
		return linear(start, end, num, dtype, stringToAccelerator(locn));
	}

	Array linear(double start, double end, int64_t num,
				 const std::string &dtype, const std::string &locn) {
		return linear(
		  start, end, num, stringToDatatype(dtype), stringToAccelerator(locn));
	}

	Array range(double start, double end, double inc, const std::string &dtype,
				const Accelerator &locn) {
		return range(start, end, inc, stringToDatatype(dtype), locn);
	}

	Array range(double start, double end, double inc, const Datatype &dtype,
				const std::string &locn) {
		return range(start, end, inc, dtype, stringToAccelerator(locn));
	}

	Array range(double start, double end, double inc, const std::string &dtype,
				const std::string &locn) {
		return range(
		  start, end, inc, stringToDatatype(dtype), stringToAccelerator(locn));
	}
} // namespace librapid
