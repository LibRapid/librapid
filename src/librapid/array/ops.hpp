#ifndef LIBRAPID_OPS
#define LIBRAPID_OPS

#include <librapid/config.hpp>
#include <librapid/math/rapid_math.hpp>
#include <librapid/array/extent.hpp>
#include <librapid/array/stride.hpp>
#include <librapid/autocast/autocast.hpp>
#include <functional>

namespace librapid
{
	namespace ops
	{
		struct TestOp
		{
			const char *name = "add";
			const char *kernel = R"V0G0N(
					b = a * 2;
				)V0G0N";

			template<typename A>
			inline auto operator()(A a) const
			{
				return a * 2;
			}
		};

		struct Add
		{
			const char *name = "add";
			const char *kernel = R"V0G0N(
					c = a + b;
				)V0G0N";

			template<typename A, typename B>
			inline auto operator()(A a, B b) const
			{
				return a + b;
			}
		};

		struct Sub
		{
			const char *name = "sub";
			const char *kernel = R"V0G0N(
					c = a - b;
				)V0G0N";

			template<typename A, typename B>
			inline auto operator()(A a, B b) const
			{
				return a - b;
			}
		};

		struct Mul
		{
			const char *name = "mul";
			const char *kernel = R"V0G0N(
					c = a * b;
				)V0G0N";

			template<typename A, typename B>
			inline auto operator()(A a, B b) const
			{
				return a * b;
			}
		};

		struct Div
		{
			const char *name = "div";
			const char *kernel = R"V0G0N(
					c = a / b;
				)V0G0N";

			template<typename A, typename B>
			inline auto operator()(A a, B b) const
			{
				return a / b;
			}
		};
	}
}

#endif // LIBRAPID_OPS