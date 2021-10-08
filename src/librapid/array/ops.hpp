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
		struct Copy
		{
			std::string name = "copy";
			std::string kernel = R"V0G0N(
				return a;
			)V0G0N";

			template<typename A>
			inline auto operator()(A a) const
			{
				return a;
			}
		};

		struct Fill
		{
			std::string name = "fill";
			std::string kernel = R"V0G0N(
				return b;
			)V0G0N";

			template<typename A, typename B>
			inline auto operator()(A a, B b) const
			{
				return b;
			}
		};

		struct Add
		{
			std::string name = "add";
			std::string kernel = R"V0G0N(
					return a + b;
				)V0G0N";

			template<typename A, typename B>
			inline auto operator()(A a, B b) const
			{
				return a + b;
			}
		};

		struct Sub
		{
			std::string name = "sub";
			std::string kernel = R"V0G0N(
					return a - b;
				)V0G0N";

			template<typename A, typename B>
			inline auto operator()(A a, B b) const
			{
				return a - b;
			}
		};

		struct Mul
		{
			std::string name = "mul";
			std::string kernel = R"V0G0N(
					return a * b;
				)V0G0N";

			template<typename A, typename B>
			inline auto operator()(A a, B b) const
			{
				return a * b;
			}
		};

		struct Div
		{
			std::string name = "div";
			std::string kernel = R"V0G0N(
					return a / b;
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