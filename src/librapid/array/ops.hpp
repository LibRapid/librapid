#ifndef LIBRAPID_OPS
#define LIBRAPID_OPS

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
			auto operator()(A a) const
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
			auto operator()(A, B b) const
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
			auto operator()(A a, B b) const
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
			auto operator()(A a, B b) const
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
			auto operator()(A a, B b) const
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
			auto operator()(A a, B b) const
			{
				return a / b;
			}
		};
	}
}

#endif // LIBRAPID_OPS