dtypes = [
	# "int64_t",
	# "float",
	"double",
	# "librapid::Complex<double>"
]

maxInputs = 15

fstring = ".def_static(\"mapKernel\", [](const std::function<{}({})> &kernel, {}, librapid::Array &dst) {{ librapid::Array::mapKernel(kernel, {}, dst); }}, py::call_guard<py::gil_scoped_release>())"

print("Running")
with open("map_kernel_interface.hpp", "w") as f:
	f.write("""
// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "map_kernel_interface_generator.py" in the same directory     //
// ====================================================== //

""")

	for dtype in dtypes:
		print("Generating for", dtype)
		for i in range(1, maxInputs + 1):
			typelist = ", ".join([dtype] * i)
			
			arrlist = ""
			for j in range(i):
				arrlist += "const librapid::Array &a{}".format(j + 1)
				if j + 1 < i:
					arrlist += ", "

			varlist = ""
			for j in range(i):
				varlist += "a{}".format(j + 1)
				if j + 1 < i:
					varlist += ", "

			f.write(fstring.format(dtype, typelist, arrlist, varlist) + "\n")


fstring = """
	template<typename T, typename Kernel>
	struct ApplyKernelImpl<T, Kernel, {}> {{
		static inline void run(T **__restrict pointers, T *__restrict dst, const Kernel &kernel, uint64_t index) {{
			dst[index] = kernel({});
		}}
	}};
"""

print("Running")
with open("../array/mapKernelUtils.hpp", "w") as f:
	f.write("""
// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "map_kernel_interface_generator.py" in the same directory     //
// ====================================================== //

#pragma once

#include <librapid/config.hpp>
#include <cstdint>

namespace librapid::utils {
	template<typename T, typename Kernel, uint64_t dims>
	struct ApplyKernelImpl {
		static inline void run(T **__restrict pointers, T *__restrict dst, const Kernel &kernel, uint64_t index) {
			throw std::runtime_error("Too many arguments passed to Array.mapKernel -- Please see the documentation for details");
		}
	};

""")

	for i in range(1, maxInputs + 1):
		arglist = ", ".join(["pointers[{}][index]".format(ind) for ind in range(i)])

		f.write(fstring.format(i, arglist) + "\n")

	f.write("\n}")
