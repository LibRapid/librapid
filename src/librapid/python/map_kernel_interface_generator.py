dtypes = [
	# "int64_t",
	# "float",
	"double",
	# "librapid::Complex<double>"
]

fstring = ".def_static(\"mapKernel\", [](const std::function<{}({})> &kernel, {}, librapid::Array &dst) {{ librapid::Array::mapKernel(kernel, {}, dst); }}, py::call_guard<py::gil_scoped_release>())"

maxInputs = 8

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
