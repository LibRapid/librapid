
// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "map_kernel_interface_generator.py" in the same directory     //
// ====================================================== //

.def_static(
  "mapKernel",
  [](

	const std::function<double(double)> &kernel,

	const librapid::Array &a1,
	librapid::Array

	  &dst) {
	  librapid::Array::mapKernel(kernel, a1, dst

	  );
  },
  py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel",
			  [](

				const librapid::GPUKernel &kernel,

				const librapid::Array &a1,
				librapid::Array

				  &dst) {
				  librapid::Array::mapKernel(kernel, a1, dst

				  );
			  })

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double)> &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(kernel, a1, a2, dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel",
			  [](

				const librapid::GPUKernel &kernel,

				const librapid::Array &a1,

				const librapid::Array &a2,
				librapid::Array

				  &dst) {
				  librapid::Array::mapKernel(kernel, a1, a2, dst

				  );
			  })

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double)> &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(kernel, a1, a2, a3, dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel",
			  [](

				const librapid::GPUKernel &kernel,

				const librapid::Array &a1,

				const librapid::Array &a2,

				const librapid::Array &a3,
				librapid::Array

				  &dst) {
				  librapid::Array::mapKernel(kernel, a1, a2, a3, dst

				  );
			  })

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double, double)> &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(kernel, a1, a2, a3, a4, dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel",
			  [](

				const librapid::GPUKernel &kernel,

				const librapid::Array &a1,

				const librapid::Array &a2,

				const librapid::Array &a3,

				const librapid::Array &a4,
				librapid::Array

				  &dst) {
				  librapid::Array::mapKernel(kernel, a1, a2, a3, a4, dst

				  );
			  })

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double, double, double)>
		&kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(kernel, a1, a2, a3, a4, a5, dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel",
			  [](

				const librapid::GPUKernel &kernel,

				const librapid::Array &a1,

				const librapid::Array &a2,

				const librapid::Array &a3,

				const librapid::Array &a4,

				const librapid::Array &a5,
				librapid::Array

				  &dst) {
				  librapid::Array::mapKernel(kernel, a1, a2, a3, a4, a5, dst

				  );
			  })

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double, double, double,
								 double)> &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,

	  const librapid::Array &a6,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(kernel, a1, a2, a3, a4, a5, a6, dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel",
			  [](

				const librapid::GPUKernel &kernel,

				const librapid::Array &a1,

				const librapid::Array &a2,

				const librapid::Array &a3,

				const librapid::Array &a4,

				const librapid::Array &a5,

				const librapid::Array &a6,
				librapid::Array

				  &dst) {
				  librapid::Array::mapKernel(kernel, a1, a2, a3, a4, a5, a6, dst

				  );
			  })

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double, double, double, double,
								 double)> &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,

	  const librapid::Array &a6,

	  const librapid::Array &a7,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(kernel, a1, a2, a3, a4, a5, a6, a7, dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel",
			  [](

				const librapid::GPUKernel &kernel,

				const librapid::Array &a1,

				const librapid::Array &a2,

				const librapid::Array &a3,

				const librapid::Array &a4,

				const librapid::Array &a5,

				const librapid::Array &a6,

				const librapid::Array &a7,
				librapid::Array

				  &dst) {
				  librapid::Array::mapKernel(
					kernel, a1, a2, a3, a4, a5, a6, a7, dst

				  );
			  })

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double, double, double, double,
								 double, double)> &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,

	  const librapid::Array &a6,

	  const librapid::Array &a7,

	  const librapid::Array &a8,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(kernel, a1, a2, a3, a4, a5, a6, a7, a8, dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel",
			  [](

				const librapid::GPUKernel &kernel,

				const librapid::Array &a1,

				const librapid::Array &a2,

				const librapid::Array &a3,

				const librapid::Array &a4,

				const librapid::Array &a5,

				const librapid::Array &a6,

				const librapid::Array &a7,

				const librapid::Array &a8,
				librapid::Array

				  &dst) {
				  librapid::Array::mapKernel(
					kernel, a1, a2, a3, a4, a5, a6, a7, a8, dst

				  );
			  })

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double, double, double, double,
								 double, double, double)> &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,

	  const librapid::Array &a6,

	  const librapid::Array &a7,

	  const librapid::Array &a8,

	  const librapid::Array &a9,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(
		  kernel, a1, a2, a3, a4, a5, a6, a7, a8, a9, dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel",
			  [](

				const librapid::GPUKernel &kernel,

				const librapid::Array &a1,

				const librapid::Array &a2,

				const librapid::Array &a3,

				const librapid::Array &a4,

				const librapid::Array &a5,

				const librapid::Array &a6,

				const librapid::Array &a7,

				const librapid::Array &a8,

				const librapid::Array &a9,
				librapid::Array

				  &dst) {
				  librapid::Array::mapKernel(
					kernel, a1, a2, a3, a4, a5, a6, a7, a8, a9, dst

				  );
			  })

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double, double, double, double,
								 double, double, double, double)> &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,

	  const librapid::Array &a6,

	  const librapid::Array &a7,

	  const librapid::Array &a8,

	  const librapid::Array &a9,

	  const librapid::Array &a10,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(
		  kernel, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel",
			  [](

				const librapid::GPUKernel &kernel,

				const librapid::Array &a1,

				const librapid::Array &a2,

				const librapid::Array &a3,

				const librapid::Array &a4,

				const librapid::Array &a5,

				const librapid::Array &a6,

				const librapid::Array &a7,

				const librapid::Array &a8,

				const librapid::Array &a9,

				const librapid::Array &a10,
				librapid::Array

				  &dst) {
				  librapid::Array::mapKernel(
					kernel, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, dst

				  );
			  })

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double, double, double, double,
								 double, double, double, double, double)>
		&kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,

	  const librapid::Array &a6,

	  const librapid::Array &a7,

	  const librapid::Array &a8,

	  const librapid::Array &a9,

	  const librapid::Array &a10,

	  const librapid::Array &a11,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(
		  kernel, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel",
			  [](

				const librapid::GPUKernel &kernel,

				const librapid::Array &a1,

				const librapid::Array &a2,

				const librapid::Array &a3,

				const librapid::Array &a4,

				const librapid::Array &a5,

				const librapid::Array &a6,

				const librapid::Array &a7,

				const librapid::Array &a8,

				const librapid::Array &a9,

				const librapid::Array &a10,

				const librapid::Array &a11,
				librapid::Array

				  &dst) {
				  librapid::Array::mapKernel(
					kernel, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, dst

				  );
			  })

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double, double, double, double,
								 double, double, double, double, double,
								 double)> &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,

	  const librapid::Array &a6,

	  const librapid::Array &a7,

	  const librapid::Array &a8,

	  const librapid::Array &a9,

	  const librapid::Array &a10,

	  const librapid::Array &a11,

	  const librapid::Array &a12,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(
		  kernel, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static(
	"mapKernel",
	[](

	  const librapid::GPUKernel &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,

	  const librapid::Array &a6,

	  const librapid::Array &a7,

	  const librapid::Array &a8,

	  const librapid::Array &a9,

	  const librapid::Array &a10,

	  const librapid::Array &a11,

	  const librapid::Array &a12,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(
		  kernel, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, dst

		);
	})

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double, double, double, double,
								 double, double, double, double, double, double,
								 double)> &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,

	  const librapid::Array &a6,

	  const librapid::Array &a7,

	  const librapid::Array &a8,

	  const librapid::Array &a9,

	  const librapid::Array &a10,

	  const librapid::Array &a11,

	  const librapid::Array &a12,

	  const librapid::Array &a13,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(
		  kernel, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static(
	"mapKernel",
	[](

	  const librapid::GPUKernel &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,

	  const librapid::Array &a6,

	  const librapid::Array &a7,

	  const librapid::Array &a8,

	  const librapid::Array &a9,

	  const librapid::Array &a10,

	  const librapid::Array &a11,

	  const librapid::Array &a12,

	  const librapid::Array &a13,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(
		  kernel, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, dst

		);
	})

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double, double, double, double,
								 double, double, double, double, double, double,
								 double, double)> &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,

	  const librapid::Array &a6,

	  const librapid::Array &a7,

	  const librapid::Array &a8,

	  const librapid::Array &a9,

	  const librapid::Array &a10,

	  const librapid::Array &a11,

	  const librapid::Array &a12,

	  const librapid::Array &a13,

	  const librapid::Array &a14,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(kernel,
								   a1,
								   a2,
								   a3,
								   a4,
								   a5,
								   a6,
								   a7,
								   a8,
								   a9,
								   a10,
								   a11,
								   a12,
								   a13,
								   a14,
								   dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel",
			  [](

				const librapid::GPUKernel &kernel,

				const librapid::Array &a1,

				const librapid::Array &a2,

				const librapid::Array &a3,

				const librapid::Array &a4,

				const librapid::Array &a5,

				const librapid::Array &a6,

				const librapid::Array &a7,

				const librapid::Array &a8,

				const librapid::Array &a9,

				const librapid::Array &a10,

				const librapid::Array &a11,

				const librapid::Array &a12,

				const librapid::Array &a13,

				const librapid::Array &a14,
				librapid::Array

				  &dst) {
				  librapid::Array::mapKernel(kernel,
											 a1,
											 a2,
											 a3,
											 a4,
											 a5,
											 a6,
											 a7,
											 a8,
											 a9,
											 a10,
											 a11,
											 a12,
											 a13,
											 a14,
											 dst

				  );
			  })

  .def_static(
	"mapKernel",
	[](

	  const std::function<double(double, double, double, double, double, double,
								 double, double, double, double, double, double,
								 double, double, double)> &kernel,

	  const librapid::Array &a1,

	  const librapid::Array &a2,

	  const librapid::Array &a3,

	  const librapid::Array &a4,

	  const librapid::Array &a5,

	  const librapid::Array &a6,

	  const librapid::Array &a7,

	  const librapid::Array &a8,

	  const librapid::Array &a9,

	  const librapid::Array &a10,

	  const librapid::Array &a11,

	  const librapid::Array &a12,

	  const librapid::Array &a13,

	  const librapid::Array &a14,

	  const librapid::Array &a15,
	  librapid::Array

		&dst) {
		librapid::Array::mapKernel(kernel,
								   a1,
								   a2,
								   a3,
								   a4,
								   a5,
								   a6,
								   a7,
								   a8,
								   a9,
								   a10,
								   a11,
								   a12,
								   a13,
								   a14,
								   a15,
								   dst

		);
	},
	py::call_guard<py::gil_scoped_release>())
  .def_static("mapKernel", [](

							 const librapid::GPUKernel &kernel,

							 const librapid::Array &a1,

							 const librapid::Array &a2,

							 const librapid::Array &a3,

							 const librapid::Array &a4,

							 const librapid::Array &a5,

							 const librapid::Array &a6,

							 const librapid::Array &a7,

							 const librapid::Array &a8,

							 const librapid::Array &a9,

							 const librapid::Array &a10,

							 const librapid::Array &a11,

							 const librapid::Array &a12,

							 const librapid::Array &a13,

							 const librapid::Array &a14,

							 const librapid::Array &a15,
							 librapid::Array

							   &dst) {
	  librapid::Array::mapKernel(kernel,
								 a1,
								 a2,
								 a3,
								 a4,
								 a5,
								 a6,
								 a7,
								 a8,
								 a9,
								 a10,
								 a11,
								 a12,
								 a13,
								 a14,
								 a15,
								 dst

	  );
  })
