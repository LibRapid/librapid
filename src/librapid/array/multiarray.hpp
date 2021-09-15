#ifndef LIBRAPID_ARRAY
#define LIBRAPID_ARRAY

#include <librapid/config.hpp>
#include <librapid/math/rapid_math.hpp>
#include <librapid/array/extent.hpp>
#include <librapid/array/stride.hpp>
#include <librapid/autocast/autocast.hpp>
#include <librapid/array/multiarray_operations.hpp>
#include <librapid/array/ops.hpp>

#include <atomic>
#include <functional>

#ifdef LIBRAPID_REFCHECK
#define increment() _increment(__LINE__)
#define decrement() _decrement(__LINE__)
#endif // LIBRAPID_REFCHECK

namespace librapid
{
	class Array
	{
	public:
		/**
		 * \rst
		 *
		 * Default constructor for the Array type. It does not initialize any values
		 * and many functions will throw an error when given an empty array.
		 *
		 * \endrst
		 */
		Array();

		/**
		 * \rst
		 *
		 * Create a new Array from an Extent and an optional Datatype and
		 * Accelerator. The Extent defines the number of dimensions of the Array,
		 * as well as the size of each dimension.
		 *
		 * String values can also be passed as input parameters for the datatype
		 * and accelerator.
		 *
		 * The datatype can be constructed in many ways. For example, all of "i32",
		 * "int32" and "int" represent a 32-bit signed integer.
		 *
		 * All possible names are listed below. A green label means that name is
		 * safe to use. Yellow represents a value which should be avoided if
		 * possible, and red means the value is strongly advised not to be used.
		 *
		 * *Note: These warnings are more for readability than anything else*
		 *
		 * .. panels::
		 *		:container: container pb-4
		 *		:column: col-lg-6 col-md-6 col-sm-6 col-xs-12 p-2
		 *
		 *		None Type
		 *
		 *		:badge:`n, badge-danger`
		 *		:badge:`none, badge-success`
		 *		:badge:`null, badge-warning`
		 *		:badge:`void, badge-warning`
		 *
		 *		---
		 *
		 *		Boolean
		 *
		 *		:badge:`b, badge-danger`
		 *		:badge:`bool, badge-success`
		 *		:badge:`boolean, badge-success`
		 *
		 *		---
		 *
		 *		Signed 8-bit integer
		 *
		 *		:badge:`i8, badge-success`
		 *		:badge:`int8, badge-success`
		 *		:badge:`short, badge-warning`
		 *
		 *		---
		 *
		 *		Unsigned 8-bit integer
		 *
		 *		:badge:`ui8, badge-success`
		 *		:badge:`uint8, badge-success`
		 *		:badge:`unsigned short, badge-warning`
		 *
		 * 		---
		 *
		 *		Signed 16-bit integer
		 *
		 *		:badge:`i16, badge-success`
		 *		:badge:`int16, badge-success`
		 *		:badge:`int, badge-warning`
		 *
		 * 		---
		 *
		 *		Unsigned 16-bit integer
		 *
		 *		:badge:`ui16, badge-success`
		 *		:badge:`uint16, badge-success`
		 *		:badge:`unsigned int, badge-warning`
		 *
		 * 		---
		 *
		 *		Signed 32-bit integer
		 *
		 *		:badge:`i32, badge-success`
		 *		:badge:`int32, badge-success`
		 *		:badge:`long, badge-warning`
		 *
		 * 		---
		 *
		 *		Unsigned 32-bit integer
		 *
		 *		:badge:`ui32, badge-success`
		 *		:badge:`uint32, badge-success`
		 *		:badge:`unsigned long, badge-warning`
		 *
		 * 		---
		 *
		 *		Signed 64-bit integer
		 *
		 *		:badge:`i, badge-danger`
		 *		:badge:`i64, badge-success`
		 *		:badge:`int64, badge-success`
		 *		:badge:`long long, badge-warning`
		 *
		 * 		---
		 *
		 *		Unsigned 64-bit integer
		 *
		 *		:badge:`ui, badge-danger`
		 *		:badge:`ui64, badge-success`
		 *		:badge:`uint64, badge-success`
		 *		:badge:`unsigned long long, badge-warning`
		 *
		 * 		---
		 *
		 *		32-bit floating point
		 *
		 *		:badge:`f32, badge-success`
		 *		:badge:`float32, badge-success`
		 *		:badge:`float, badge-warning`
		 *
		 * 		---
		 *
		 *		64-bit floating point
		 *
		 *		:badge:`f, badge-danger`
		 *		:badge:`f64, badge-success`
		 *		:badge:`float64, badge-success`
		 *		:badge:`double, badge-warning`
		 *
		 * 		---
		 *
		 *		Complex 32-bit floating point
		 *
		 *		:badge:`cf32, badge-success`
		 *		:badge:`cfloat32, badge-success`
		 *		:badge:`complex float, badge-warning`
		 *
		 * 		---
		 *
		 *		Complex 64-bit floating point
		 *
		 *		:badge:`c, badge-danger`
		 *		:badge:`cf64, badge-success`
		 *		:badge:`cfloat64, badge-success`
		 *		:badge:`complex double, badge-warning`
		 *
		 * The accelerator value must be "CPU" or "GPU".
		 *
		 * Parameters
		 * ----------
		 *
		 * extent: ``Extent``
		 *		The dimensions for the Array
		 * dtype: ``Datatype = FLOAT64``
		 *		The datatype for the Array
		 * location: ``Accelerator = CPU``
		 *		Where the Array will be stored. GPU is only allowed if CUDA support
		 *		is enabled at compiletime
		 *
		 * \endrst
		 */
		Array(const Extent &extent, Datatype dtype = Datatype::FLOAT64,
			  Accelerator location = Accelerator::CPU);

		inline Array(const Extent &extent, std::string dtype = "float64",
					 Accelerator location = Accelerator::CPU)
			: Array(extent, stringToDatatype(dtype), location)
		{}

		inline Array(const Extent &extent, Datatype dtype = Datatype::FLOAT64,
					 std::string accelerator = "cpu")
			: Array(extent, dtype, stringToAccelerator(accelerator))
		{}

		inline Array(const Extent &extent, std::string dtype = "float64",
					 std::string accelerator = "cpu")
			: Array(extent, stringToDatatype(dtype),
					stringToAccelerator(accelerator))
		{}

		/**
		 * \rst
		 *
		 * Create an Array object from an existing one. This constructor copies all
		 * values, and the new Array shares the data of the Array passed to it, so
		 * an update in one will result in an update in the other.
		 *
		 * Note that if the input Array is not initialized, the function will
		 * quick return and not initialize the new Array.
		 *
		 * .. Hint::
		 *
		 *		If you want to create an exact copy of an Array, but don't want the
		 *		data to be linked, see the ``Array::copy()`` function.
		 *
		 * Parameters
		 * ----------
		 *
		 * other: ``Array``
		 *		The Array instance to construct from
		 *
		 * \endrst
		 */
		Array(const Array &other);

		/**
		 * \rst
		 *
		 * Set one Array equal to a value.
		 *
		 * If this Array on is invalid (i.e. it was created using the default
		 * constructor), the array will be initialized and the relevant data
		 * will be copied into it.
		 *
		 * If the left-hand-side of the operation is another Array instance, the
		 * data from that array will be copied into this array. If the arrays are
		 * identical in terms of their Extent, the data will be copied, otherwise
		 * this array will be recreated with the correct size.
		 *
		 * .. Attention::
		 *		There is a single exception to this, which occurs when this array is
		 *		a direct subscript of another (e.g. ``myArray[0]``). If this is the
		 *		case, the left-hand-side of this operation *must* have the same
		 *		extent, otherwise an error will be thrown
		 *
		 * \endrst
		 */
		Array &operator=(const Array &other);
		Array &operator=(bool val);
		Array &operator=(int8_t val);
		Array &operator=(uint8_t val);
		Array &operator=(int16_t val);
		Array &operator=(uint16_t val);
		Array &operator=(int32_t val);
		Array &operator=(uint32_t val);
		Array &operator=(int64_t val);
		Array &operator=(uint64_t val);
		Array &operator=(float val);
		Array &operator=(double val);
		Array &operator=(const Complex<float> &val);
		Array &operator=(const Complex<double> &val);

		~Array();

		/**
		 * \rst
		 *
		 * Return the number of dimensions of the Array
		 *
		 * \endrst
		 */
		inline size_t ndim() const
		{
			return m_extent.ndim();
		}

		/**
		 * \rst
		 *
		 * Return the Extent of the Array
		 *
		 * \endrst
		 */
		inline Extent extent() const
		{
			return m_extent;
		}

		/**
		 * \rst
		 *
		 * Return the Stride of the Array
		 *
		 * \endrst
		 */
		inline Stride stride() const
		{
			return m_stride;
		}

		/**
		 * \rst
		 *
		 * Return the datatype of the Array
		 *
		 * \endrst
		 */
		inline Datatype dtype() const
		{
			return m_dtype;
		}

		/**
		 * \rst
		 *
		 * Return the accelerator of the Array
		 *
		 * \endrst
		 */
		inline Accelerator location() const
		{
			return m_location;
		}

		/**
		 * \rst
		 *
		 * Return a sub-array or scalar value at a particular index in the Array. If
		 * the index is below zero or is greater than the size of the first
		 * dimension of the Array, an exception will be thrown
		 *
		 * \endrst
		 */
		inline const Array operator[](lr_int index) const
		{
			return subscript(index);
		}

		inline Array operator[](lr_int index)
		{
			using nonConst = typename std::remove_const<Array>::type;
			return (nonConst) subscript(index);
		}

		/**
		 * \rst
		 *
		 * Fill the every element of the Array with a particular value
		 *
		 * \endrst
		 */
		void fill(double val);
		void fill(const Complex<double> &val);

		Array operator+(const Array &other) const;
		void add(const Array &other, Array &res) const;

		Array operator-(const Array &other) const;

		void transpose(const Extent &order = Extent());

		inline std::string str(size_t indent = 0, bool showCommas = false) const
		{
			static int64_t tmpRows, tmpCols;
			return str(indent, showCommas, tmpRows, tmpCols);
		}

		std::string str(size_t indent, bool showCommas,
						int64_t &printedRows, int64_t &printedCols) const;

	private:
	#ifdef LIBRAPID_REFCHECK
		inline void _increment(int line) const
		{
			if (m_references == nullptr)
				return;

			(*m_references)++;

			std::cout << "Incrementing at line " << line << ". References is now "
				<< *m_references << "\n";
		}
	#else
		inline void initializeCudaStream() const
		{
		#ifdef LIBRAPID_HAS_CUDA
		#ifdef LIBRAPID_CUDA_STREAM
			if (!streamCreated)
			{
				checkCudaErrors(cudaStreamCreateWithFlags(&cudaStream,
								cudaStreamNonBlocking));
				streamCreated = true;
			}
		#endif // LIBRAPID_CUDA_STREAM
		#endif // LIBRAPID_HAS_CUDA
		}

		inline void increment() const
		{
			if (m_references == nullptr)
				return;

			(*m_references)++;
		}
	#endif // LIBRAPID_REFCHECK

	#ifdef LIBRAPID_REFCHECK
		inline void _decrement(int line)
		{
			if (m_references == nullptr)
				return;

			(*m_references)--;

			if (*m_references == 0)
			{
				std::cout << "Freeing data at line " << line << "\n";

				// Delete data
				AUTOCAST_FREE({m_dataOrigin, m_dtype, m_location});
				delete m_references;
			}
			else
			{
				printf("Decrementing at line %i. References is now %i\n", line,
					   (int) *m_references);

				std::cout << "Decrementing at line " << line
					<< ". References is now " << *m_references << "\n";
			}
		}
	#else
		inline void decrement()
		{
			if (m_references == nullptr)
				return;

			(*m_references)--;

			if (*m_references == 0)
			{
				// Delete data
				AUTOCAST_FREE({m_dataOrigin, m_dtype, m_location});
				delete m_references;
			}
		}
	#endif // LIBRAPID_REFCHECK

		inline VoidPtr makeVoidPtr() const
		{
			return {m_dataStart, m_dtype, m_location};
		}

		void constructNew(const Extent &e, const Stride &s,
						  const Datatype &dtype,
						  const Accelerator &location);

		void constructHollow(const Extent &e, const Stride &s,
							 const Datatype &dtype, const Accelerator &location);

		const Array subscript(size_t index) const;

		template<typename A, typename B, typename C, class FUNC>
		static void simpleCPUop(librapid::Accelerator locnA,
								librapid::Accelerator locnB,
								librapid::Accelerator locnC,
								const A *a, const B *b, C *c, size_t size,
								const FUNC &op, const std::string &name);

		template<typename A, typename B, typename C>
		static void simpleFill(librapid::Accelerator locnA,
							   librapid::Accelerator locnB,
							   A *data, B *, size_t size,
							   C val);

		std::pair<lr_int, lr_int> stringifyFormatPreprocess(bool stripMiddle,
															bool autoStrip) const;

		std::string stringify(lr_int indent, bool showCommas,
							  bool stripMiddle, bool autoStrip,
							  std::pair<lr_int, lr_int> &longest,
							  int64_t &printedRows, int64_t &printedCols) const;

		// private:
	public:
		Accelerator m_location = Accelerator::CPU;
		Datatype m_dtype = Datatype::NONE;

		void *m_dataStart = nullptr;
		void *m_dataOrigin = nullptr;

		// std::atomic to allow for multithreading, because multiple threads may
		// increment/decrement at the same clock cycle, resulting in values being
		// incorrect and errors turning up.
		std::atomic<size_t> *m_references = nullptr;

		Extent m_extent;
		Stride m_stride;

		bool m_isScalar = false; // Array is a scalar value
		bool m_isChild = false; // Array is a direct subscript of another (e.g. x[0])
	};
}

#ifdef LIBRAPID_REFCHECK
#undef increment
#undef decrement
#endif // LIBRAPID_REFCHECK

#endif // LIBRAPID_ARRAY