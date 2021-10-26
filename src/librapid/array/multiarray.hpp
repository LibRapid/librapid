#ifndef LIBRAPID_ARRAY
#define LIBRAPID_ARRAY

#include <atomic>
#include <functional>
#include <variant>

#include <librapid/config.hpp>
#include <librapid/math/rapid_math.hpp>
#include <librapid/array/extent.hpp>
#include <librapid/array/stride.hpp>
#include <librapid/autocast/autocast.hpp>
#include <librapid/array/multiarray_operations.hpp>
#include <librapid/array/ops.hpp>

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
		 *		Signed 64-bit integer
		 *
		 *		:badge:`i, badge-danger`
		 *		:badge:`int, badge-warning`
		 *		:badge:`i64, badge-success`
		 *		:badge:`int64, badge-success`
		 *		:badge:`long long, badge-warning`
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

		inline Array(const Extent &extent, std::string dtype,
					 Accelerator location = Accelerator::CPU)
			: Array(extent, stringToDatatype(dtype), location)
		{}

		inline Array(const Extent &extent, Datatype dtype,
					 std::string accelerator = "cpu")
			: Array(extent, dtype, stringToAccelerator(accelerator))
		{}

		inline Array(const Extent &extent, std::string dtype,
					 std::string accelerator)
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
		 * Create an array from a scalar value. The array will be created on host
		 * memory (even if CUDA is enabled) and will be stored as a zero-dimensional
		 * Array.
		 *
		 * \endrst
		 */
		Array(bool val);
		Array(float val);
		Array(double val);

		template<typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
		inline Array(T val)
		{
			initializeCudaStream();

			constructNew(Extent(1), Stride(1), Datatype::INT64, Accelerator::CPU);
			m_isScalar = true;
			std::visit([&](auto *data)
			{
				*data = val;
			}, m_dataStart);
		}

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

		template<typename T, typename std::enable_if<std::is_integral<T>::value, int>::type = 0>
		inline Array operator=(T val)
		{
			if (m_isChild && !m_isScalar)
				throw std::invalid_argument("Cannot set an array with more than zero"
											" dimensions to a scalar value. Array must"
											" have zero dimensions (i.e. scalar)");
			if (!m_isChild)
			{
				if (m_references != nullptr) decrement();
				constructNew(Extent(1), Stride(1), Datatype::INT64, Accelerator::CPU);
			}

			auto raw = createRaw();
			int64_t tmp = val;
			rawArrayMemcpy(raw, RawArray{&tmp, Datatype::INT64, Accelerator::CPU}, 1);

			m_isScalar = true;
			return *this;
		}

		Array &operator=(float val);
		Array &operator=(double val);

		~Array();

		/**
		 * \rst
		 *
		 * Return the number of dimensions of the Array
		 *
		 * \endrst
		 */
		inline int64_t ndim() const
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
		 * For C++ use only -- returns a VoidPtr object containing the memory
		 * location of the Array's data, its datatype and its location
		 *
		 * \endrst
		 */
		RawArray createRaw() const;

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

		const Array subscript(int64_t index) const;

		/**
		 * \rst
		 *
		 * Return a sub-array or scalar value at a particular index in the Array. If
		 * the index is below zero or is greater than the size of the first
		 * dimension of the Array, an exception will be thrown
		 *
		 * \endrst
		 */
		inline const Array operator[](int64_t index) const
		{
			return subscript(index);
		}

		inline Array operator[](int64_t index)
		{
			// using nonConst = typename std::remove_const<Array>::type;
			// return static_cast<nonConst>(subscript(index));
			return subscript(index);
		}

		/**
		 * \rst
		 *
		 * Fill the every element of the Array with a particular value
		 *
		 * \endrst
		 */
		void fill(double val);

		Array copy(const Datatype &dtype = Datatype::NONE,
				   const Accelerator &locn = Accelerator::NONE);

		Array operator+(const Array &other) const;
		Array operator-(const Array &other) const;
		Array operator*(const Array &other) const;
		Array operator/(const Array &other) const;

		void transpose(const Extent &order = Extent());

		inline std::string str(int64_t indent = 0, bool showCommas = false) const
		{
			static int64_t tmpRows, tmpCols;
			return str(indent, showCommas, tmpRows, tmpCols);
		}

		std::string str(int64_t indent, bool showCommas,
						int64_t &printedRows, int64_t &printedCols) const;

		template<typename FUNC>
		static inline void applyUnaryOp(Array &dst, const Array &src,
										const FUNC &operation)
		{
			// Operate on one array and store the result in another array

			if (dst.m_references == nullptr || dst.m_extent != src.m_extent)
			{
				throw std::invalid_argument("Cannot operate on array with "
											+ src.m_extent.str()
											+ " and store the result in "
											+ dst.m_extent.str());
			}

			auto srcPtr = src.createRaw();
			auto dstPtr = dst.createRaw();
			auto size = src.m_extent.size();

			if (src.m_stride.isTrivial() && src.m_stride.isContiguous())
			{
				// Trivial
				imp::multiarrayUnaryOpTrivial(dstPtr, srcPtr, size, operation);
			}
			else
			{
				// Not trivial, so use advanced method
				imp::multiarrayUnaryOpComplex(dstPtr, srcPtr, size, dst.m_extent,
											  dst.m_stride, src.m_stride, operation);
			}

			dst.m_isScalar = src.m_isScalar;
		}

		template<typename FUNC>
		static inline Array applyUnaryOp(Array &src, const FUNC &operation)
		{
			// Operate on one array and store the result in another array

			if (src.m_references == nullptr)
			{
				throw std::invalid_argument("Cannot operate on an "
											"uninitialized array");
			}

			Array dst(src.m_extent, src.m_dtype, src.m_location);
			auto srcPtr = src.createRaw();
			auto dstPtr = dst.createRaw();
			auto size = src.m_extent.size();

			if (src.m_stride.isTrivial() && src.m_stride.isContiguous())
			{
				// Trivial
				imp::multiarrayUnaryOpTrivial(dstPtr, srcPtr, size, operation);
			}
			else
			{
				// Not trivial, so use advanced method
				imp::multiarrayUnaryOpComplex(dstPtr, srcPtr, size, dst.m_extent,
											  dst.m_stride, src.m_stride, operation);
			}

			dst.m_isScalar = src.m_isScalar;

			return dst;
		}

		template<class FUNC>
		static inline void applyBinaryOp(Array &dst, const Array &srcA,
										 const Array &srcB,
										 const FUNC &operation)
		{
			// Operate on two arrays and store the result in another array

			if (!srcA.m_isScalar && !srcB.m_isScalar && srcA.m_extent != srcB.m_extent)
				throw std::invalid_argument("Cannot operate on two arrays with "
											+ srcA.m_extent.str() + " and "
											+ srcA.m_extent.str());

			if (dst.m_references == nullptr || dst.m_extent != srcA.m_extent)
				throw std::invalid_argument("Cannot operate on two arrays with "
											+ srcA.m_extent.str()
											+ " and store the result in "
											+ dst.m_extent.str());

			auto ptrSrcA = srcA.createRaw();
			auto ptrSrcB = srcB.createRaw();
			auto ptrDst = dst.createRaw();
			auto size = dst.m_extent.size();

			if ((srcA.m_stride.isTrivial() && srcA.m_stride.isContiguous() &&
				srcB.m_stride.isTrivial() && srcB.m_stride.isContiguous()) ||
				(srcA.m_stride == srcB.m_stride))
			{
				// Trivial
				imp::multiarrayBinaryOpTrivial(ptrDst, ptrSrcA, ptrSrcB,
											   srcA.m_isScalar, srcB.m_isScalar,
											   size, operation);

				// Update the result stride too
				dst.m_stride = srcA.m_isScalar ? srcB.m_stride : srcA.m_stride;
			}
			else
			{
				// Not trivial, so use advanced method
				imp::multiarrayBinaryOpComplex(ptrDst, ptrSrcA, ptrSrcB,
											   srcA.m_isScalar, srcB.m_isScalar,
											   size, dst.m_extent, dst.m_stride,
											   srcA.m_stride, srcB.m_stride,
											   operation);
			}

			if (srcA.m_isScalar && srcB.m_isScalar)
				dst.m_isScalar = true;
		}

		template<class FUNC>
		static inline Array applyBinaryOp(const Array &srcA,
										  const Array &srcB,
										  const FUNC &operation)
		{
			// Operate on two arrays and store the result in another array

			if (!(srcA.m_isScalar || srcB.m_isScalar) &&
				srcA.m_extent != srcB.m_extent)
				throw std::invalid_argument("Cannot operate on two arrays with "
											+ srcA.m_extent.str() + " and "
											+ srcB.m_extent.str());

			Accelerator newLoc = max(srcA.m_location, srcB.m_location);
			Datatype newType = max(srcA.m_dtype, srcB.m_dtype);

			Array dst(srcA.m_isScalar ? srcB.m_extent : srcA.m_extent, newType, newLoc);

			auto ptrSrcA = srcA.createRaw();
			auto ptrSrcB = srcB.createRaw();
			auto ptrDst = dst.createRaw();
			auto size = dst.m_extent.size();

			if ((srcA.m_stride.isTrivial() && srcA.m_stride.isContiguous() &&
				srcB.m_stride.isTrivial() && srcB.m_stride.isContiguous()) ||
				(srcA.m_stride == srcB.m_stride))
			{
				// Trivial
				imp::multiarrayBinaryOpTrivial(ptrDst, ptrSrcA, ptrSrcB,
											   srcA.m_isScalar, srcB.m_isScalar,
											   size, operation);

				// Update the result stride too
				dst.m_stride = srcA.m_isScalar ? srcB.m_stride : srcA.m_stride;
			}
			else
			{
				// Not trivial, so use advanced method
				imp::multiarrayBinaryOpComplex(ptrDst, ptrSrcA, ptrSrcB,
											   srcA.m_isScalar, srcB.m_isScalar,
											   size, dst.m_extent, dst.m_stride,
											   srcA.m_stride, srcB.m_stride,
											   operation);
			}

			if (srcA.m_isScalar && srcB.m_isScalar)
				dst.m_isScalar = true;

			return dst;
		}

	private:
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

			++(*m_references);
		}

		inline void decrement()
		{
			if (m_references == nullptr)
				return;

			--(*m_references);

			if (*m_references == 0)
			{
				// Delete data
				freeRawArray(createRaw());
				delete m_references;
			}
		}

		void constructNew(const Extent &e, const Stride &s,
						  const Datatype &dtype,
						  const Accelerator &location);

		void constructHollow(const Extent &e, const Stride &s,
							 const Datatype &dtype, const Accelerator &location);

		std::pair<int64_t, int64_t> stringifyFormatPreprocess(bool stripMiddle,
															  bool autoStrip) const;

		std::string stringify(int64_t indent, bool showCommas,
							  bool stripMiddle, bool autoStrip,
							  std::pair<int64_t, int64_t> &longest,
							  int64_t &printedRows, int64_t &printedCols) const;

	private:
		Accelerator m_location = Accelerator::CPU;
		Datatype m_dtype = Datatype::NONE;

		RawArrayData m_dataStart;
		RawArrayData m_dataOrigin;

		// std::atomic to allow for multithreading, because multiple threads may
		// increment/decrement at the same clock cycle, resulting in values being
		// incorrect and errors turning up all over the place
		std::atomic<int64_t> *m_references = nullptr;

		Extent m_extent;
		Stride m_stride;

		bool m_isScalar = false; // Array is a scalar value
		bool m_isChild = false; // Array is a direct subscript of another (e.g. x[0])
	};

	void add(const Array &a, const Array &b, Array &res);
	void sub(const Array &a, const Array &b, Array &res);
	void mul(const Array &a, const Array &b, Array &res);
	void div(const Array &a, const Array &b, Array &res);

	Array add(const Array &a, const Array &b);
	Array sub(const Array &a, const Array &b);
	Array mul(const Array &a, const Array &b);
	Array div(const Array &a, const Array &b);

	template<typename T>
	inline Array operator+(T lhs, const Array &rhs)
	{
		return Array::applyBinaryOp(lhs, rhs, ops::Add());
	}

	template<typename T>
	inline Array operator-(T lhs, const Array &rhs)
	{
		return Array::applyBinaryOp(lhs, rhs, ops::Sub());
	}

	template<typename T>
	inline Array operator*(T lhs, const Array &rhs)
	{
		return Array::applyBinaryOp(lhs, rhs, ops::Mul());
	}

	template<typename T>
	inline Array operator/(T lhs, const Array &rhs)
	{
		return Array::applyBinaryOp(lhs, rhs, ops::Div());
	}
}

#endif // LIBRAPID_ARRAY