#ifndef LIBRAPID_ARRAY
#define LIBRAPID_ARRAY

#include <librapid/config.hpp>
#include <librapid/math/rapid_math.hpp>
#include <librapid/array/extent.hpp>
#include <librapid/array/stride.hpp>
#include <librapid/autocast/autocast.hpp>
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
		 * Parameters
		 * ----------
		 *
		 * extent: ``Extent``
		 *		The dimensions for the Array
		 * dtype: ``Datatype = FLOAT32``
		 *		The datatype for the Array
		 * location: ``Accelerator = CPU``
		 *
		 * \endrst
		 */
		Array(const Extent &extent, Datatype dtype = Datatype::FLOAT32,
			  Accelerator location = Accelerator::CPU);

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

		Array &operator=(const Array &other);

		~Array();

		void fill(double val);
		void fill(const Complex<double> &val);

		Array operator+(const Array &other) const;

		void add(const Array &other, Array &res) const;

		std::string str() const;

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

		template<typename A, typename B, typename C, class FUNC>
		static void simpleCPUop(librapid::Accelerator locnA,
								librapid::Accelerator locnB,
								librapid::Accelerator locnC,
								const A *a, const B *b, C *c, size_t size,
								const FUNC &op);

		template<typename A, typename B, typename C>
		static void simpleFill(librapid::Accelerator locnA,
							   librapid::Accelerator locnB,
							   A *data, B *, size_t size,
							   C val);

		template<typename A, typename B>
		static void printLinear(librapid::Accelerator locnA,
								librapid::Accelerator locnB,
								A *data, B *, size_t size,
								std::string &res);

	private:
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