#pragma once

#include "reducedMath.hpp"

namespace librapid { namespace detail {
	template<typename T>
	struct ColumnMajorOrder {
		typedef T result_type;

		int m;
		int n;

		ColumnMajorOrder(const int &_m, const int &_n) : m(_m), n(_n) {}

		T operator()(const int &idx) const {
			int row = idx % m;
			int col = idx / m;
			return row * m + col;
		}
	};

	template<typename T>
	struct RowMajorOrder {
		typedef T ResultType;

		int m;
		int n;

		RowMajorOrder(const int &_m, const int &_n) : m(_m), n(_n) {}

		T operator()(const int &idx) const {
			int row = idx % n;
			int col = idx / n;
			return col * n + row;
		}
	};

	template<typename T>
	struct TxColumnMajorOrder {
		typedef T ResultType;

		int m;
		int n;

		TxColumnMajorOrder(const int &_m, const int &_n) : m(_m), n(_n) {}

		T operator()(const int &idx) const {
			int row = idx / m;
			int col = idx % m;
			return col * n + row;
		}
	};

	template<typename T>
	struct TxRowMajorOrder {
		typedef T ResultType;

		int m;
		int n;

		TxRowMajorOrder(const int &_m, const int &_n) : m(_m), n(_n) {}

		T operator()(const int &idx) const {
			int row = idx % n;
			int col = idx / n;
			return row * m + col;
		}
	};

	struct ColumnMajorIndex {
		int m;
		int n;

		ColumnMajorIndex(const int &_m, const int &_n) : m(_m), n(_n) {}

		int operator()(const int &i, const int &j) const { return i + j * m; }
	};

	struct RowMajorIndex {
		int m;
		int n;

		RowMajorIndex(const int &_m, const int &_n) : m(_m), n(_n) {}

		RowMajorIndex(const ReducedDivisor &_m, const int &_n) : m(_m.get()), n(_n) {}

		int operator()(const int &i, const int &j) const { return j + i * n; }
	};
}} // namespace librapid::detail