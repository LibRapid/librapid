#include <librapid/array/multiarray.hpp>
#include <librapid/utils/array_utils.hpp>

namespace librapid {
	std::pair<int64_t, int64_t> Array::stringifyFormatPreprocess(bool stripMiddle,
																 bool autoStrip) const {
		if (autoStrip) {
			if (m_extent.size() >= 1000) stripMiddle = true;

			// Edge case for row and column vectors
			if (ndim() == 1) stripMiddle = false;
			else if (ndim() == 2 && m_extent[1] == 1) stripMiddle = false;
			else if (ndim() == 2 && m_extent[0] == 1) stripMiddle = false;
		}

		// Scalar values
		if (m_isScalar) {
			// std::pair<int64_t, int64_t> res;
			// AUTOCAST_UNARY(imp::autocastBeforeAfterDecimal, makeVoidPtr(),
			// 			   validRawArray, res);

			std::pair<int64_t, int64_t> res;
			imp::autocastBeforeAfterDecimal(createRaw(), res);
			return res;
		}

		int64_t longestIntegral = 0, longestDecimal = 0;

		// Vectors
		if (ndim() == 1) {
			int64_t index = 0;

			for (int64_t i = 0; i < m_extent.size(); ++i, ++index) {
				if (stripMiddle && i == 3)
					i = m_extent.size() - 3;

				auto sublongest = subscript(i).stringifyFormatPreprocess(stripMiddle,
																		 false);

				if (sublongest.first > longestIntegral)
					longestIntegral = sublongest.first;

				if (sublongest.second > longestDecimal)
					longestDecimal = sublongest.second;
			}

			return {longestIntegral, longestDecimal};
		}

		// Everything else
		int64_t index = 0;
		int64_t vec_size = stripMiddle ? 6 : m_extent[0];
		std::string res = "[";

		for (int64_t i = 0; i < m_extent[0]; ++i, ++index) {
			if (stripMiddle && i == 3)
				i = m_extent[0] - 3;

			auto sublongest = subscript(i).stringifyFormatPreprocess(stripMiddle,
																	 false);

			if (sublongest.first > longestIntegral)
				longestIntegral = sublongest.first;

			if (sublongest.second > longestDecimal)
				longestDecimal = sublongest.second;
		}

		return {longestIntegral, longestDecimal};
	}

	std::string Array::stringify(int64_t indent, bool showCommas,
								 bool stripMiddle, bool autoStrip,
								 std::pair<int64_t, int64_t> &longest,
								 int64_t &printedRows, int64_t &printedCols) const {
		printedRows = 0;
		printedCols = 0;

		// Non-initialized arrays
		if (m_references == nullptr)
			return "[NONE]";

		if (autoStrip) {
			if (m_extent.size() >= 1000) stripMiddle = true;

			// Edge case for row and column vectors
			if (ndim() == 1) stripMiddle = false;
			else if (ndim() == 2 && m_extent[1] == 1) stripMiddle = false;
			else if (ndim() == 2 && m_extent[0] == 1) stripMiddle = false;
		}

		// Scalar values
		if (m_isScalar) {
			std::string res;
			imp::autocastFormatValue(createRaw(), res);
			return res;
		}

		// Find the numbers being printed with the most
		// digits before and after the decimal point
		if (longest.first == 0 && longest.second == 0)
			longest = stringifyFormatPreprocess(false, true);

		// Vectors
		if (ndim() == 1) {
			int64_t index = 0;
			std::string res = "[";

			for (int64_t i = 0; i < m_extent.size(); ++i, ++index) {
				if (stripMiddle && i == 3) {
					i = m_extent.size() - 3;
					res += "... ";
					printedCols += 4;
				}

				int64_t tmpRows, tmpCols;
				std::string tempVal = subscript(i).stringify(indent + 1,
															 showCommas,
															 stripMiddle,
															 false, longest,
															 tmpRows, tmpCols);

				// Locate the decimal point and calculate
				// the number of digits before and after it
				int64_t before, after;
				index = tempVal.find('.');

				// Align the +/- for complex datatypes
				if (m_dtype == Datatype::CFLOAT64) {
					index = tempVal.find('+', 1);
					if (index == std::string::npos)
						index = tempVal.find('-', 1);
				}

				if (index == std::string::npos) {
					// No decimal point
					before = tempVal.length();
					after = 0;
				} else {
					before = index;
					after = tempVal.length() - index - 1;
				}

				int64_t addBefore, addAfter;
				addBefore = longest.first - before;
				addAfter = longest.second - after;

				std::string formatted;
				formatted += std::string(addBefore, ' ');
				formatted += tempVal;
				formatted += std::string(addAfter, ' ');

				if (i + 1 < m_extent.size()) formatted += showCommas ? ", " : " ";

				printedCols += formatted.length();
				res += formatted;
			}

			return res + "]";
		}

		// Everything else
		int64_t index = 0;
		std::string res = "[";

		for (int64_t i = 0; i < m_extent[0]; ++i, ++index) {
			if (stripMiddle && i == 3) {
				i = m_extent[0] - 3;
				res += "...\n" + std::string(indent + 1, ' ');
				printedRows++;
			}

			int64_t tmpRows, tmpCols;
			res += subscript(i).stringify(indent + 1, showCommas,
										  stripMiddle, false, longest,
										  tmpRows, tmpCols);

			printedRows++;
			printedCols = tmpCols;

			if (i + 1 < m_extent[0]) {
				res += std::string((ndim() > 2) + 1, '\n');
				res += std::string(indent + 1, ' ');
			}
		}

		printedCols += ndim() * 2;

		return res + "]";
	}

	std::string Array::str(int64_t indent, bool showCommas,
						   int64_t &printedRows, int64_t &printedCols) const {
		std::pair<int64_t, int64_t> longest;
		return stringify(indent, showCommas, false, true, longest,
						 printedRows, printedCols);
	}
}