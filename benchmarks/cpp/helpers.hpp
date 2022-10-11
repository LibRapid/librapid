#pragma once

#include <librapid>

namespace lrc = librapid;

void printTable(const std::vector<std::string> &headings,
				const std::vector<std::vector<std::string>> &rows) {
	std::vector<lrc::i64> colWidths(headings.size());
	for (lrc::i64 i = 0; i < headings.size(); i++) {
		colWidths[i] = static_cast<lrc::i64>(headings[i].size());
		for (const auto &row : rows) {
			colWidths[i] = static_cast<lrc::i64>(lrc::max(colWidths[i], row[i].size()));
		}
	}

	std::string line;
	for (lrc::i64 i = 0; i < headings.size(); ++i) {
		fmt::print("{:>{}}", headings[i], colWidths[i]);
		line += std::string(colWidths[i], '-');
		if (i < headings.size()) {
			fmt::print(" | ");
			line += "-+-";
		}
	}

	fmt::print("\n{}", line);

	for (const auto &row : rows) {
		fmt::print("\n");
		for (lrc::i64 i = 0; i < row.size(); ++i) {
			fmt::print("{:>{}}", row[i], colWidths[i]);
			if (i < row.size()) { fmt::print(" | "); }
		}
	}
	fmt::print("\n");
}
