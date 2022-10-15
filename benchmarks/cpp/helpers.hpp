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

	std::string heading;
	std::string line;
	for (lrc::i64 i = 0; i < headings.size(); ++i) {
		heading += fmt::format("{:>{}}", headings[i], colWidths[i]);
		line += std::string(colWidths[i], '-');
		if (i < headings.size()) {
			heading += " | ";
			line += "-+-";
		}
	}

	fmt::print("{}\n{}", heading, line);

	for (const auto &row : rows) {
		std::string rowString;
		for (lrc::i64 i = 0; i < row.size(); ++i) {
			rowString += fmt::format("{:>{}}", row[i], colWidths[i]);
			if (i < row.size()) { rowString += " | "; }
		}
		fmt::print("\n{}", rowString);
	}
	fmt::print("\n");
}
