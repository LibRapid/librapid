#pragma once

namespace librapid::assert {
	template<typename RaiseType, typename... Args>
	void librapidAssert(bool condition, const std::string &message, uint64_t line,
						std::string signature, const std::string &filename,
						const std::string &conditionString, const Args &...args) {
		if (!condition) {
			std::string formattedMessage = fmt::vformat(message, fmt::make_format_args(args...));

			if (global::printOnAssert) {
				if (signature.length() > 70) {
					// Truncate the signature
					signature = signature.substr(0, 67) + "...";
				}

				int maxLen = detail::internalMax((int)std::ceil(std::log(line)),
												 (int)filename.length(),
												 (int)conditionString.length(),
												 (int)signature.length(),
												 (int)strlen("ASSERTION FAILED"));

				// std::string formatted = fmt::format(
				//   "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "
				//   "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "
				//   "{4:>{10}}]\n{5}\n",
				//   "ASSERTION FAILED",
				//   filename,
				//   signature,
				//   line,
				//   conditionString,
				//   formattedMessage,
				//   maxLen + 14,
				//   maxLen + 9,
				//   maxLen + 5,
				//   maxLen + 9,
				//   maxLen + 4);

				// fmt::print(fmt::fg(fmt::color::red), formatted);
				// fmt::vprint(fmt::fg(fmt::color::red), formatted);

				fmt::print(fmt::fg(fmt::color::red),
				  "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "
				  "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "
				  "{4:>{10}}]\n{5}\n",
				  "ASSERTION FAILED",
				  filename,
				  signature,
				  line,
				  conditionString,
				  formattedMessage,
				  maxLen + 14,
				  maxLen + 9,
				  maxLen + 5,
				  maxLen + 9,
				  maxLen + 4);
			}

			throw RaiseType(formattedMessage);
		}
	}

	template<typename Format, typename... Args>
	void librapidStatusGeneric(const std::string &message, uint64_t line, std::string signature,
							   const std::string &filename, const std::string warningType,
							   const Format &format, Args &&...args) {
		if (signature.length() > 70) {
			// Truncate the signature
			signature = signature.substr(0, 67) + "...";
		}

		int maxLen = librapid::detail::internalMax((int)std::ceil(std::log(line)) + 6,
												   (int)filename.length() + 6,
												   (int)signature.length() + 6,
												   (int)warningType.length());

		std::string formattedMessage = fmt::vformat(message, fmt::make_format_args(args...));

		fmt::print(format,
				   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "
				   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",
				   "STATUS",
				   filename,
				   signature,
				   line,
				   formattedMessage,
				   maxLen + 5,
				   maxLen + 0,
				   maxLen - 4,
				   maxLen);
	}

	template<typename... Args>
	void librapidStatus(const std::string &message, uint64_t line, std::string signature,
						const std::string &filename, Args &&...args) {
		librapidStatusGeneric(message,
							  line,
							  signature,
							  filename,
							  "STATUS",
							  fmt::fg(fmt::color::green),
							  std::forward<Args>(args)...);
	}

	template<typename... Args>
	void librapidWarn(const std::string &message, uint64_t line, std::string signature,
					  const std::string &filename, Args &&...args) {
		librapidStatusGeneric(message,
							  line,
							  signature,
							  filename,
							  "WARNING",
							  fmt::fg(fmt::color::yellow),
							  std::forward<Args>(args)...);
	}

	template<typename RaiseType, typename... Args>
	void librapidError(const std::string &message, uint64_t line, std::string signature,
					   const std::string &filename, Args &&...args) {
		librapidStatusGeneric(message,
							  line,
							  signature,
							  filename,
							  "ERROR",
							  fmt::fg(fmt::color::red),
							  std::forward<Args>(args)...);
		throw RaiseType(message);
	}
} // namespace librapid::assert

#if !defined(LIBRAPID_ENABLE_ASSERT)
#	define LIBRAPID_ASSERT(condition, message, ...)                                               \
		do {                                                                                       \
		} while (false)

#	define LIBRAPID_ASSERT_WITH_EXCEPTION(raiseType, condition, message, ...)                     \
		do {                                                                                       \
		} while (false)

#	define LIBRAPID_STATUS(message, ...)                                                          \
		do {                                                                                       \
		} while (false)

#	define LIBRAPID_WARN(message, ...)                                                            \
		do {                                                                                       \
		} while (false)

#	define LIBRAPID_ERROR(message, ...)                                                           \
		do {                                                                                       \
		} while (false)
#endif // LIBRAPID_ENABLE_ASSERT
