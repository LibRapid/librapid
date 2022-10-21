#ifndef LIBRAPID_CORE_GNU_CONFIG_HPP
#define LIBRAPID_CORE_GNU_CONFIG_HPP

#define LIBRAPID_INLINE		   inline
#define LIBRAPID_ALWAYS_INLINE inline __attribute__((always_inline))

#if defined(LIBRAPID_ENABLE_ASSERT)
#	define LIBRAPID_STATUS(msg, ...)                                                              \
		do {                                                                                       \
			std::string funcName = FUNCTION;                                                       \
			if (funcName.length() > 75) funcName = "<Signature too Long>";                         \
			int maxLen = librapid::detail::internalMax((int)std::ceil(std::log(__LINE__)) + 6,     \
													   (int)strlen(FILENAME) + 6,                  \
													   (int)funcName.length() + 6,                 \
													   (int)strlen("WARN ASSERTION FAILED"));      \
			fmt::print(fmt::fg(fmt::color::green),                                                 \
					   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                                  \
					   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                        \
					   "STATUS",                                                                   \
					   FILENAME,                                                                   \
					   funcName,                                                                   \
					   __LINE__,                                                                   \
					   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                                \
					   maxLen + 5,                                                                 \
					   maxLen + 0,                                                                 \
					   maxLen - 4,                                                                 \
					   maxLen);                                                                    \
		} while (0)

#	define LIBRAPID_WARN(msg, ...)                                                                \
		do {                                                                                       \
			std::string funcName = FUNCTION;                                                       \
			if (funcName.length() > 75) funcName = "<Signature too Long>";                         \
			int maxLen = librapid::detail::internalMax((int)std::ceil(std::log(__LINE__)) + 6,     \
													   (int)strlen(FILENAME) + 6,                  \
													   (int)funcName.length() + 6,                 \
													   (int)strlen("WARN ASSERTION FAILED"));      \
			fmt::print(fmt::fg(fmt::color::yellow),                                                \
					   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                                  \
					   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                        \
					   "WARNING",                                                                  \
					   FILENAME,                                                                   \
					   funcName,                                                                   \
					   __LINE__,                                                                   \
					   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                                \
					   maxLen + 5,                                                                 \
					   maxLen + 0,                                                                 \
					   maxLen - 4,                                                                 \
					   maxLen);                                                                    \
		} while (0)

#	define LIBRAPID_ERROR(msg, ...)                                                               \
		do {                                                                                       \
			std::string funcName = FUNCTION;                                                       \
			if (funcName.length() > 75) funcName = "<Signature too Long>";                         \
			int maxLen = librapiod::detail::internalMax((int)std::ceil(std::log(__LINE__)) + 6,    \
														(int)strlen(FILENAME) + 6,                 \
														(int)funcName.length() + 6,                \
														(int)strlen("WARN ASSERTION FAILED"));     \
			fmt::print(fmt::fg(fmt::color::red),                                                   \
					   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                                  \
					   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                        \
					   "ERROR",                                                                    \
					   FILENAME,                                                                   \
					   funcName,                                                                   \
					   __LINE__,                                                                   \
					   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                                \
					   maxLen + 5,                                                                 \
					   maxLen + 0,                                                                 \
					   maxLen - 4,                                                                 \
					   maxLen);                                                                    \
			if (librapid::global::throwOnAssert) {                                                 \
				throw std::runtime_error(formatted);                                               \
			} else {                                                                               \
				fmt::print(fmt::fg(fmt::color::red), formatted);                                   \
				std::exit(1);                                                                      \
			}                                                                                      \
		} while (0)

#	define LIBRAPID_WASSERT(cond, msg, ...)                                                       \
		do {                                                                                       \
			if (!(cond)) {                                                                         \
				std::string funcName = FUNCTION;                                                   \
				if (funcName.length() > 75) funcName = "<Signature too Long>";                     \
				\ int maxLen =                                                                     \
				  librapid::detail::internalMax((int)std::ceil(std::log(__LINE__)) + 6,            \
												(int)strlen(FILENAME) + 6,                         \
												(int)funcName.length() + 6,                        \
												(int)strlen(#cond) + 6,                            \
												(int)strlen("WARN ASSERTION FAILED"));             \
				fmt::print(fmt::fg(fmt::color::yellow),                                            \
						   "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                              \
						   "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                               \
						   "{4:>{10}}]\n{5}\n",                                                    \
						   "WARN ASSERTION FAILED",                                                \
						   FILENAME,                                                               \
						   funcName,                                                               \
						   __LINE__,                                                               \
						   #cond,                                                                  \
						   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                            \
						   maxLen + 5,                                                             \
						   maxLen + 0,                                                             \
						   maxLen - 4,                                                             \
						   maxLen + 0,                                                             \
						   maxLen - 5);                                                            \
			}                                                                                      \
		} while (0)

#	define LIBRAPID_ASSERT(cond, msg, ...)                                                        \
		do {                                                                                       \
			std::string funcName = FUNCTION;                                                       \
			if (funcName.length() > 75) funcName = "<Signature too Long>";                         \
			if (!(cond)) {                                                                         \
				int maxLen = librapid::detail::internalMax((int)std::ceil(std::log(__LINE__)),     \
														   (int)strlen(FILENAME),                  \
														   (int)funcName.length(),                 \
														   (int)strlen(#cond),                     \
														   (int)strlen("ASSERTION FAILED"));       \
				std::string formatted = fmt::format(                                               \
				  "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                                       \
				  "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                                        \
				  "{4:>{10}}]\n{5}\n",                                                             \
				  "ASSERTION FAILED",                                                              \
				  FILENAME,                                                                        \
				  funcName,                                                                        \
				  __LINE__,                                                                        \
				  #cond,                                                                           \
				  fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                                     \
				  maxLen + 14,                                                                     \
				  maxLen + 9,                                                                      \
				  maxLen + 5,                                                                      \
				  maxLen + 9,                                                                      \
				  maxLen + 4);                                                                     \
				if (librapid::global::throwOnAssert) {                                             \
					throw std::runtime_error(formatted);                                           \
				} else {                                                                           \
					fmt::print(fmt::fg(fmt::color::red), formatted);                               \
					std::exit(1);                                                                  \
				}                                                                                  \
			}                                                                                      \
		} while (0)
#else
#	define LIBRAPID_WARN_ONCE(msg, ...)                                                           \
		do {                                                                                       \
		} while (0)
#	define LIBRAPID_STATUS(msg, ...)                                                              \
		do {                                                                                       \
		} while (0)
#	define LIBRAPID_WARN(msg, ...)                                                                \
		do {                                                                                       \
		} while (0)
#	define LIBRAPID_ERROR(msg, ...)                                                               \
		do {                                                                                       \
		} while (0)
#	define LIBRAPID_LOG(msg, ...)                                                                 \
		do {                                                                                       \
		} while (0)
#	define LIBRAPID_WASSERT(cond, ...)                                                            \
		do {                                                                                       \
		} while (0)
#	define LIBRAPID_ASSERT(cond, ...)                                                             \
		do {                                                                                       \
		} while (0)
#endif // LIBRAPID_ENABLE_ASSERT

#endif // LIBRAPID_CORE_GNU_CONFIG_HPP