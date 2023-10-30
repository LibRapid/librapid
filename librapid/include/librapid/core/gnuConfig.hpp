#ifndef LIBRAPID_CORE_GNU_CONFIG_HPP
#define LIBRAPID_CORE_GNU_CONFIG_HPP

#define LIBRAPID_INLINE        inline
#define LIBRAPID_ALWAYS_INLINE inline __attribute__((always_inline))

#if defined(LIBRAPID_ENABLE_ASSERT)

#	define LIBRAPID_ASSERT(condition, message, ...)                                               \
		librapid::assert::librapidAssert<std::runtime_error>(condition,                            \
															 message,                              \
															 __LINE__,                             \
															 FUNCTION,                             \
															 FILENAME,                             \
															 STRINGIFY(condition) __VA_OPT__(, )   \
															   __VA_ARGS__)

#	define LIBRAPID_ASSERT_WITH_EXCEPTION(raiseType, condition, message, ...)                     \
		librapid::assert::librapidAssert<raiseType>(condition,                                     \
													message,                                       \
													__LINE__,                                      \
													FUNCTION,                                      \
													FILENAME,                                      \
													STRINGIFY(condition) __VA_OPT__(, )            \
													  __VA_ARGS__)

#	define LIBRAPID_STATUS(message, ...)                                                          \
		librapid::assert::librapidStatus(                                                          \
		  message, __LINE__, FUNCTION, FILENAME __VA_OPT__(, ) __VA_ARGS__)

#	define LIBRAPID_WARN(message, ...)                                                            \
		librapid::assert::librapidWarn(                                                            \
		  message, __LINE__, FUNCTION, FILENAME __VA_OPT__(, ) __VA_ARGS__)

#	define LIBRAPID_ERROR(message, ...)                                                           \
		librapid::assert::librapidError<std::runtime_error>(                                       \
		  message, __LINE__, FUNCTION, FILENAME __VA_OPT__(, ) __VA_ARGS__)

#endif // LIBRAPID_ENABLE_ASSERT

#endif // LIBRAPID_CORE_GNU_CONFIG_HPP