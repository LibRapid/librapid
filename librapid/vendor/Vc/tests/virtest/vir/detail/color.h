/*{{{
Copyright © 2017 Matthias Kretz <kretz@kde.org>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the names of contributing organizations nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

}}}*/

#ifndef VIR_DETAIL_MAY_USE_COLOR_H_
#define VIR_DETAIL_MAY_USE_COLOR_H_

#include "macros.h"
#include <ostream>

#if defined(__GNUC__) && !defined(_WIN32) && defined(_GLIBCXX_OSTREAM)
#define VIR_HACK_OSTREAM_FOR_TTY 1
#endif

#ifdef VIR_HACK_OSTREAM_FOR_TTY
#include <unistd.h>
#include <ext/stdio_sync_filebuf.h>
#endif

namespace vir
{
namespace detail
{
#ifdef VIR_HACK_OSTREAM_FOR_TTY
static bool isATty(const std::ostream &os)
{
  __gnu_cxx::stdio_sync_filebuf<char> *hack =
      dynamic_cast<__gnu_cxx::stdio_sync_filebuf<char> *>(os.rdbuf());
  if (!hack) {
    return false;
  }
  FILE *file = hack->file();
  return 1 == isatty(fileno(file));
}
VIR_ALWAYS_INLINE VIR_CONST bool may_use_color(const std::ostream &os)
{
  static int result = -1;
  if (VIR_IS_UNLIKELY(result == -1)) {
    result = isATty(os);
  }
  return result;
}
#else
constexpr bool may_use_color(const std::ostream &) { return false; }
#endif

namespace color
{
struct Color {
  const char *data;
};

static constexpr Color red    = {"\033[1;40;31m"};
static constexpr Color green  = {"\033[1;40;32m"};
static constexpr Color yellow = {"\033[1;40;33m"};
static constexpr Color blue   = {"\033[1;40;34m"};
static constexpr Color normal = {"\033[0m"};

inline std::ostream &operator<<(std::ostream &out, const Color &c)
{
  if (may_use_color(out)) {
    out << c.data;
  }
  return out;
}
}  // namespace color

}  // namespace detail
}  // namespace vir

#endif  // VIR_DETAIL_MAY_USE_COLOR_H_
