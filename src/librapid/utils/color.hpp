#ifndef LIBRAPID_COLOR
#define LIBRAPID_COLOR

#include <string>
#include <librapid/math/rapid_math.hpp>

// If RGB is defined by wingdi.h, undefine it
#ifdef RGB
#undef RGB
#define LIBRAPID_REDEF_RGB
#endif

namespace librapid {
	// RGB color container
	typedef struct RGB {
		int red = 0;
		int green = 0;
		int blue = 0;

		RGB(int r, int g, int b);

#ifdef LIBRAPID_REDEF_RGB

		operator COLORREF() const;

#endif
	} RGB;

	// HSL color container
	typedef struct HSL {
		double hue = 0;
		double saturation = 0;
		double lightness = 0;

		HSL(double h, double s, double l);
	} HSL;

	/**
	 * \rst
	 *
	 * Convert an RGB value to an HSL value
	 *
	 * \endrst
	 */
	HSL rgbToHsl(const RGB &col);

	/**
	 * \rst
	 *
	 * Convert an HSL value to an RGB value
	 *
	 * \endrst
	 */
	RGB hslToRgb(const HSL &col);

	/**
	 * \rst
	 *
	 * Merge two RGB color values
	 *
	 * \endrst
	 */
	RGB mergeColors(const RGB &colorA, const RGB &colorB);

	RGB mergeColors(const RGB &colorA, const HSL &colorB);

	HSL mergeColors(const HSL &colorA, const RGB &colorB);

	HSL mergeColors(const HSL &colorA, const HSL &colorB);

	constexpr char clear[] = "\033[0m";
	constexpr char bold[] = "\033[1m";
	constexpr char blink[] = "\033[5m";

	constexpr char black[] = "\033[30m";
	constexpr char red[] = "\033[31m";
	constexpr char green[] = "\033[32m";
	constexpr char yellow[] = "\033[33m";
	constexpr char blue[] = "\033[34m";
	constexpr char magenta[] = "\033[35m";
	constexpr char cyan[] = "\033[36m";
	constexpr char white[] = "\033[37m";
	constexpr char brightBlack[] = "\033[90m";
	constexpr char brightRed[] = "\033[91m";
	constexpr char brightGreen[] = "\033[92m";
	constexpr char brightYellow[] = "\033[93m";
	constexpr char brightBlue[] = "\033[94m";
	constexpr char brightMagenta[] = "\033[95m";
	constexpr char brightCyan[] = "\033[96m";
	constexpr char brightWhite[] = "\033[97m";

	constexpr char blackBackground[] = "\033[40m";
	constexpr char redBackground[] = "\033[41m";
	constexpr char greenBackground[] = "\033[42m";
	constexpr char yellowBackground[] = "\033[43m";
	constexpr char blueBackground[] = "\033[44m";
	constexpr char magentaBackground[] = "\033[45m";
	constexpr char cyanBackground[] = "\033[46m";
	constexpr char whiteBackground[] = "\033[47m";
	constexpr char brightBlackBackground[] = "\033[100m";
	constexpr char brightRedBackground[] = "\033[101m";
	constexpr char brightGreenBackground[] = "\033[102m";
	constexpr char brightYellowBackground[] = "\033[103m";
	constexpr char brightBlueBackground[] = "\033[104m";
	constexpr char brightMagentaBackground[] = "\033[105m";
	constexpr char brightCyanBackground[] = "\033[106m";
	constexpr char brightWhiteBackground[] = "\033[107m";

	std::string fore(const RGB &col);

	std::string fore(const HSL &col);

	std::string fore(int r, int g, int b);

	std::string back(const RGB &col);

	std::string back(const HSL &col);

	std::string back(int r, int g, int b);

	namespace imp {
		class ColorReset {
		public:
			ColorReset();

			~ColorReset();
		};
	}

	std::ostream &operator<<(std::ostream &os, const RGB &col);

	std::ostream &operator<<(std::ostream &os, const HSL &col);
}

#endif //LIBRAPID_COLOR