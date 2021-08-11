#ifndef LIBRAPID_COLOR
#define LIBRAPID_COLOR

#include <string>
#include <librapid/math/rapid_math.hpp>

// If RGB is defined by wingdi.h, undefine it
#ifdef RGB
#undef RGB
#define LIBRAPID_REDEF_RGB
#endif

namespace librapid
{
	namespace color
	{
		// RGB color container
		typedef struct RGB
		{
			int red = 0;
			int green = 0;
			int blue = 0;

			RGB(int r, int g, int b) :
				red(r), green(g), blue(b)
			{}

		#ifdef LIBRAPID_REDEF_RGB
			LR_INLINE operator COLORREF() const
			{
				return (COLORREF) (((BYTE) (red) | ((WORD) ((BYTE) (green)) << 8)) | (((DWORD) (BYTE) (blue)) << 16));
			}
		#endif
		} RGB;

		// HSL color container
		typedef struct HSL
		{
			double hue = 0;
			double saturation = 0;
			double lightness = 0;

			HSL(double h, double s, double l) :
				hue(h), saturation(s), lightness(l)
			{}
		} HSL;

		/**
		 * \rst
		 *
		 * Convert an RGB value to an HSL value
		 *
		 * \endrst
		 */
		HSL rgbToHsl(const RGB &col)
		{
			const double rp = col.red / 255.0;
			const double gp = col.green / 255.0;
			const double bp = col.blue / 255.0;
			const auto cMax = math::max(rp, gp, bp);
			const auto cMin = math::min(rp, gp, bp);
			const auto delta = cMax - cMin;

			double hue = 0;
			double saturation = 0;
			double lightness;

			// Hue
			if (cMax == rp) hue = 60 * fmod(((gp - bp) / delta), 6);
			else if (cMax == gp) hue = 60 * ((bp - rp) / delta + 2);
			else if (cMax == gp) hue = 60 * ((rp - gp) / delta + 4);

			// Lightness
			lightness = (cMax - cMin) / 2;

			// Saturation
			if (delta != 0)
				saturation = delta / (1 - math::abs(2 * lightness - 1));

			return {hue, saturation, lightness};
		}

		/**
		 * \rst
		 *
		 * Convert an HSL value to an RGB value
		 *
		 * \endrst
		 */
		RGB hslToRgb(const HSL &col)
		{
			const double c = (1 - math::abs(2 * col.lightness - 1)) * col.saturation;
			const double x = c * (1 - math::abs(fmod(col.hue / 60, 2) - 1));
			const double m = col.lightness - c / 2;
			const double h = col.hue;

			double rp = 0, gp = 0, bp = 0;

			if (h >= 0 && h < 60) rp = c, gp = x, bp = 0;
			else if (h >= 60 && h < 120) rp = x, gp = c, bp = 0;
			else if (h >= 120 && h < 180) rp = 0, gp = c, bp = x;
			else if (h >= 180 && h < 240) rp = 0, gp = x, bp = c;
			else if (h >= 240 && h < 300) rp = x, gp = 0, bp = c;
			else if (h >= 300 && h < 360) rp = c, gp = 0, bp = x;

			return {
				(int) ((rp + m) * 255),
				(int) ((gp + m) * 255),
				(int) ((bp + m) * 255)
			};
		}

		/**
		 * \rst
		 *
		 * Merge two RGB color values
		 *
		 * \endrst
		 */
		RGB mergeColors(RGB colorA, RGB colorB)
		{
			int r = colorA.red + colorB.red;
			int g = colorA.green + colorB.green;
			int b = colorA.blue + colorB.blue;

			if (r > 255 || g > 255 || b > 255)
			{
				double max = r;

				if (g > max) max = g;
				if (b > max) max = b;

				max = 255 / max;

				return {(int) (r * max), (int) (g * max), (int) (b * max)};
			}

			return {r / 2, g / 2, b / 2};
		}

		RGB mergeColors(RGB colorA, HSL colorB)
		{
			return mergeColors(colorA, hslToRgb(colorB));
		}

		HSL mergeColors(HSL colorA, RGB colorB)
		{
			return rgbToHsl(mergeColors(hslToRgb(colorA), colorB));
		}

		HSL mergeColors(HSL colorA, HSL colorB)
		{
			return rgbToHsl(mergeColors(hslToRgb(colorA), hslToRgb(colorB)));
		}

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

		std::string fore(const RGB &col)
		{
			std::string result = "\033[38;2;";

			result += std::to_string(col.red) + ";";
			result += std::to_string(col.green) + ";";
			result += std::to_string(col.blue);

			return result + "m";
		}

		std::string fore(const HSL &col)
		{
			return fore(hslToRgb(col));
		}

		std::string fore(int r, int g, int b)
		{
			return fore(RGB(r, g, b));
		}

		std::string back(const RGB &col)
		{
			std::string result = "\033[48;2;";

			result += std::to_string(col.red) + ";";
			result += std::to_string(col.green) + ";";
			result += std::to_string(col.blue);

			return result + "m";
		}

		std::string back(const HSL &col)
		{
			return back(hslToRgb(col));
		}

		std::string back(int r, int g, int b)
		{
			return back(RGB(r, g, b));
		}

		namespace imp
		{
			class ColorReset
			{
			public:
				ColorReset()
				{
					std::cout << "\033[0m";
				}

				~ColorReset()
				{
					std::cout << "\033[0m";
				}
			};

			ColorReset reset_after_close = ColorReset();
		}

		std::ostream &operator<<(std::ostream &os, const RGB &col)
		{
			return os << color::fore(col);
		}

		std::ostream &operator<<(std::ostream &os, const HSL &col)
		{
			return os << color::fore(hslToRgb(col));
		}
	}
}

#endif //LIBRAPID_COLOR
