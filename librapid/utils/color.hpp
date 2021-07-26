#ifndef LIBRAPID_COLOR
#define LIBRAPID_COLOR

#include <string>
#include <librapid/math/rapid_math.hpp>

namespace librapid
{
	namespace color
	{
		// RGB color container
		typedef struct rgb
		{
			int red = 0;
			int green = 0;
			int blue = 0;

			rgb(int r, int g, int b) :
				red(r), green(g), blue(b)
			{}
		} rgb;

		// HSL color container
		typedef struct hsl
		{
			double hue = 0;
			double saturation = 0;
			double lightness = 0;

			hsl(double h, double s, double l) :
				hue(h), saturation(s), lightness(l)
			{}
		} hsl;

		/**
		 * \rst
		 *
		 * Convert an RGB value to an HSL value
		 *
		 * \endrst
		 */
		hsl rgb_to_hsl(const rgb &col)
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
		rgb hsl_to_rgb(const hsl &col)
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
		rgb merge_colors(rgb colorA, rgb colorB)
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

		rgb merge_colors(rgb colorA, hsl colorB)
		{
			return merge_colors(colorA, hsl_to_rgb(colorB));
		}

		hsl merge_colors(hsl colorA, rgb colorB)
		{
			return rgb_to_hsl(merge_colors(hsl_to_rgb(colorA), colorB));
		}

		hsl merge_colors(hsl colorA, hsl colorB)
		{
			return rgb_to_hsl(merge_colors(hsl_to_rgb(colorA), hsl_to_rgb(colorB)));
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
		constexpr char bright_black[] = "\033[90m";
		constexpr char bright_red[] = "\033[91m";
		constexpr char bright_green[] = "\033[92m";
		constexpr char bright_yellow[] = "\033[93m";
		constexpr char bright_blue[] = "\033[94m";
		constexpr char bright_magenta[] = "\033[95m";
		constexpr char bright_cyan[] = "\033[96m";
		constexpr char bright_white[] = "\033[97m";

		std::string fore(const rgb &col)
		{
			std::string result = "\033[38;2;";

			result += std::to_string(col.red) + ";";
			result += std::to_string(col.green) + ";";
			result += std::to_string(col.blue);

			return result + "m";
		}

		std::string fore(const hsl &col)
		{
			return fore(hsl_to_rgb(col));
		}

		std::string fore(int r, int g, int b)
		{
			return fore(rgb(r, g, b));
		}

		std::string back(const rgb &col)
		{
			std::string result = "\033[48;2;";

			result += std::to_string(col.red) + ";";
			result += std::to_string(col.green) + ";";
			result += std::to_string(col.blue);

			return result + "m";
		}

		std::string back(const hsl &col)
		{
			return back(hsl_to_rgb(col));
		}

		std::string back(int r, int g, int b)
		{
			return back(rgb(r, g, b));
		}

		namespace imp
		{
			class color_reset
			{
			public:
				color_reset()
				{
					std::cout << "\033[0m";
				}

				~color_reset()
				{
					std::cout << "\033[0m";
				}
			};

			color_reset reset_after_close = color_reset();
		}

		std::ostream &operator<<(std::ostream &os, const rgb &col)
		{
			return os << color::fore(col);
		}

		std::ostream &operator<<(std::ostream &os, const hsl &col)
		{
			return os << color::fore(hsl_to_rgb(col));
		}
	}
}

#endif //LIBRAPID_COLOR
