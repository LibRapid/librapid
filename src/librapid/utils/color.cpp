#include <librapid/utils/color.hpp>
#include <librapid/math/rapid_math.hpp>

namespace librapid {
	RGB::RGB(int r, int g, int b) :
			red(r), green(g), blue(b) {}

#ifdef LIBRAPID_REDEF_RGB

	RGB::operator COLORREF() const {
		return (COLORREF) (((BYTE) (red) | ((WORD) ((BYTE) (green)) << 8)) | (((DWORD) (BYTE) (blue)) << 16));
	}

#endif

	HSL::HSL(double h, double s, double l) :
			hue(h), saturation(s), lightness(l) {}

	HSL rgbToHsl(const RGB &col) {
		const double rp = col.red / 255.0;
		const double gp = col.green / 255.0;
		const double bp = col.blue / 255.0;
		const auto cMax = max(rp, gp, bp);
		const auto cMin = min(rp, gp, bp);
		const auto delta = cMax - cMin;

		double hue = 0;
		double saturation = 0;
		double lightness;

		// Hue
		if (cMax == rp) hue = 60 * fmod(((gp - bp) / delta), 6);
		else if (cMax == gp) hue = 60 * ((bp - rp) / delta + 2);
		else if (cMax == bp) hue = 60 * ((rp - gp) / delta + 4);

		// Lightness
		lightness = (cMax - cMin) / 2;

		// Saturation
		if (delta != 0)
			saturation = delta / (1 - std::abs(2 * lightness - 1));

		return {hue, saturation, lightness};
	}

	RGB hslToRgb(const HSL &col) {
		const double c = (1 - abs(2 * col.lightness - 1)) * col.saturation;
		const double x = c * (1 - abs(fmod(col.hue / 60, 2) - 1));
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

	RGB mergeColors(const RGB &colorA, const RGB &colorB) {
		int r = colorA.red + colorB.red;
		int g = colorA.green + colorB.green;
		int b = colorA.blue + colorB.blue;

		if (r > 255 || g > 255 || b > 255) {
			double max = r;

			if (g > max) max = g;
			if (b > max) max = b;

			max = 255 / max;

			return {(int) (r * max), (int) (g * max), (int) (b * max)};
		}

		return {r / 2, g / 2, b / 2};
	}

	RGB mergeColors(const RGB &colorA, const HSL &colorB) {
		return mergeColors(colorA, hslToRgb(colorB));
	}

	HSL mergeColors(const HSL &colorA, const RGB &colorB) {
		return rgbToHsl(mergeColors(hslToRgb(colorA), colorB));
	}

	HSL mergeColors(const HSL &colorA, const HSL &colorB) {
		return rgbToHsl(mergeColors(hslToRgb(colorA), hslToRgb(colorB)));
	}

	std::string fore(const RGB &col) {
		std::string result = "\033[38;2;";

		result += std::to_string(col.red) + ";";
		result += std::to_string(col.green) + ";";
		result += std::to_string(col.blue);

		return result + "m";
	}

	std::string fore(const HSL &col) {
		return fore(hslToRgb(col));
	}

	std::string fore(int r, int g, int b) {
		return fore(RGB(r, g, b));
	}

	std::string back(const RGB &col) {
		std::string result = "\033[48;2;";

		result += std::to_string(col.red) + ";";
		result += std::to_string(col.green) + ";";
		result += std::to_string(col.blue);

		return result + "m";
	}

	std::string back(const HSL &col) {
		return back(hslToRgb(col));
	}

	std::string back(int r, int g, int b) {
		return back(RGB(r, g, b));
	}

	namespace imp {
		ColorReset::ColorReset() {
			std::cout << "\033[0m";
		}

		ColorReset::~ColorReset() {
			std::cout << "\033[0m";
		}

		ColorReset reset_after_close = ColorReset();
	}

	std::ostream &operator<<(std::ostream &os, const RGB &col) {
		return os << fore(col);
	}

	std::ostream &operator<<(std::ostream &os, const HSL &col) {
		return os << fore(hslToRgb(col));
	}
}