import os
import sys
import math
from collections.abc import Iterable
import shutil
import time

# For some reason this fixes some problems on Windows
os.system('')

class frange:
	def __init__(self, start, end=None, step=None):
		self.start = None
		self.end = None
		self.step = None

		if end is None:
			self.end = float(start)
			self.start = 0.0
			self.step = float(step) if step is not None else 1.0
		else:
			self.start = float(start)
			self.end = float(end)
			self.step = float(step) if step is not None else 1.0

	def __str__(self):
		return "frange(start={}, end={}, step={}".format(self.start, self.end, self.step)

	def __repr__(self):
		return str(self)

	def __len__(self):
		return int((self.end - self.start) / self.step)

	def __iter__(self):
		l = len(self)
		for i in range(l):
			yield self.start + self.step * i

	def __getitem__(self, index):
		if index < len(self):
			return self.start + self.step * index
		raise IndexError("Index out of range")

class Progress:
	def __init__(self, iterable=None, message=None, erase=False, start=None, end=None, step=None, smoothness=0.85):
		self.iterable = None
		self.message = None
		self.start = None
		self.end = None
		self.step = None
		self.length = None
		self.smoothness = smoothness
		self.erase = erase

		self.fillChar = "█"
		self.emptyChar = "█"

		# Provide a special update routine for IDLE, in which carriage return does not work
		self.idleUpdate = None

		if isinstance(message, str):
			self.message = message

		if isinstance(iterable, Iterable):
			# Use an iterable

			self.iterable = iterable
			self.length = len(iterable)
		elif isinstance(end, (int, float)):
			# End is being used for sure

			self.end = end

			if start is None:
				self.start = 0
			elif isinstance(start, (int, float)):
				self.start = start
			else:
				raise TypeError("Start must be int or float")

			if step is None:
				self.step = 1
			elif isinstance(step, (int, float)):
				self.step = step
			else:
				raise TypeError("Step must be int or float")
		else:
			raise ValueError("At a minimum, an Iterable must be passed or an end point must be specified")
		
		self.iterStart = None
		self.iterCount = 0
		self.termWidth = shutil.get_terminal_size(fallback=(120, 50)).columns
		self.itsPerSec = 0
		self.itsPassed = 0
		self.prevYield = 0
		self.deltaTime = 0
		self.iterMod = 1
		self.timeMod = 1
		self.maxLen = None
		self.its = 1
		self.deltaTimeAdjusted = 0
		self.currentMinutes = "00"
		self.currentSeconds = "00"
		self.remainingMinutes = "00"
		self.remainingSeconds = "00"
		self.iterationsPerSecond = 0
		self.iterationsPerSecond = "it/s"

		if self.iterable is not None:
			self.maxLen = len(str(self.iterable[-1])) + 1
		elif self.end is not None:
			self.maxLen = len(str(self.end)) + 1


		if "idlelib.run" in sys.modules:
			# Running in IDLE
			self.idleUpdate = len(self.iterable) // shutil.get_terminal_size(fallback=(120, 50)).columns

	def reset(self):
		self.iterStart = time.perf_counter()
		self.iterCount = 0
		self.termWidth = shutil.get_terminal_size(fallback=(120, 50)).columns
		self.itsPerSec = 0
		self.itsPassed = 0
		self.prevYield = 0
		self.deltaTime = 0
		self.iterMod = 1
		self.timeMod = 1
		self.maxLen = None
		self.its = 0
		self.deltaTimeAdjusted = 0

		if self.iterable is not None:
			self.maxLen = len(str(self.iterable[-1])) + 1
		elif self.end is not None:
			self.maxLen = len(str(self.end)) + 1

	@staticmethod
	def generateBar(fill, length, fillChar="█", emptyChar="█"):
		# Create the bar section of the entire progress bar
		fillChars = math.floor(length * fill)
		res = "\033[32m" + fillChar * fillChars
		res += "\033[31m" + emptyChar * (length - len(res) + 5) + "\033[0m"

		return res

	def _eval(self):
		# Reset the bar on the first iteration just in case
		if self.iterCount == 0:
			self.reset()

		# Update the progress bar without adjusting any iterator values

		# Percentage completion
		percentage = "{}%".format(str(round(self.iterCount / self.length * 100)).rjust(3))

		# Fraction
		fraction = "{}/{}".format(str(self.iterCount).rjust(self.maxLen if self.maxLen is not None else 0), self.length)

		# Timing
		if self.iterCount % self.timeMod == 0:
			dt = round(time.perf_counter() - self.iterStart)
			seconds = dt % 60
			minutes = (dt // 60) % 60
			self.currentMinutes = ("0" if len(str(minutes)) < 2 else "") + str(minutes)
			self.currentSeconds = ("0" if len(str(seconds)) < 2 else "") + str(seconds)
		
			dt = int((self.length - self.iterCount) / (self.its if self.its != 0.0 else 0.0000001))
			seconds = dt % 60
			minutes = dt // 60
			self.remainingMinutes = ("0" if len(str(minutes)) < 2 else "") + str(minutes)
			self.remainingSeconds = ("0" if len(str(seconds)) < 2 else "") + str(seconds)
	
			self.iterationsPerSecond = round(self.its if self.its >= 1 else ((1 / self.its) if self.its != 0.0 else 0.0000001), 1)
			self.iterationsPerSecondTxt = "it/s" if self.its >= 1 else "s/it"
	
		timing = "[{}:{}|{}:{}, {}{}]".format(self.currentMinutes, self.currentSeconds, self.remainingMinutes, self.remainingSeconds, self.iterationsPerSecond, self.iterationsPerSecondTxt)

		# Bar properties
		messageWidth = len(self.message) + 2 if self.message is not None else 0

		barWidth = self.termWidth - messageWidth - len(percentage) - len(fraction) - len(timing) - 3
		usePercentage = True
		useFraction = True
		useTiming = True

		if barWidth < 5:
			# Console is too small, so remove fraction
			barWidth += len(fraction)
			useFraction = False

		if barWidth < 5:
			# Console is too small, so remove fraction
			barWidth += len(timing) + 1
			useTiming = False

		if barWidth < 5:
			# Console is still too small so remove percentage
			barWidth += len(percentage)
			usePercentage = False

		if barWidth < 5:
			# Console is STILL too small!
			raise OSError("Your console is far too small to use a progress bar")

		# Bar
		bar = Progress.generateBar(self.iterCount / self.length,
								   barWidth,
								   self.fillChar,
								   self.emptyChar)

		# return percentage + "|" + bar + "| " + fraction + " " + timing
		res = self.message + " |" if self.message is not None else ""
		res += percentage if usePercentage else ""
		res += "|"
		res += bar
		res += "|"
		res += fraction if useFraction else ""
		res += (" " + timing) if useTiming else ""

		return res

	def update(self):
		self.iterCount += 1
		self.itsPassed += 1

		if self.iterCount % self.iterMod == 0:
			# The most precise time counter
			current = time.perf_counter()

			self.deltaTime = current - self.prevYield
			self.deltaTimeAdjusted = self.deltaTime / self.itsPassed
			self.prevYield = current
			self.its = 1 / self.deltaTimeAdjusted
			print(self._eval(), end="\r")
			self.itsPassed = 0

			if self.deltaTime > (1 - self.smoothness):
				if self.iterMod > 2:
					self.iterMod /= 2
			else:
				self.iterMod *= 2

			if self.deltaTime > (0.1):
				if self.timeMod >= 2:
					self.timeMod /= 2
			else:
				if self.timeMod < 5000:
					self.timeMod *= 2

	def __iter__(self):
		self.reset()

		print(self._eval(), end="\r")
		for value in self.iterable:
			yield value

			if self.idleUpdate is None:
				self.update()
			else:
				self.iterCount += 1
				if self.iterCount % self.idleUpdate == 0:
					print("#", end="")

		if self.idleUpdate is None:
			if not self.erase:
				print(self._eval())
			else:
				print("\r", end="")
				print(" " * shutil.get_terminal_size(fallback=(120, 50)).columns, end="")
				print("\r", end="")
		else:
			print("")
