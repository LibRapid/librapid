import _librapid
import argCastHelper as caster

class Extent:
	def __init__(self, *args):
		self._extent = caster.cast(args, int, _librapid.Extent)
		if self._extent is None:
			raise ValueError("Invalid constructor arguments. Input must be list-like object or individual values")

	@staticmethod
	def zero(dims:int):
		return _librapid.Extent.zero(dims)

	def stride(self):
		return Extent(self._extent.stride())

	def strideAdjusted(self):
		return Extent(self._extent.strideAdjusted())

	def index(self, *args):
		tmp = caster.cast(args, int, _librapid.Extent, Extent, lambda x: x._extent)
		if tmp is None:
			raise ValueError("{} is invalid for Extent.index(). Input must be Extent, list-like or integer values".format(args))
		return self._extent.index(tmp._extent)

	def indexAdjusted(self, *args):
		tmp = caster.cast(args, int, _librapid.Extent, Extent, lambda x: x._extent)
		if tmp is None:
			raise ValueError("{} is invalid for Extent.indexAdjusted(). Input must be Extent, list-like or integer values".format(args))
		return self._extent.indexAdjusted(tmp._extent)

	def reverseIndex(self, index:int):
		return Extent(self._extent.reverseIndex(index))

	def partial(self, start:int=0, end:int=-1):
		return Extent(self._extent.partial(start, end))

	def swivelled(self, *args):
		tmp = caster.cast(args, int, _librapid.Extent, Extent, lambda x: x._extent)
		if tmp is None:
			pass
