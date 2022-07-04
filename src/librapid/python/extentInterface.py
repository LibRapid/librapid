import _librapid
from . import argCastHelper as caster

class Extent:
	def __init__(self, *args):
		if len(args) == 0:
			self._extent = _librapid.Extent()
		else:
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
			return Extent(self._extent.swivelled(tmp))

	def swivel(self, *args):
		tmp = caster.cast(args, int, _librapid.Extent, Extent, lambda x: x._extent)
		if tmp is None:
			self._extent.swivel(tmp)

	def size(self):
		return self._extent.size()

	def sizeAdjusted(self):
		return self._extent.sizeAdjusted()

	def dims(self):
		return self._extent.dims()

	def __getitem__(self, index:int):
		return self._extent[index]

	def __setitem__(self, index:int, val:int):
		self._extent[index] = val

	def adjusted(self, index:int):
		return self._extent.adjusted(index)

	def __eq__(self, *args):
		tmp = caster.cast(args, int, _librapid.Extent, Extent, lambda x: x._extent)
		if tmp is None:
			raise ValueError("{} is invalid for Extent equality check".format(args))
		return self._extent == tmp

	def __neq__(self, other):
		return not (self == other)

	def str(self):
		return self._extent.str()

	def __str__(self):
		return self._extent.str()

	def __repr__(self):
		return "<librapid::" + self._extent.str() + ">"
