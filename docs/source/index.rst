LibRapid
########

.. plot::

	import matplotlib.pyplot as plt
	import matplotlib as mpl
	import librapid as lrp

	mpl.style.use("seaborn")
	plt.figure(facecolor=(0.07, 0.08, 0.09))

	vals = []
	for i in range(1000000):
		vals.append(lrp.randomGaussian())
	plt.hist(vals, 50)
	plt.title("Hello")

	# Plotting Style
	ax = plt.gca()
	ax.tick_params(axis='x', colors='white')
	ax.tick_params(axis='y', colors='white')
	ax.spines['left'].set_color('white')
	ax.spines['right'].set_color('white')
	ax.spines['top'].set_color('white') 
	ax.spines['bottom'].set_color('white')
	ax.title.set_color('white')
	ax.set_facecolor((0.07, 0.08, 0.09))
	plt.show()

Contents
========

.. toctree::
	:hidden:
	:maxdepth: 2
	:glob:

	modules/modules

.. panels::
	Modules

	+++

	.. link-button:: modules/modules
		:type: ref
		:text: View Page
		:classes: btn-outline-info btn-block stretched-link

	---

Licencing
=========

LibRapid is produced under the MIT License, so you are free to use the library
how you like for personal and commercial purposes, though this is subject to
some conditions, which can be found in full here: `LibRapid License`_

.. _LibRapid License: https://github.com/Pencilcaseman/librapid/blob/master/LICENSE
