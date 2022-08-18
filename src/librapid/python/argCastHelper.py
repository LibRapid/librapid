def cast(args, scalarTypes, retType, allowed=None, extractor=None):
	if isinstance(args, retType):
		return retType(args)

	if allowed is not None and extractor is not None:
		if isinstance(args, allowed):
			return extractor(args)
	
	if isinstance(args, (list, tuple)):
		if len(args) == 0:
			return None

		# Check for a list type
		if allowed is not None:
			typeList = (list, tuple, retType, allowed)
		else:
			typeList = (list, tuple, retType)

		if isinstance(args[0], typeList):
			if len(args) > 1:
				return None
			return cast(args[0], scalarTypes, retType, allowed, extractor)

		# Cast elements in args to return type
		for val in args:
			if not isinstance(val, scalarTypes):
				return None
			return retType(args)
			