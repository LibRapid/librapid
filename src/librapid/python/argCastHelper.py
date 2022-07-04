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
		if isinstance(args[0], (list, tuple)):
			if len(args) > 1:
				return None
			return cast(args[0], scalarTypes, retType)

		# Cast elements in args to return type
		for val in args:
			if not isinstance(val, scalarTypes):
				return None
			return retType(args)