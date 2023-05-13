import sympy as sp


class Function(dict):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		function = kwargs.get('f', None)
		symbols = kwargs.get('symbols', None)

		self._symbols = symbols
		self._function = sp.sympify(function)

		self.clear()


	def __missing__(self, key):
		value = self._function.subs(list(zip(self._symbols, key)))
		self[key] = value
		return float(value)


	@property
	def times_called(self):
		keys = self.keys()
		return len(keys)


	@property
	def f(self):
		return self._function


class FunctionWrapper():
	def __init__(self, *args, **kwargs):
		function = kwargs.get('function', None)
		self._derivatives = {}
		symbols = kwargs.get('symbols', None)

		self._symbols = symbols
		self._function = Function(f=function, symbols=symbols)


	@property
	def function(self):
		return self._function


	def at(self, x, r):
		"""
		Computes the value of the function at a point x.
		"""
		if not self._function.f:
			raise ValueError("No function defined")
		return self._function[*x, r]


	def clear_cache(self):
		self._function.clear()
		for derivative in self._derivatives.values():
			derivative.clear()
		for gradient in self._gradient.values():
			gradient.clear()
		self._gradient_points = []
