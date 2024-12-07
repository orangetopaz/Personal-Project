extends base

class_name NN

var a = []; var b = []; var w = []; var z = []
var currentSet: String; var currentImage: Array; var currentImageN: int; var currentImageLabels: Array

func σ(x: float):
	sigmoid(x, 1, 0.3)

func dσ(x: float):
	sigmoid_deriv(x, 1, 0.3)

func _init(layers: Array, w_spread: float, b_spread) -> void:
	for i in len(layers):
		a.append([])
		b.append([])
		w.append([])
		#z.append([])
		for j in layers[i]:
			a.append(0)
			w.append([])
			#z.append(0)
			if i != 0:
				b.append(rng.randf_range(-b_spread, b_spread))
				for k in layers[i-1]:
					w[i][j].append(rng.randf_range(-w_spread, w_spread))
			"""else:
				w[i][j].append(0)"""
	z = a

func setSet(sett: String) -> void:
	currentSet = sett
func resetImageN() -> void:
	currentImageN = 0
func loadImage(imageN: int = currentImageN):
	a[0] = data[currentSet].slice(imageN*784, (imageN+1)*784)

func propigate_forward():
	for i in range(1, len(a)):
		for j in len(a[i]):
			z[i][j] = b[i][j]
			for k in len(a[i+1]):
				z[i][j] += a[i][k]*w[i][j][k]  # i=layer, j=neuron ending, k=start neuron
			a[i][j] = σ(z[i][j])

func idealOut():
	pass

func cost(idealOut: Array) -> float:
	var outCost: float = 0
	for i in len(a[len(a)]): 
		outCost += (a[len(a)][i] - idealOut[i])**2
	return outCost

func sensitivities():
	var sensitivities: Dictionary = {"weights": [], "biases": [], "inNode": []}
	var last2terms  # since these stay the same for weights, biases, and last terms, I don't need to calculate it every k cycle
	sensitivities["weights"] = w
	sensitivities["biases"] = b
	sensitivities["inNode"] = a
	for L in len(w):
		for j in w[L]:
			last2terms = dσ(z[L][j]) * 2*(a[L][j]-y[j])  # keeping it out of the k cycle
			sensitivities["biases"][L][j] = last2terms  # technicaly there's a 1* before that, but not conna deal with that
			for k in w[L][j]:
				sensitivities["weights"][L][j][k] = a[L-1][k] * last2terms
				sensitivities["inNode"][L-1][j]
				
