extends base

class_name NN

var a = []; var b = []; var w = []; var z = []; var y = []
var currentSet: String; var currentImageN: int; var currentSetLabels: Array
var stepSize = 1

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
	y = a.slice(1)  # I could do b, but that has values in it already

func setSet(sett: String) -> void:
	currentSet = sett
func resetImageN(sett: int = 0) -> void:
	currentImageN = sett
func loadImage(imageN: int = currentImageN):
	a[0] = data[currentSet].slice(imageN*784, (imageN+1)*784)
	currentImageN += 1
func currentImageCount():
	return len(data[currentSet])/784

func forepropigate():
	for i in range(1, len(a)):
		for j in len(a[i]):
			z[i][j] = b[i][j]
			for k in len(a[i+1]):
				z[i][j] += a[i][k]*w[i][j][k]  # i=layer, j=neuron ending, k=start neuron
			a[i][j] = σ(z[i][j])

func finalIdealOut(size: int, indice: int = currentSetLabels[currentImageN]):
	var out = []
	for i in size:
		out.append(0)
	out[indice] = 1
	return out

func cost(idealOut: Array) -> float:
	var outCost: float = 0
	for i in len(a[len(a)]): 
		outCost += (a[len(a)][i] - idealOut[i])**2
	return outCost

func sensitivities(initIdealOut: Array) -> Dictionary:
	var sensitivities: Dictionary = {"weights": [], "biases": [], "inputNeuron": []}
	var last2terms  # since these stay the same for weights, biases, and last terms, I don't need to calculate it every k cycle
	y[len(y)] = finalIdealOut(len(a[len(a)]), currentSetLabels[currentImageN])
	sensitivities["weights"] = w
	sensitivities["biases"] = b
	sensitivities["inputNeuron"] = a
	# starts from the last layer cuz it's *back*propigation - start at the front and move back. also we know the ideals for the finalout, so we move back from there
	for L in range(len(w), 0):  # don't need to do -1 cuz already did that in the declaration of w. coulda used b but w gives me the k loop easier
		for j in w[L]:
			last2terms = dσ(z[L][j]) * 2*(a[L][j]-y[L][j])  # keeping it out of the k cycle
			sensitivities["biases"][L+1][j] = last2terms  # technicaly there's a 1* before that, but not conna deal with that
			sensitivities["inputNeuron"][L][j] = 0
			for k in w[L][j]:
				sensitivities["weights"][L][j][k] = a[L-1][k] * last2terms
				if L > 0:
					sensitivities["inputNeuron"][L-1][k] += w[L][j][k] * last2terms
				else:
					sensitivities["inputNeuron"][L-1][k] = 0  # don't need to loop every j cycle, fix maybe
			# set the ideal value for the next layer
			y[L][j] = a[L][j]+sensitivities["inputNeuron"][L][j]*stepSize  # I believe this is correct, and making it proportional helps make sure you don't overshoot
	return sensitivities

"""
-∇Co = the negative/descent gradient of the cost, and I believe that 
the sensitivity function above gives ∇Co. However, I need ∇C, which means 
averaging the sensitivities over all training examples. In any case, even if 
I forgot to add the - in there somewhere, it just means the output set to 0
is the choice instead of the ones set to 1.
"""

func testAndFindAvgMods():
	var mods: Dictionary = {"weights": [], "biases": []}
	var currentSensitivities: Dictionary
	var currentImageCount = currentImageCount()
	loadImage()
	forepropigate()
	mods = sensitivities(finalIdealOut(len(a[len(a)])))
	for i in range(1, currentImageCount):
		loadImage()
		forepropigate()
		currentSensitivities = sensitivities(finalIdealOut(len(a[len(a)])))
		for L in len(w):
			for j in len(w[L]):
				mods["biases"][L][j] += sensitivities(finalIdealOut(len(a[len(a)])))["biases"][L][j]
				for k in w[L][j]:
					mods["weights"][L][j][k] += sensitivities(finalIdealOut(len(a[len(a)])))["weights"][L][j][k]
	for L in len(w):
		for j in len(w[L]):
			mods["biases"][L][j] /= currentImageCount
			for k in w[L][j]:
				mods["weights"][L][j][k] /= currentImageCount
	return mods

func backpropigate():
	var avgMods = testAndFindAvgMods()
	for L in len(w):
		for j in len(w[L]):
			b[L][j] += avgMods["biases"][L][j]*stepSize
			for k in w[L][j]:
				w[L][j][k] += avgMods["weights"][L][j][k]*stepSize
