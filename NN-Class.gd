extends base

class_name NN

var a = []; var aTemplate = []; var b = []; var bTemplate = []; var w = []; var wTemplate = []; var z = []; var y = []
var currentSet: String; var currentImageN: int; var currentSetLabels: Array
var stepSize = 1

func σ(x: float):
	return sigmoid(x, 1, 0.3)

func dσ(x: float):
	return sigmoid_deriv(x, 1, 0.3)

func _init(layers: Array, w_spread: float = 5, b_spread = 30) -> void:
	for i in len(layers):
		a.append([])
		aTemplate.append([])
		if i != 0:
			b.append([])
			bTemplate.append([])
			w.append([])
			wTemplate.append([])
		#z.append([])
		for j in layers[i]:
			a[i].append(0.0)
			aTemplate[i].append(0.0)
			#z.append(0)
			if i != 0:
				b[i-1].append(rng.randf_range(-b_spread, b_spread))
				bTemplate[i-1].append(0)
				w[i-1].append([])
				wTemplate[i-1].append([])
				for k in layers[i-1]:
					w[i-1][j].append(rng.randf_range(-w_spread, w_spread))
					wTemplate[i-1][j].append(0)
	z = bTemplate
	y = bTemplate  # I could do b, but that has values in it already
	print("initialized")

func setSet(sett: String) -> void:
	currentSet = sett
func resetImageN(sett: int = 0) -> void:
	currentImageN = sett

func printIn(localx:int, localy:int):
	var pr = []
	for i in a[0]:
		pr.append(round(a[0][i]))
	for j in localy:
		for i in localx:
			pr.slice((j*localx), (j*localx)+localx)
		
		print(pr[j])

func loadImage(imageN: int = currentImageN):
	if (imageN + 1) * 784 > data[currentSet].size():
		print("Incomplete image data, stopping at image index: ", imageN)
		return
	var imageArray = data[currentSet].slice(imageN * 784, (imageN + 1) * 784)
	#print("Loading image ", imageN)
	for j in range(len(a[0])):  # Ensure this loops through valid indices
		# setting it to a value from 0-1 instead of 0-255
		a[0][j] = imageArray[j] / 255.0  # screw this .0 in particular, it screwed me up for an hour
	if imageN == currentImageN:
		currentImageN += 1

func currentImageCount():
	return len(data[currentSet])/784

func forepropigate():
	z = bTemplate
	#print(image)
	for L in len(w):
		#print("L: ", L)
		for j in len(w[L]):
			#print("j: ", j)
			for k in len(a[L]):
				#print("k: ", k)
				z[L][j] += a[L][k] * w[L][j][k]  # using w already accounts for the L-1 I would usualy have to do in a[L]
			z[L][j] += b[L][j]
			a[L+1][j] = σ(z[L][j])
			#print(a[L][j])
	#print(z)
	#print(a)

func genIdealOut(size: int, indice: int = currentSetLabels[currentImageN-1]):
	var out = []
	for i in size:
		out.append(0)
	out[indice] = 1
	return out

func cost(idealOut: Array) -> float:
	var outCost: float = 0
	for i in len(a[-1]): 
		outCost += (a[-1][i] - idealOut[i])**2
	return outCost

func sensitivities(initIdealOut: Array = genIdealOut(len(a[-1]))) -> Dictionary:
	#print(genIdealOut(len(a[-1])))
	var sensitivities: Dictionary = {"weights": [], "biases": [], "inputNeuron": []}
	var last2terms: float = 0.0 # since these stay the same for weights, biases, and last terms, I don't need to calculate it every k cycle
	y[-1] = initIdealOut
	sensitivities["weights"] = wTemplate
	sensitivities["biases"] = bTemplate
	sensitivities["inputNeuron"] = aTemplate
	# starts from the last layer cuz it's *back*propigation - start at the front and move back. also we know the ideals for the finalout, so we move back from there
	for L in range(len(w)-1, -1, -1):  # don't need to do -1 cuz already did that in the declaration of w. coulda used b but w gives me the k loop easier
		#print("YES!")
		for j in len(w[L]):
			last2terms = dσ(z[L][j]) * 2*(a[L+1][j]-y[L][j])  # keeping it out of the k cycle
			#print(dσ(z[L][j]))
			#print(2*(a[L+1][j]-y[L][j]))
			#print(dσ(z[L][j]) * 2*(a[L+1][j]-y[L][j]))
			#print(last2terms)
			sensitivities["biases"][L][j] = last2terms  # technicaly there's a 1* before that, but not conna deal with that
			sensitivities["inputNeuron"][L+1][j] = 0
			for k in len(w[L][j]):
				sensitivities["weights"][L][j][k] = a[L][k] * last2terms
				sensitivities["inputNeuron"][L][k] += w[L][j][k] * last2terms
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

func testAndFindAvgMods(imageCount: int = currentImageCount()):
	var mods: Dictionary = {"weights": [], "biases": []}
	var currentSensitivities: Dictionary
	loadImage()
	#print(a)
	forepropigate()
	#print(a)
	mods["weights"] = sensitivities()["weights"]
	#print(mods["weights"])
	mods["biases"] = sensitivities()["biases"]
	print(mods["biases"][-1])
	print(b[-1])
	for i in imageCount:
		#print(i+1)
		loadImage()
		#print(i+1)
		forepropigate()
		#print(a)
		#printIn(28, 28)
		#print("forepropigated: ", currentImageN, ", out: ", a[-1])
		currentSensitivities["weights"] = sensitivities(genIdealOut(len(a[-1])))["weights"]
		#print(mods["weights"])
		currentSensitivities["biases"] = sensitivities(genIdealOut(len(a[-1])))["biases"]
		#print(mods["biases"])
		for L in len(w):
			for j in len(w[L]):
				mods["biases"][L][j] += currentSensitivities["biases"][L][j]
				for k in len(w[L][j]):
					mods["weights"][L][j][k] += currentSensitivities["weights"][L][j][k]
		#if i%1000 == 0:
			#print(i)
		print(i)
	for L in len(w):
		for j in len(w[L]):
			mods["biases"][L][j] /= imageCount
			for k in len(w[L][j]):
				mods["weights"][L][j][k] /= imageCount
	print(mods["biases"])
	#print(mods["weights"])
	return mods

func backpropigate():
	var avgMods = testAndFindAvgMods()
	for L in len(w):
		for j in len(w[L]):
			b[L][j] += avgMods["biases"][L][j]*stepSize
			for k in w[L][j]:
				w[L][j][k] += avgMods["weights"][L][j][k]*stepSize
# these are the same, just for naming ease
func trainAStep():
	var avgMods = testAndFindAvgMods(1000)
	print("found!")
	for L in len(w):
		for j in len(w[L]):
			b[L][j] += avgMods["biases"][L][j]*stepSize
			for k in len(w[L][j]):
				w[L][j][k] += avgMods["weights"][L][j][k]*stepSize


func test():
	pass
