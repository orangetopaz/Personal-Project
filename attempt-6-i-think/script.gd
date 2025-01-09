extends Node2D

func load_data() -> Dictionary:
	var t10k_images_idx3_ubyte = FileAccess.open("res://MNIST_Data/t10k-images-idx3-ubyte", FileAccess.READ)
	var t10k_labels_idx1_ubyte = FileAccess.open("res://MNIST_Data/t10k-labels-idx1-ubyte", FileAccess.READ)
	var train_images_idx3_ubyte = FileAccess.open("res://MNIST_Data/train-images-idx3-ubyte", FileAccess.READ)
	var train_labels_idx1_ubyte = FileAccess.open("res://MNIST_Data/train-labels-idx1-ubyte", FileAccess.READ)
	
	
	var test_images = t10k_images_idx3_ubyte.get_buffer(t10k_images_idx3_ubyte.get_length()).slice(16)
	var test_labels = t10k_labels_idx1_ubyte.get_buffer(t10k_labels_idx1_ubyte.get_length()).slice(8)
	var train_images = train_images_idx3_ubyte.get_buffer(train_images_idx3_ubyte.get_length()).slice(16)
	var train_labels = train_labels_idx1_ubyte.get_buffer(train_labels_idx1_ubyte.get_length()).slice(8)
	
	#print(len(train_labels))
	#print(len(train_images))
	
	t10k_images_idx3_ubyte.close()
	t10k_labels_idx1_ubyte.close()
	train_images_idx3_ubyte.close()
	train_labels_idx1_ubyte.close()
	
	return {
		"test_images" : test_images,
		"test_labels" : test_labels,
		"train_images" : train_images,
		"train_labels" : train_labels,
	}
var data = load_data()

const e = 2.718281828459045
var rng = RandomNumberGenerator.new()
#var pos_in_dimensions: Array = []
#var ideal_pos: Array = []
const LR: float = 0.03
@onready var cost: float = 0.0

var images = data["test_images"]
var labels = data["test_labels"]
var input_size: int = 784
var input: Array = empty_array(784)

var layers: Array = [16, 16, 10]
var a: Array = []
var w: Array = []
var b: Array = []
var z: Array = []
var y: Array = []

func per_axis_derivate(pos: Array, ideal: Array) -> Array:  # pd = partial derivative. I would use ∂ but it doesn't let me
	var pds: Array = []
	for i in range(len(pos)):
		pds.append(2*(pos[i]-ideal[i]))  # derivative of the cost function (the other sections of adding each of the other costants for the other demensions don't matter becuse derivation puts constants at 0
	return pds

func initialize():
	var previous_layer_length: float = input_size  # float because needed for other calcs. I could do the (float) operation (evaluate as float) but this is much easier
	for L in range(len(layers)):
		a.append([])
		w.append([])
		b.append([])
		z.append([])
		y.append([])
		for j in range(layers[L]):
			a[L].append(0.0)
			w[L].append([])
			b[L].append(0.0)
			z[L].append(0.0)
			y[L].append(0.0)
			for k in range(previous_layer_length):
				w[L][j].append(rng.randf_range(-sqrt(1/previous_layer_length), sqrt(1/previous_layer_length)))
		#print(previous_layer_length)
		previous_layer_length = layers[L]

func σ(x):
	return 1/(1+e**(-0.3*x))

func dσ(x):
	var part_sig: float = e**(-0.3*x)
	return (0.3*part_sig)/(1+part_sig)**2

func sum(list: Array):
	var out = 0
	for i in list:
		out += i
	return out

func map(x:float, xmin:float, xmax:float, min:float, max:float):
	return (max-min)*((x-xmin)/(xmax-xmin))+min

func empty_array(size: int) -> Array:
	var arr: Array = []
	for i in range(size):
		arr.append(0.0)
	return arr

func empty_matrix(sizex:int, sizey:int) -> Array:
	var out: Array = []
	for y in range(sizey):
		out.append(empty_array(sizex))
	return out

func set_input(imageN: int = 0, refined = true):
	input = images.slice(imageN*input_size, (imageN+1)*input_size)
	if !refined:
		for pixel in range(len(input)):
			input[pixel] = map(input[pixel], 0, 255, 0, 1)

func foreprop():
	var previous_layer: Array = input
	for L in range(len(layers)):
		for j in range(layers[L]):
			z[L][j] = b[L][j]
			for k in range(len(previous_layer)):
				z[L][j] += previous_layer[k]*w[L][j][k]
			a[L][j] = σ(z[L][j])
		previous_layer = a[L]

func calc_cost(ideal):
	var final: float
	if typeof(ideal) == 2:  # 2 = int variable type id
		for i in range(len(y[-1])): y[-1][i] = 0
		y[-1][ideal] = 1
	for i in range(len(a[-1])):
		final += (a[-1][i]-y[-1][i])**2
	return final

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	initialize()
	#print(len(w[0][0]))
	#print(len(w[0]))
	#print(a)
	#print(z)
	#print(sum(input))
	#print(w)

# Called every frame. 'delta' is the elapsed time since the previous frame.
var reps:int = 0
func _process(delta: float) -> void:
	for L in range(len(y)):  # resets it all, maybe find a more efficient way of doing but idk
		for j in range(len(y[L])):
			y[L][j] = 0.0
	#print(reps)
	# foreprop
	set_input(0, false)  # constant image set cuz I just want to see if the math works, not if it can recognize
	foreprop()
	# get/print cost
	#print(z)
	#print()
	#print(a)
	#print()
	cost = calc_cost(labels[0])
	print(cost)
	print(a[-1])
	#print()
	#print()
	#print()
	#print(a)
	for L in range(len(y)-1, 0, -1):  # the len(y)-1 makes it start on the last, don't ask. the 0 makes it stop at 1, idk
		for j in range(len(y[L])):
			for k in range(len(y[L-1])):
				y[L-1][k] += w[L][j][k] * dσ(z[L][j]) * 2*(a[L][j]-y[L][j])*LR
				w[L][j][k] -= a[L-1][k] * dσ(z[L][j]) * 2*(a[L][j]-y[L][j])*LR
				b[L][j] -= 1 * dσ(z[L][j]) * 2*(a[L][j]-y[L][j])*LR
		for k in range(len(y[L-1])):
			y[L-1][k] = a[L-1][k]-y[L-1][k]
	#print(y)
	#var previous_layer: Array = input
	#for L in range(len(y)):
		#for j in range(len(y[L])):
			#for k in range(len(previous_layer)):
				#pass
		#previous_layer = a[L]
	# get derrivative of last layers inputs (y for the last layer (but moving backwards) is the final ideal, and for previous layers it changes)
		# get ∂ to each weight (a[L-1][k] * dσ(z[L][j]) * 2(a[L][j] - y[L][j]))
		# bias (1 * dσ(z[L][j]) * 2(a[L][j] - y[L][j]))
		# previous neuron (a[L-1][j]) ∑(k): (w[L][j][k] * dσ(z[L][j]) * 2(a[L][j] - y[L][j]))
			# set y(ideal) for previous layers to current value - previously established derrivitive * LR: y[L][j] = a[L][j]-(∑(k): (w[L][j][k] * dσ(z[L][j]) * 2(a[L][j] - y[L][j])))*LR (INCORRECT THIS SUMS UP ALL THE WAYS STUFF CONNECTS TO IT, NOT ALL THE WAY IT CONNECTS TO THE NEXT LAYER!!!! PROB HAS SCREWED ME BEFORE)
	# minus each weight by it's derrivitive times the LR: w[L][j][k] -= (∂w[L][j][k])*LR:		((a[L-1][k] * dσ(z[L][j]) * 2(a[L][j] - y[L][j]))*LR)
	# bias: b[L][j] -= (∂b[L][j])*LR:		(1 * dσ(z[L][j]) * 2(a[L][j] - y[L][j]))
	
	"""This will train as 1 image at a time, eventualy I will want to get the average modifications per weight/bias and use that for the changes. this works for now"""
	
	reps += 1
