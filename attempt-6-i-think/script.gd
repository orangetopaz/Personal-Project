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
const LR: float = 0.03
@onready var cost: float = 0.0

var images = data["train_images"]
var labels = data["train_labels"]
var input_size: int = 784
var input: Array = empty_array(784)

var layers: Array = [16, 16, 10]
var a: Array = []
var w: Array = []
var b: Array = []
var z: Array = []
var y: Array = []
var wderivs: Array = []
var bderivs: Array = []

func initialize():
	var previous_layer_length: float = input_size  # float because needed for other calcs. I could do the (float) operation (evaluate as float) but this is much easier
	for L in range(len(layers)):
		a.append([])
		w.append([])
		b.append([])
		z.append([])
		y.append([])
		wderivs.append([])
		bderivs.append([])
		for j in range(layers[L]):
			a[L].append(0.0)
			w[L].append([])
			b[L].append(0.001)
			z[L].append(0.0)
			y[L].append(0.0)
			wderivs[L].append([])
			bderivs[L].append(0.0)
			for k in range(previous_layer_length):
				w[L][j].append(rng.randf_range(-sqrt(1/previous_layer_length), sqrt(1/previous_layer_length)))
				wderivs[L][j].append(0.0)
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

func snapped_array(arr: Array, nearest: float) -> Array:
	var out: Array = []
	for term in arr:
		out.append(snapped(term, nearest))
	return out

func set_input(imageN: int = 0, refined = false):
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

func sum_array(arr: Array):
	var out: float = 0.0
	for i in range(len(arr)):
		out += arr[i]
	return out

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	initialize()


var epoch_size: int = 32  # 32 cuz I'm not running through the whole stack every step. maybe if it ends up being a problem in the future
@onready var epochs:int = 0
# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	print("Epoch: ", epochs-1)
	print("\nNN output last: \n", snapped_array(a[-1], 0.01))
	print("Label last: ", labels[epoch_size-1], "\n")
	print("Cost for Last Image: ", cost, "\n\n\n")
	for L in range(len(y)):  # resets it all, maybe find a more efficient way of doing but idk
		for j in range(len(y[L])):
			y[L][j] = 0.0
			bderivs[L][j] = 0.0
			for k in range(len(w[L][j])):
				wderivs[L][j][k] = 0.0
	
	
	for image in range(epoch_size):
		#var imageN: int = 0  # rng.randi_range(0, 59999)
		set_input(image, false)
		foreprop()
		cost = calc_cost(labels[image])
		#print(cost)
		
		# this runs throught the neural network and does the back propigation calculations
		var next_in: Array = input
		for L in range(len(y)-1, -1, -1):  # the len(y)-1 makes it start on the last, don't ask. the 0 makes it stop at 1, idk
			if L == 0:
				next_in = input
			else:
				next_in = a[L-1]
			
			#this runs throught the layer's length, starting from the last layer
			for j in range(len(y[L])):
				var delt: float = dσ(z[L][j]) * cost  # calculating seperately for efficeincy 
				# (delt means delta but deltas already a variable)
				
				
				bderivs[L][j] += 1 * delt  # gets the ∂ of each 
				
				#this runs through the length of the previous layer that feeds into the one the j loop us running though
				for k in range(len(next_in)):
					#print(k)
					#print("len: ", len(next_in))
					if L != 0:
						y[L-1][k] += w[L][j][k]
					wderivs[L][j][k] += next_in[k] * delt
				for k in range(len(next_in)):
					if L != 0:
						y[L-1][k] *= delt
			for k in range(len(y[L-1])):  # fix if possible, try to integrate into the rest of the loops
				y[L-1][k] = a[L-1][k]-y[L-1][k]
		#print(wderivs)
		#print(bderivs)
	for L in range(len(w)):
			for j in range(len(w[L])):
				b[L][j] -= (bderivs[L][j]/epoch_size)*LR
				for k in range(len(w[L][j])):
					w[L][j][k] -= (wderivs[L][j][k]/epoch_size)*LR
	
	epochs += 1
