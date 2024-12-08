extends Node

class_name base

var rng = RandomNumberGenerator.new()
var e = 2.7182818284590452353602874713527
var π = 3.141592653589793238462643383279502884197

func sigmoid(x: float, amplitude: float = 1, wavelegnth: float = 1) -> float:
	return amplitude/(1+(e**(-x*wavelegnth)))

func sigmoid_deriv(x: float, amplitude: float = 1, wavelegnth: float = 1) -> float:
	var sig = sigmoid(x, amplitude, wavelegnth)
	return sig*(1-sig)

func σ(x: float):
	return sigmoid(x, 1, 0.3)

func dσ(x: float):
	return sigmoid_deriv(x, 1, 0.3)

func load_data() -> Dictionary:
	var t10k_images_idx3_ubyte = FileAccess.open("res://MNIST_Data/t10k-images-idx3-ubyte", FileAccess.READ)
	var t10k_labels_idx1_ubyte = FileAccess.open("res://MNIST_Data/t10k-labels-idx1-ubyte", FileAccess.READ)
	var train_images_idx3_ubyte = FileAccess.open("res://MNIST_Data/train-images-idx3-ubyte", FileAccess.READ)
	var train_labels_idx1_ubyte = FileAccess.open("res://MNIST_Data/train-labels-idx1-ubyte", FileAccess.READ)
	
	
	var test_images = t10k_images_idx3_ubyte.get_buffer(t10k_images_idx3_ubyte.get_length()).slice(16)
	var test_labels = t10k_labels_idx1_ubyte.get_buffer(t10k_labels_idx1_ubyte.get_length()).slice(8)
	var train_images = train_images_idx3_ubyte.get_buffer(train_images_idx3_ubyte.get_length()).slice(16)
	var train_labels = train_labels_idx1_ubyte.get_buffer(train_labels_idx1_ubyte.get_length()).slice(8)
	
	#print(train_labels_idx1_ubyte.get_length())
	#print(train_labels.size())
	#print(train_images.size())
	
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
