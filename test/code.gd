extends Matricies

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
var images = data["train_images"]
var labels = data["train_labels"]

const e = 2.718281828459045
var rng = RandomNumberGenerator.new()
const LR: float = 0.03
const epoch_size: int = 32

var IN = matrix(784, 1)
var W = matrix(10, 784)
var B = matrix(10, 1)
var A = matrix(10, 1)
var Z = matrix(10, 1)

func LReLU(x: float):
	return max(0.05*x, x)

func dLReLU(x: float):
	if x > 0:
		return 1
	else:
		return 0.05

func σ(x: float):
	return 1/(1+e**(-0.03*x))

func dσ(x: float):
	var part_sig: float = e**(-0.3*x)
	return (0.3*part_sig)/(1+part_sig)**2

func load_image(imageN: int = 0):
	for pixel in range(784): IN[pixel][0] = images[imageN*784+pixel]/255

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	# initialize weights & biases<(unnecassary)
	for i in range(len(W)): B[i][0] = 0.01; for j in range(len(W[i])): W[i][j] = rng.randf_range(-sqrt(1.0/784.0), sqrt(1.0/784.0))


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	# foreprop
	for image in epoch_size:
		load_image(0)
		Z = matrix_add(matrix_multiply(W, IN), B)
		# this would be a matrix function but martrix_sigmoid seems a bit specific
		for i in range(10): A[i][0] = σ(Z[i][0])  # could be ReLU if I figure out how to get a cost function to work with it
		
	
	
