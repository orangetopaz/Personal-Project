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

var IN = matrix(784, 1)
var W = matrix(10, 784)
var B = matrix(10, 1)
var A = matrix(10, 1)
var Z = matrix(10, 1)
var Y = matrix(10, 1)
var Wmod = matrix(10, 784)
var Bmod = matrix(10, 1)
var INmod = matrix(784, 1)

func load_image(imageN: int = 0):
	for pixel in range(784): IN[pixel][0] = images[imageN*784.0+pixel]/255.0

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	# initialize weights & biases<(unnecassary)
	for i in range(len(W)): B[i][0] = rng.randf_range(-0.01, 0.01); for j in range(len(W[i])): W[i][j] = rng.randf_range(-sqrt(1.0/784.0), sqrt(1.0/784.0))
	load_image(0)
	#print(IN)
	#print(transpose(IN))
	#print(matrix_dupe_down(transpose(IN), 3))



# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	# foreprop
	for image in epoch_size:
		load_image(image)
		Z = matrix_add(matrix_multiply(W, IN), B)
		A = matrix_σ(Z)  # could be ReLU if I figure out how to get a cost function to work with it
		
		# get ideal
		Y = matrix(10, 1) # resets ideal
		Wmod = matrix(10, 784)
		Bmod = matrix(10, 1)
		INmod = matrix(784, 1)
		Y[labels[image]][0] = 1
		#for i in range(10): print(A[i][0], " ", Y[i])
		#print(labels[image])
		#print()
		
		# derrivs
		var dA: Array = matrix_sub(A, Y)  # this gets the derrivative for cross entropy cost/loss, in relation to 
		var delt: Array = hadamard_multiply(matrix_dσ(Z), dA)  #scal(matrix_sub(A, Y), 2))) <- for MSE cost
		Wmod = matrix_multiply(delt, transpose(IN)) #hadamard_multiply(matrix_dupe_down(transpose(IN), 10), matrix_dupe_across(delt, 784))
		Bmod = delt
		#INmod = # each N collumn in weight matrix corrisponds to the weights applied to that N previous neuron
	
	W = matrix_sub(W, scal(scal(Wmod, 1/32), LR))  # same as doing Wmod/32, gets the average from the mods
	B = matrix_sub(B, scal(scal(Bmod, 1/32), LR))  # same as doing Bmod/32, gets the average from the mods
	
	for i in range(10): print(A[i][0], " ", Y[i])
	print(labels[epoch_size-1])
	print()
	print(-sum(hadamard_multiply(Y, matrix_ln(A))))  # all the equations say log, not ln, but with machine learning its almost always ln
	print("\n\n\n")
