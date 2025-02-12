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
var tests = data["test_images"]
var testl = data["test_labels"]

var cost: float = 0.0

var IN = matrix(784, 1)
var W = matrix(10, 784)
var B = matrix(10, 1)
var A = matrix(10, 1)
var Z = matrix(10, 1)
var Y = matrix(10, 1)
var Wmod = matrix(10, 784)
var Bmod = matrix(10, 1)
var INmod = matrix(784, 1)

func load_image(train: bool, imageN: int):
	if train:
		for pixel in range(784): IN[pixel][0] = images[(imageN*784.0)+pixel]/255.0
	else:
		for pixel in range(784): IN[pixel][0] = tests[(imageN*784.0)+pixel]/255.0

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	# initialize weights & biases<(unnecassary)
	for i in range(len(W)): B[i][0] = rng.randf_range(-0.01, 0.01); for j in range(len(W[i])): W[i][j] = rng.randf_range(-sqrt(1.0/784.0), sqrt(1.0/784.0))
	#load_image(train0)
	#print(IN)
	#print(transpose(IN))
	#print(matrix_dupe_down(transpose(IN), 3))
	#get_node(".").paused = true

var epochs: int = 0
var corrects: int = 0
var testCorrects: int = 0
var accuracy: float = 0.0
var testAccuracy: float = 0.0

var base_image: int = 0

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	#print($"..".paused)
	if !$"..".paused:
		epoch()
		

func foreprop():
	Z = matrix_add(matrix_multiply(W, IN), B)
	A = matrix_σ(Z)  # could be ReLU if I figure out how to get a cost function to work with it

func epoch():
	epochs += 1
	base_image = rng.randi_range(0, (len(images)/784)-epoch_size)
	#A = matrix(10, 1)
	#Z = matrix(10, 1)
	#Y = matrix(10, 1)
	IN = matrix(784, 1)
	Wmod = matrix(10, 784)
	Bmod = matrix(10, 1)
	cost = 0
	for image in range(epoch_size):
		load_image(true, base_image+image)
		foreprop()
		
		# get ideal
		Y = matrix(10, 1) # resets ideal
		Y[labels[base_image+image]][0] = 1 
		
		# derrivs
		var dA: Array = matrix_sub(A, Y)  # this gets the derrivative for cross entropy cost/loss, in relation to 
		var delt: Array = hadamard_multiply(matrix_dσ(Z), dA)  #scal(matrix_sub(A, Y), 2))) <- for MSE cost
		Wmod = matrix_add(Wmod, matrix_multiply(delt, transpose(IN))) #hadamard_multiply(matrix_dupe_down(transpose(IN), 10), matrix_dupe_across(delt, 784))
		Bmod = matrix_add(Bmod, delt)  
		cost += -sum(hadamard_multiply(Y, matrix_ln(A)))
	
	W = matrix_sub(W, scal(scal(Wmod, 1.0/epoch_size), LR))  # same as doing Wmod/32, gets the average from the mods
	B = matrix_sub(B, scal(scal(Bmod, 1.0/epoch_size), LR))  # same as doing Bmod/32, gets the average from the mods
	
	for i in range(10): print(A[i][0], " ", Y[i])
	
	var lastimagelabelindex: int = base_image+epoch_size-1
	print(labels[lastimagelabelindex])
	print()
	
	
	cost /= epoch_size
	print("Cost of Epoch: ", cost)  # all the equations say log, not ln, but with machine learning its almost always ln
	print("Learning Rate: ", LR)
	if MatrixMaxIndex(A)[0] == labels[lastimagelabelindex]:
		corrects += 1
	
	var testImageindex: int = rng.randi_range(0, len(testl)-2)
	load_image(false, testImageindex)
	foreprop()
	if MatrixMaxIndex(A)[0] == testl[testImageindex]:
		testCorrects += 1
	
	print("# Correct: ", corrects)
	print("# Non-trained Correct: ", testCorrects)
	print("# Epochs: ", epochs)
	accuracy = float(corrects)/float(epochs)
	print("Accuracy: ", accuracy)
	testAccuracy = float(testCorrects)/float(epochs)
	print("Accuracy with non-training images: ", testAccuracy)
	print("\n\n\n")
