extends base

class_name NN

var nn: Dictionary = {"a": [], "w": [], "b": [], "z": [], "y": []}
var modsPerImage: Dictionary = {"a": [], "w": [], "b": []}
var totalMods: Dictionary = {"a": [], "w": [], "b": []}
var layers = [784, 16, 16, 10]
var LR = 1  # LR = learning rate, I like to call it step/stride size, cuz it's how much you move in the 13,002 dimentions downhill
# Called when the node enters the scene tree for the first time.
func _init() -> void:
	print("Initialized")
	for L in len(layers):  # initializing the arrays
		nn["a"].append([])
		totalMods["a"].append([])
		modsPerImage["a"].append([])
		
		nn["w"].append([])
		totalMods["w"].append([])
		modsPerImage["w"].append([])
		
		nn["b"].append([])
		totalMods["b"].append([])
		modsPerImage["b"].append([])
		
		nn["z"].append([])
		nn["y"].append([])
		for j in layers[L]:
			nn["a"][L].append(0.0)
			totalMods["a"][L].append(0.0)
			modsPerImage["a"][L].append(0.0)
			
			nn["w"][L].append([])
			totalMods["w"][L].append([])
			modsPerImage["w"][L].append([])
			if L > 0:
				nn["b"][L].append(rng.randf_range(-30, 30))
				totalMods["b"][L].append(0.0)
				modsPerImage["b"][L].append(0.0)
				
				nn["z"][L].append(0.0)
				nn["y"][L].append(0.0)
				for k in layers[L-1]:
					nn["w"][L][j].append(rng.randf_range(-5, 5))
					totalMods["w"][L][j].append(0.0)
					modsPerImage["w"][L][j].append(0.0)
	
	#print(nn["a"])
	#print(nn["y"])
	for reps in 100:
		for image in 100: #len(data["test_images"])/784:
			for val in len(data["test_images"].slice(image*784, (image+1)*784)):  # reset the image for 0-1 not 256 vals
				nn["a"][0][val] = data["test_images"].slice(image*784, (image+1)*784)[val]/255.0  # load the next image
			#print(nn["a"])
			for L in len(layers):  # forward propigate
				for j in layers[L]:
					if L > 0:
						for k in layers[L-1]:
							nn["z"][L][j] += nn["a"][L-1][k] * nn["w"][L][j][k]
						nn["z"][L][j] += nn["b"][L][j]
						#print('\n', nn["z"][L][j])
						nn["a"][L][j] = σ(nn["z"][L][j])  #should replace any weird values in there at the beginning
						#print(nn["z"][L][j])
			#print(nn["a"])
			#print(nn["z"])
			for i in len(nn["y"][-1]): nn["y"][-1][i] = 0;
			nn["y"][-1][data["test_labels"][image]] = 1
			#print(nn["y"][-1])
			for L in range(len(layers)-1, 0, -1):  # backpropigate to find mods for this image
				# technicaly this loop will end at 1, becuase i don't need the modifications for layer 0
				for j in layers[L]:
					# I need the extra speed, soz
					var delta: float = dσ(nn["z"][L][j]) * 2*(nn["a"][L][j]-nn["y"][L][j])
					# keeping the 1 for posterity, as it is the partial derrivitive of the cost in relation to bias, because the coefficient is 1
					modsPerImage["b"][L][j] = 1 * delta  #dσ(nn["z"][L][j]) * 2*(nn["a"][L][j]-nn["y"][L][j])
					for k in layers[L-1]:
						modsPerImage["w"][L][j][k] = nn["a"][L-1][k] * delta  #dσ(nn["z"][L][j]) * 2*(nn["a"][L][j]-nn["y"][L][j])
						# sum becuase the effect of a[L-1] is distributed accross different inputs to the cost
						modsPerImage["a"][L-1][j] += nn["w"][L][j][k] * delta  #dσ(nn["z"][L][j]) * 2*(nn["a"][L][j]-nn["y"][L][j])
					if L > 1:
						#print(nn["y"])
						nn["y"][L-1][j] = nn["a"][L-1][j] + (modsPerImage["a"][L-1][j]*LR)
						#print(nn["y"])
					# the ideal value for neuron (L, j) is its current value combined with how much it should change for this image (sensitivity of the cost to a change in this value and the learning rate/step size)
			#print("modsPerImage[\"a\"]: ", modsPerImage["a"])
			#print("modsPerImage[\"b\"]: ", modsPerImage["b"])
			
			for L in range(1, len(layers), 1):  # add image mods to the total mods
				# using a range starting at 1 here so it can avoid dealing with and doing the calculations for the input layer
				for j in layers[L]:
					totalMods["b"][L][j] += modsPerImage["b"][L][j]
					modsPerImage["b"][L][j] = 0.0
					for k in layers[L-1]:
						totalMods["w"][L][j][k] += modsPerImage["w"][L][j][k]
						modsPerImage["w"][L][j][k] = 0.0
						modsPerImage["a"][L-1][k] = 0.0
			
			#print(nn["a"])
			#print()
			#print()
			#print()
			print(nn["y"][-1])
			print(nn["a"][-1])
			print("Image N: ", image, " Label: ", data["test_labels"][image]) #, "Cost: ", (nn["a"]-nn["y"])**2)
		
		print("Out: ", nn["a"][-1], "\n     ", nn["y"][-1])
		
		for L in range(1, len(layers)):
			# apply & reset updates to weights and biases
			for j in range(layers[L]):
				nn["b"][L][j] -= (LR * totalMods["b"][L][j])  # negative becuase minimizing 
				totalMods["b"][L][j] = 0.0
				for k in range(layers[L-1]):
					nn["w"][L][j][k] -= (LR * totalMods["w"][L][j][k])
					totalMods["w"][L][j][k] = 0.0
					

		
		
	for L in range(1, len(layers), 1):  # average the totalmods
		# same applies here
		for j in layers[L]:
			totalMods["b"][L][j] /= (len(data["test_images"])/784)
			for k in layers[L-1]:
				totalMods["w"][L][j][k] /= (len(data["test_images"])/784)


##Chat GPT suggestions:
"""
Additional Suggestions
Normalize Initialization of Weights and Biases: Instead of using arbitrary 
ranges like [-5, 5] or [-30, 30], consider using a Gaussian distribution scaled 
by the layer size:

rng.randf_range(-sqrt(1/layers[L-1]), sqrt(1/layers[L-1]))

Implement Mini-Batch Gradient Descent: Instead of accumulating modifications over 
100 images, divide your dataset into smaller mini-batches (e.g., 32 images) to 
compute and apply updates more frequently. This improves training dynamics.

Adjust Learning Rate: A learning rate (LR) of 1 is likely too high. Start with a 
smaller value, such as 0.01, and tune it based on performance.

Monitor Cost (Loss) Over Epochs: Compute and print the loss after each epoch 
(or batch) to ensure the model is converging:

var cost = 0.0
for i in range(len(nn["y"][-1])):
	cost += (nn["a"][-1][i] - nn["y"][-1][i]) ** 2
cost /= 2  # Mean squared error
print("Epoch cost: ", cost)
Debugging Tips
Print Modifications: After each backpropagation step, print out small parts of totalMods to ensure they are being computed correctly.
Use Validation Data: Evaluate on a separate validation set after each epoch to check generalization.
Experiment with Smaller Subsets: Train on a small subset (e.g., 100 images) to quickly observe the effect of updates.
"""


#func _ready():
	#if has_node(".."):
		#var parent = get_parent()
		#for child in parent.get_children():
			#print("Child:", child.name)
