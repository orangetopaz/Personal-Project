extends base

class_name NN

var nn: Dictionary = {"a": [], "w": [], "b": [], "z": [], "y": []}
var layers = [784, 16, 16, 10]
# Called when the node enters the scene tree for the first time.
func _init() -> void:
	print("Initialized")
	for L in len(layers):  # initializing the arrays
		nn["a"].append([])
		nn["w"].append([])
		nn["b"].append([])
		for j in layers[L]:
			nn["a"][L].append(0.0)
			nn["w"][L].append([])
			if L > 0:
				nn["b"][L].append(rng.randf_range(-30, 30))
				for k in layers[L-1]:
					nn["w"][L][j].append(rng.randf_range(-5, 5))
	
	nn["z"] = nn["a"]  # creating the supplemental arrays
	nn["y"] = nn["a"]
	#print(nn["a"])
	
	for image in len(data["test_images"])/784:
		nn["y"][-1][data["test_labels"][image]] = 1
		#print(nn["y"][-1])
		for val in len(data["test_images"].slice(image*784, (image+1)*784)):  # reset the image for 0-1 not 256 vals
			nn["a"][0][val] = data["test_images"].slice(image*784, (image+1)*784)[val]/255.0  # load the next image
		#print(nn["a"])
		for L in len(layers):  # forward propigate
			for j in layers[L]:
				if L > 0:
					for k in layers[L-1]:
						nn["z"][L][j] += nn["a"][L-1][k] * nn["w"][L][j][k]
					nn["z"][L][j] += nn["b"][L][j]
					nn["a"][L][j] = σ(nn["z"][L][j])  #should replace any weird values in there at the beginning
		
		print(nn["a"])
		break

func _ready():
	if has_node(".."):
		var parent = get_parent()
		for child in parent.get_children():
			print("Child:", child.name)
