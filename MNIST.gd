extends Node2D

var net = NN.new([784, 16, 16, 10], 5, 30)
# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	net.setSet("train_images")
	net.currentSetLabels = net.data["train_labels"]
	print('[')
	


# Called every frame. 'delta' is the elapsed time since the previous frame.
var iter:int = 0
func _process(delta: float) -> void:
	net.trainAStep()
	#print(net.a[-1])
	#print(net.w[-1])
	#print('(', iter, " , ", net.b[-1][0], ')', " , ")
	print('(', iter, ", ", net.cost(), ')', ", ")
	iter +=1
