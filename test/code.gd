extends Matricies


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	var inp = matrix(8,1)
	var weights = matrix(5,8)
	var biases = matrix(5, 1)
	for i in range(len(inp)): inp[i][0] = i
	for i in range(len(weights)): for j in range(len(weights[i])): weights[i][j] = abs(j-i)
	for i in range(len(biases)): biases[i][0] = -i+len(biases)
	mprint(inp)
	print()
	mprint(weights)
	print()
	mprint(biases)
	print()
	var out = matrix_add(matrix_multiply(weights, inp), biases)
	mprint(out)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
