extends Node2D

var rng = RandomNumberGenerator.new()
var dimensions = []

func cost() -> float:
	var cost: float = 0.0
	for i in range(len(dimensions)):
		cost += (dimensions[i]-1)**2  # 1 is the ideal value for all the vars to set to, i can change it if needed on a per-var basis
	return cost

func pd() -> Array:  # partial derrivitive 
	var pd_dimensions: Array
	for i in range(len(dimensions)):
		pd_dimensions.append(2*(dimensions[i]-1))  # 1 is the ideal value for all the vars to set to, i can change it if needed on a per-var basis
	return pd_dimensions

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	for i in range(2): dimensions.append(rng.randf_range(-30, 30))
	


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
