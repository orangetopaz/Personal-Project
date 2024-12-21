extends Node2D

var rng = RandomNumberGenerator.new()
var pos_in_dimensions: Array = []
var ideal_pos: Array = []
var LR: float = 0.03
@onready var cost: float = 0.0

func calculate_cost(pos: Array, ideal: Array) -> float:
	var cost: float = 0.0
	for i in range(len(pos)):
		cost += (pos[i]-ideal[i])**2  # 1 is the ideal value for all the vars to set to, i can change it if needed on a per-var basis
	return cost

func per_axis_derivate(pos: Array, ideal: Array) -> Array:  # pd = partial derivative. I would use ∂ but it doesn't let me
	var pds: Array = []
	for i in range(len(pos)):
		pds.append(2*(pos[i]-ideal[i]))  # derivative of the cost function (the other sections of adding each of the other costants for the other demensions don't matter becuse derivation puts constants at 0
	return pds

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	for i in range(13002): pos_in_dimensions.append(rng.randf_range(-30, 30)); ideal_pos.append(i);
	

# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	cost = calculate_cost(pos_in_dimensions, ideal_pos)
	print(cost)
	var pds = per_axis_derivate(pos_in_dimensions, ideal_pos)
	for i in range(len(pos_in_dimensions)):
		pos_in_dimensions[i] -= pds[i]*LR
	#print(pos_in_dimensions)
