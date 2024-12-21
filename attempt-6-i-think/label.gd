extends Label


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass #print(get_parent().cost)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	$".".text = "Cost: " + str($"..".cost)  # .. = the parent node, same as get_parent().cost
