extends Line2D


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	pass # Replace with function body.


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass

func _on_graph_data_changed(old_val: Variant) -> void:
	print("data_change")
	points = old_val
