extends Label


# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	$".".text = "Learning Rate: " + str($"..".value)


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass


func _on_lr_slider_value_changed(value: float) -> void:
	$".".text = "Learning Rate: " + str(value)
