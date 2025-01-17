extends Node2D

@export var Title: String
@export var XAxisName: String
@export var YAxisName: String

#emit a signal when data var is changed
signal data_changed(old_val)
@export var Data: PackedVector2Array:
	set(val):
		if val != Data:
			var old_val = Data
			Data = val
			data_changed.emit(old_val)

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	$XAxisLable.text = XAxisName
	$YAxisLable.text = YAxisName
	$Title.text = Title


# Called every frame. 'delta' is the elapsed time since the previous frame.
func _process(delta: float) -> void:
	pass
