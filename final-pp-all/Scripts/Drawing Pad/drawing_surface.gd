extends Node2D

@onready var _lines: Node2D = $Line2D

@export var PixelDimentions: Vector2i
var _pressed: bool = false
var _current_line: Line2D = null

func _input(event: InputEvent) -> void:
	if event == InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT:
			_pressed = event.pressed
			
			
			if _pressed:
				_current_line = Line2D.new()
				_current_line.default_color = Color.BLUE
				_current_line.width = 4
				_lines.add_child(_current_line)
				_current_line.add_point(event.position)
	
	elif event == InputEventMouseMotion && _pressed:
		_current_line.add_point(event.position)
