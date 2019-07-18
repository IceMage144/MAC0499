extends Control

const CityScene = preload("res://Maps/City/City.tscn")
const CreditsScene = preload("res://Menus/Credits.tscn")

var debug_mode = false
var focused_button_id = 0

onready var buttons = $MarginContainer/VBoxContainer.get_children()

func _ready():
	for i in range(len(self.buttons)):
		self.buttons[i].connect("mouse_entered", self, "_on_button_mouse_entered", [i])

func _process(delta):
	var focused_button = self.buttons[self.focused_button_id]
	if not focused_button.is_hovered():
		focused_button.release_focus()
		focused_button.grab_focus()
	if Input.is_action_just_pressed("ui_down") and self.focused_button_id < len(self.buttons) - 1:
		self.focused_button_id += 1
		while self.buttons[self.focused_button_id].disabled and self.focused_button_id < len(self.buttons) - 1:
			self.focused_button_id += 1
	if Input.is_action_just_pressed("ui_up") and self.focused_button_id > 0:
		self.focused_button_id -= 1
		while self.buttons[self.focused_button_id].disabled and self.focused_button_id > 0:
			self.focused_button_id -= 1

func _on_button_mouse_entered(button_id):
	if not self.buttons[button_id].disabled:
		self.buttons[self.focused_button_id].release_focus()
		self.focused_button_id = button_id


func _on_NewGame_pressed():
	var main = global.find_entity("main")
	main.change_map(CityScene)


func _on_Credits_pressed():
	var main = global.find_entity("main")
	main.change_map(CreditsScene)


func _on_Quit_pressed():
	get_tree().quit()
