extends "res://UI/Menus/MenuBase.gd"

const CityScene = preload("res://Maps/City/City.tscn")
const CreditsScene = preload("res://UI/Menus/Credits.tscn")

export(NodePath) var LoadGamePath

var name_to_id = {}
var focused_button_id = 0

onready var LoadGame = self.get_node(LoadGamePath)
onready var buttons = $MarginContainer/VBoxContainer.get_children()

func _ready():
	for i in range(len(self.buttons)):
		self.buttons[i].connect("mouse_entered", self, "_on_button_mouse_entered", [i])
		self.name_to_id[self.buttons[i].name] = i
	if SaveManager.has_save():
		LoadGame.disabled = false
		self.focused_button_id = name_to_id[LoadGame.name]

func init(params):
	if params != null:
		self.focused_button_id = self.name_to_id[params["from"]]

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
	SaveManager.delete_data()
	self._on_LoadGame_pressed()

func _on_LoadGame_pressed():
	var main = global.find_entity("main")
	main.change_map(CityScene, {"player_pos": "game_started"})

func _on_Credits_pressed():
	var main = global.find_entity("main")
	main.change_map(CreditsScene)

func _on_Quit_pressed():
	get_tree().quit()

