extends "res://UI/Menus/MenuBase.gd"

func _ready():
	$Back.grab_focus()

func _process(delta):
	if not $Back.is_hovered():
		$Back.release_focus()
		$Back.grab_focus()

func _on_Back_pressed():
	var MainMenu = load("res://UI/Menus/MainMenu.tscn")
	var main = global.find_entity("main")
	main.change_map(MainMenu, {"from": "Credits"})
