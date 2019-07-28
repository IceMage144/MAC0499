extends "res://UI/Menus/MenuBase.gd"

const MainMenu = preload("res://UI/Menus/MainMenu.tscn")

func _ready():
	$Back.grab_focus()

func _process(delta):
	if not $Back.is_hovered():
		$Back.release_focus()
		$Back.grab_focus()

func _on_Back_pressed():
	var main = global.find_entity("main")
	main.change_map(MainMenu, {"from": "Credits"})
