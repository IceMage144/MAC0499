extends CanvasLayer

var debug_mode = false

func init(params):
	pass

func close_popup():
	var main = global.find_entity("main")
	main.close_popup()