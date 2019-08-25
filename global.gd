extends Node

const SAVE_PATH = "user://save.save"

func has_entity(entity_name):
	return len(get_tree().get_nodes_in_group(entity_name)) != 0

func find_entity(entity_name):
	if not has_entity(entity_name):
		return null
	
	var entity_list = get_tree().get_nodes_in_group(entity_name)
	return entity_list[0]

func get_team(entity):
	if entity.is_in_group("team1"):
		return "team1"
	elif entity.is_in_group("team2"):
		return "team2"
	return ""

func get_enemy(entity):
	if entity.is_in_group("team1"):
		return self.find_entity("team2")
	elif entity.is_in_group("team2"):
		return self.find_entity("team1")
	return null

func randi_range(start, end):
	return int(start + floor(randf() * (end - start + 1)))

func save_info(nodes):
	var saveFile = File.new()
	var saveInfo = []
	if saveFile.file_exists(SAVE_PATH):
		saveFile.open(SAVE_PATH, saveFile.READ)
		saveInfo = JSON.parse(saveFile.get_as_text()).result
		saveFile.close()
	else:
		saveInfo = []

	var newInfos = {}
	for node in nodes:
		var name = node.get_name()
		var info = node.get_info()
		newInfos[name] = info
	
	saveInfo.append(newInfos)

	saveFile.open(SAVE_PATH, saveFile.WRITE)
	saveFile.store_string(JSON.print(saveInfo))
	saveFile.close()

