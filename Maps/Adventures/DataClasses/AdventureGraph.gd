extends Object

const EXIT_ENTRANCE_MAP = {
	"left": "right",
	"right": "left",
	"top": "bottom",
	"bottom": "top"
}

const CITY_ROOM = -1

var rooms
var room_count
var actual_room
var connected_to_city

func _init():
	self.rooms = {}
	self.room_count = 0
	self.actual_room = CITY_ROOM
	self.connected_to_city = null

func create_room(type, exits, num_enemies, enemies, num_resources, resources):
	var room_id = self.room_count
	self.rooms[room_id] = Room.new(type, exits, num_enemies, enemies,
								   num_resources, resources)
	self.rooms[room_id].connect("room_dropped", self, "back_to_city")
	self.room_count += 1
	return room_id

func has_room(room_id):
	return self.rooms.has(room_id)

func get_room(room_id):
	# Assert that the room exists
	assert(self.has_room(room_id))
	return self.rooms[room_id]

func connect_rooms(room_id1, exit, room_id2):
	# Assert that first room exists
	assert(self.rooms.has(room_id1))
	# Assert that second room exists
	assert(self.rooms.has(room_id2))
	var room1 = self.rooms[room_id1]
	var room2 = self.rooms[room_id2]
	room1.connect_to_room(room2, exit)
	if EXIT_ENTRANCE_MAP.has(exit):
		room2.connect_to_room(room1, EXIT_ENTRANCE_MAP[exit])
	else:
		room2.connect_to_room(room1, exit)

func connect_to_city(room_id, exit):
	# Assert no other room is connected to city
	assert(self.connected_to_city == null)
	var room = self.rooms[room_id]
	room.connect(exit, CITY_ROOM)
	self.connected_to_city = room

func back_to_city():
	var CityScene = load("res://Maps/City/City.tscn")
	var main = global.find_entity("main")
	main.change_map(CityScene, {"player_pos": "dungeon"})

func change_room(exit=null):
	var next_room
	if self.actual_room == CITY_ROOM:
		# Assert some room was connected to city
		assert(self.connected_to_city != null)
		next_room = self.connected_to_city
	else:
		next_room = self.actual_room.get_connected(exit)
		self.actual_room.kill_instance()
	if next_room == CITY_ROOM:
		self.back_to_city()
	else:
		self.actual_room = next_room
		return next_room.get_instance()