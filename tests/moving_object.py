from dessia_common.forms import MovingStandaloneObject

mso = MovingStandaloneObject(origin=0, name="Moving Test")

mso._check_platform()
disp = mso._displays()

assert len(disp) == 3
markdown = disp[0]
assert markdown["type_"] == "markdown"
assert len(markdown["data"]) == 620

plot_data = disp[1]
assert plot_data["type_"] == "plot_data"

babylon = disp[2]
assert babylon["type_"] == "babylon_data"

print("script 'moving_object.py' has passed")
