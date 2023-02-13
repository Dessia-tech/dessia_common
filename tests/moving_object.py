from dessia_common.forms import MovingStandaloneObject

mso = MovingStandaloneObject(origin=0, name="Moving Test")

mso._check_platform()
disp = mso._displays()
