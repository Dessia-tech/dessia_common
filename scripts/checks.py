from dessia_common.core import DessiaObject
import dessia_common.checks as checks


class Battery(DessiaObject):
    """ Mock a battery. """

    def __init__(self, capacity: float, number_cells: int, name: str = ''):

        DessiaObject.__init__(self, name=name)
        self.capacity = capacity
        self.number_cells = number_cells

    def check_list(self, level='info'):
        check_list = DessiaObject.check_list(self, level=level, check_platform=False)

        check_list += checks.is_float(self.capacity, level=level)
        check_list += checks.is_int(self.number_cells, level=level)
        check_list += checks.is_str(self.name, level=level)

        return check_list


BATTERY = Battery(3., 2, 'Good name')
CHECK_LIST = BATTERY.check_list()
print(CHECK_LIST)
CHECK_LIST.raise_if_above_level('error')

BATTERY = Battery(None, 22.2, 1)
INVALID_CHECK_LIST = BATTERY.check_list()
print(INVALID_CHECK_LIST)

raised = False
try:
    INVALID_CHECK_LIST.raise_if_above_level('error')
except:
    raised = True

assert raised


class ElectricChar2(DessiaObject):
    """ Mock a system using a battery. """

    def __init__(self, battery: Battery, brand: str, model: str, price: int, autonomy: int, name: str = ''):

        DessiaObject.__init__(self, name=name)
        self.battery = battery
        self.brand = brand
        self.model = model
        self.price = price
        self.autonomy = autonomy


print("script 'checks.py' has passed")
