from dessia_common import DessiaObject
# from dessia_common.datatools.dataset import Dataset
import dessia_common.checks as checks


class Battery(DessiaObject):

    def __init__(self, capacity: float, number_cells: int,
                 name: str = ''):

        DessiaObject.__init__(self, name=name)
        self.capacity = capacity
        self.number_cells = number_cells

    # def is_valid(self):
    #     if self.max_dc < 50:
    #         return False
    #     return True

    def check_list(self, level='info'):
        check_list = DessiaObject.check_list(self, level=level)

        check_list += checks.is_float(self.capacity, level=level)
        check_list += checks.is_int(self.number_cells, level=level)
        check_list += checks.is_str(self.name, level=level)

        return check_list


battery = Battery(3., 2, 'Good name')
check_list = battery.check_list()
print(check_list)
check_list.raise_if_above_level('error')

battery = Battery(None, 22.2, 1)
check_list2 = battery.check_list()
print(check_list2)

raised = False
try:
    check_list2.raise_if_above_level('error')
except:
    raised = True

assert raised


class ElectricChar2(DessiaObject):

    def __init__(self, battery: Battery, brand: str, model: str, price: int, autonomy: int,
                 name: str = ''):

        DessiaObject.__init__(self, name=name)
        self.battery = battery
        self.brand = brand
        self.model = model
        self.price = price
        self.autonomy = autonomy
