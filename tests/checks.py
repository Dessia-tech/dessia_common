from dessia_common import DessiaObject
# from dessia_common.datatools.dataset import Dataset
import dessia_common.checks as checks


class Battery(DessiaObject):
    
    def __init__(self, capacity: int, max_dc: int,
                  name: str = ''):
        
        DessiaObject.__init__(self, name=name)
        self.capacity = capacity
        self.max_dc = max_dc
        
    # def is_valid(self):
    #     if self.max_dc < 50:
    #         return False
    #     return True
    
    def is_valid(self):
        if checks.is_float(self.max_dc) or checks.is_float(self.max_dc):
            return False
        return True

class ElectricChar2(DessiaObject):
    
    def __init__(self, battery: Battery, brand: str, model: str, price: int, autonomy: int,
                  name: str = ''):
        
        DessiaObject.__init__(self, name=name)
        self.battery = battery
        self.brand = brand
        self.model = model
        self.price = price
        self.autonomy = autonomy