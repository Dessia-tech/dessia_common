"""
    test excel reader
"""

from dessia_common.models import all_cars_no_feat
from dessia_common.datatools.dataset import Dataset
from dessia_common.excel_reader import ExcelReader

dataset = Dataset(all_cars_no_feat)

dataset.to_xlsx("dataset.xlsx'")

excel_file_path = 'dataset.xlsx'
reader = ExcelReader(excel_file_path)
main_obj = reader.read_object()
