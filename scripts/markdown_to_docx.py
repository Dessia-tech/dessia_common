"""
    test markdown to docx
"""

from dessia_common.models import all_cars_no_feat
from dessia_common.datatools import HeterogeneousList
from dessia_common.document_generator import DocxWriter

dataset = HeterogeneousList(all_cars_no_feat)
docx_writer = DocxWriter.from_markdown(markdown_text=dataset.to_markdown())
docx_writer.save_file()

print("script 'markdown_to_docx.py' has passed")
