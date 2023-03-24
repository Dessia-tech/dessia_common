"""
Tests for document generator
"""

from dessia_common.document_generator import DocxWriter

writer = DocxWriter("test_docx.docx")
writer = writer.add_headings([('Heading 1', 1), ('Heading 2', 2), ('Heading 3', 1), ('Heading 4', 2)])
paragraph_1 = "This is the first paragraph."
paragraph_2 = "The second paragraph is here."
paragraph_3 = "In the third paragraph, we discuss some important details."
paragraph_4 = "Paragraph number four is shorter than the others."
paragraph_5 = "The fifth paragraph is longer and more detailed than the previous ones."
paragraph_6 = "Finally, the sixth and last paragraph concludes our discussion."
writer = writer.add_paragraphs([paragraph_1, paragraph_2, paragraph_3, paragraph_4, paragraph_5, paragraph_6])
writer = writer.add_table([['Librairie', 'Platform'], ['dessia_common', 'testing'], ['volmdlr', 'dev']])
writer = writer.add_header_footer(text="DessIA Technologies", is_header=False, align='left')
writer.add_page_breaks(num_page_breaks=1)
writer.save_file()

document = writer.document

# Check the headings
assert document.paragraphs[0].text == 'Heading 1'
assert document.paragraphs[0].style.name == 'Heading 1'
assert document.paragraphs[1].text == 'Heading 2'
assert document.paragraphs[1].style.name == 'Heading 2'
assert document.paragraphs[2].text == 'Heading 3'
assert document.paragraphs[2].style.name == 'Heading 1'
assert document.paragraphs[3].text == 'Heading 4'
assert document.paragraphs[3].style.name == 'Heading 2'

# Check the paragraphs
assert document.paragraphs[5].text == 'This is the first paragraph.'
assert document.paragraphs[6].text == 'The second paragraph is here.'
assert document.paragraphs[7].text == 'In the third paragraph, we discuss some important details.'
assert document.paragraphs[8].text == 'Paragraph number four is shorter than the others.'
assert document.paragraphs[9].text == 'The fifth paragraph is longer and more detailed than the previous ones.'
assert document.paragraphs[10].text == 'Finally, the sixth and last paragraph concludes our discussion.'

# Check the table
table = document.tables[0]
assert table.cell(0, 0).text == 'Librairie'
assert table.cell(0, 1).text == 'Platform'
assert table.cell(1, 0).text == 'dessia_common'
assert table.cell(1, 1).text == 'testing'
assert table.cell(2, 0).text == 'volmdlr'
assert table.cell(2, 1).text == 'dev'
