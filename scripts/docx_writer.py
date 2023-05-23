"""
Tests for document generator
"""

from dessia_common.document_generator import DocxWriter, Paragraph, Heading, Header, Footer, Section, Table

paragraph_1 = Paragraph(text="This is the first paragraph.")
paragraph_2 = Paragraph(text="The second paragraph is here.")
paragraph_3 = Paragraph(text="In the third paragraph, we discuss some important details.")
paragraph_4 = Paragraph(text="Paragraph number four is shorter than the others.")
paragraph_5 = Paragraph(text="The fifth paragraph is longer and more detailed than the previous ones.")
paragraph_6 = Paragraph(text="Finally, the sixth and last paragraph concludes our discussion.")
paragraphs = [paragraph_1, paragraph_2, paragraph_3, paragraph_4, paragraph_5, paragraph_6]

heading_1 = Heading(text='Heading 1', level=1)
heading_2 = Heading(text='Heading 2', level=2)
heading_3 = Heading(text='Heading 3', level=1)
heading_4 = Heading(text='Heading 4', level=2)
headings = [heading_1, heading_2, heading_3, heading_4]

footer = Footer(text="DessIA Technologies", align="left")
footer_2 = Footer(text="DessIA Technologies", align="right")
section = Section()
section.add_element(element=footer)
section.add_element(element=footer_2)

table = Table([['Library', 'Platform'], ['dessia_common', 'testing'], ['volmdlr', 'dev']])

writer = DocxWriter(filename="test_docx.docx", paragraphs=paragraphs,
                    headings=headings, section=section, tables=[table])

writer.add_headings()
writer.add_paragraphs()
writer.add_table(table_index=0)
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
assert document.paragraphs[6].text == 'This is the first paragraph.'
assert document.paragraphs[8].text == 'The second paragraph is here.'
assert document.paragraphs[10].text == 'In the third paragraph, we discuss some important details.'
assert document.paragraphs[12].text == 'Paragraph number four is shorter than the others.'
assert document.paragraphs[13].text == 'The fifth paragraph is longer and more detailed than the previous ones.'
assert document.paragraphs[14].text == 'Finally, the sixth and last paragraph concludes our discussion.'

# Check the table
table = document.tables[0]
assert table.cell(0, 0).text == 'Library'
assert table.cell(0, 1).text == 'Platform'
assert table.cell(1, 0).text == 'dessia_common'
assert table.cell(1, 1).text == 'testing'
assert table.cell(2, 0).text == 'volmdlr'
assert table.cell(2, 1).text == 'dev'
