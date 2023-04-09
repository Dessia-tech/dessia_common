#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Write document file. """
import os
import tempfile
from typing import List, Union

import docx

from dessia_common.files import BinaryFile


class LayoutElement:
    """
    Represents a header or footer in document.

    This class is used to create headers and footers for a docx document. Subclasses
    `Header` and `Footer` are available for convenience.
    """

    def __init__(self, text: str, align: str = 'center'):
        self.text = text
        self.align = align

    def _add_to_section(self, section, type_: str):
        """
        Add the header or footer to the specified section.

        :param section: The `docx.section.Section` object to add the header/footer to.
        :param type_: The type of element to add, either "header" or "footer".
        """
        element = getattr(section, type_)
        paragraph = element.add_paragraph()
        paragraph.text = self.text
        paragraph.alignment = getattr(docx.enum.text.WD_ALIGN_PARAGRAPH, self.align.upper())

    def _add_picture(self, section, type_: str, image_path: str, width: int = None, height: int = None):
        """ Add the header or footer picture to the specified section. """
        element = getattr(section, type_)
        paragraph = element.paragraphs[0] if element.paragraphs else element.add_paragraph()
        run = paragraph.add_run()
        run.add_picture(image_path, width=width, height=height)


class Header(LayoutElement):
    """ Represents a header in a docx document. """

    def add_to_document(self, document: docx.Document):
        """ Add the header to the document. """
        section = document.sections[0]
        super()._add_to_section(section, type_='header')

    def add_to_section(self, section):
        """ Add header to section. """
        super()._add_to_section(section=section, type_='header')

    def add_picture(self, section, image_path: str, width: int = None, height: int = None):
        """ Add picture for header. """
        super()._add_picture(section=section, type_='header', image_path=image_path, width=width, height=height)


class Footer(LayoutElement):
    """ Represents a footer in document. """

    def add_to_document(self, document: docx.Document):
        """ Add the footer to the document. """
        section = document.sections[0]
        super()._add_to_section(section, type_='footer')

    def add_to_section(self, section):
        """ Add footer to section. """
        super()._add_to_section(section=section, type_='footer')

    def add_picture(self, section, image_path: str, width: int = None, height: int = None):
        """ Add picture for footer. """
        super()._add_picture(section=section, type_='footer', image_path=image_path, width=width, height=height)


class Heading:
    """ Represents a heading in document. """

    def __init__(self, text: str, level: int):
        self.text = text
        self.level = level

    def add_to_document(self, document: docx.Document):
        """ Add the heading to the document. """
        document.add_heading(self.text, self.level)


class Paragraph:
    """ Represents a paragraph in the document. """

    def __init__(self, text: str):
        self.text = text

    def add_to_document(self, document: docx.Document):
        """ Add paragraph to the document. """
        document.add_paragraph(self.text)


class Section:
    """
    Represents a section in the document.

    A section is a part of the document with its own headers, footers.
    """

    def __init__(self):
        self.elements = []

    def add_element(self, element: Union[Header, Footer]):
        """ Add a header to the section. """
        self.elements.append(element)

    def add_to_document(self, document: docx.Document):
        """ Add the section to the document. """
        section = document.sections[0]
        for element in self.elements:
            element.add_to_section(section)

    def add_picture_to_document(self, document: docx.Document, image_path, width, height):
        """ Add picture to the section. """
        section = document.sections[0]
        if self.elements:
            self.elements[0].add_picture(section, image_path, width, height)


class DocxWriter:
    """ write a docx file. """

    def __init__(self, filename: str, paragraphs: List[Paragraph], section: Section, headings: List[Heading] = None):
        self.filename = filename
        self.headings = headings
        self.section = section
        self.paragraphs = paragraphs
        self.document = docx.Document()

        self.section.add_to_document(document=self.document)

    def add_headings(self) -> 'DocxWriter':
        """ Add a list of headings to the document. """
        for heading in self.headings:
            self.document.add_heading(heading.text, level=heading.level)
        self.add_page_breaks(num_page_breaks=1)
        return self

    def add_paragraphs(self) -> 'DocxWriter':
        """ Add a list of paragraphs to the document. """
        document = self.document
        headings = self.headings
        for paragraph in self.paragraphs:
            if headings:
                document.add_heading(headings[0].text, headings[0].level)
                headings = headings[1:]
            document.add_paragraph(paragraph.text)
        self.document = document
        self.headings = headings
        return self

    def add_page_breaks(self, num_page_breaks: int):
        """ Add an empty page to the document. """
        for _ in range(num_page_breaks):
            self.document.add_page_break()

    def add_table(self, rows: List[List[str]]) -> 'DocxWriter':
        """ Add table to the document. """
        document = self.document
        table = document.add_table(rows=1, cols=len(rows[0]), style="TableGrid")
        for i, cell in enumerate(table.rows[0].cells):
            cell.text = rows[0][i]
        for row in rows[1:]:
            row_cells = table.add_row().cells
            for i, cell in enumerate(row_cells):
                cell.text = row[i]
        self.document = document
        return self

    def add_list_items(self, items: List[str], style: str = 'List Bullet'):
        """
        Add a list (bullet or numbered) to the document.

        :param items: A list of strings to be added as list items
        :type items: List[str]

        :param style: The style of the list. Valid values are 'List Bullet' (default) or 'List Number'
        :type style: str
        """
        document = self.document
        for item in items:
            document.add_paragraph(item, style=style)
        self.document = document
        return self

    def add_header_footer_picture(self, image_path: str, width: int = None, height: int = None)\
            -> 'DocxWriter':
        """ Add a picture to the header or footer of the document. """
        self.section.add_picture_to_document(document=self.document, image_path=image_path, width=width, height=height)
        return self

    def add_picture(self, image_path: str, width: int = None, height: int = None) -> 'DocxWriter':
        """ Add an image to the document. """
        self.document.add_picture(image_path, width=width, height=height)
        return self

    def delete_header_footer(self):
        """ Remove the header and footer from all sections in a Word document. """
        for section in self.document.sections:
            section.different_first_page_header_footer = False
            section.header.is_linked_to_previous = True
            section.footer.is_linked_to_previous = True

    def convert_to_pdf(self) -> bytes:
        """ Convert the document content to a PDF byte stream. """
        with tempfile.NamedTemporaryFile(suffix=".docx") as temp_file:
            self.document.save(temp_file.name)

            pdf_file = BinaryFile()
            os.system(f"libreoffice --convert-to pdf --outdir {os.path.dirname(temp_file.name)} {temp_file.name}")
            with open(temp_file.name.replace(".docx", ".pdf"), "rb") as f:
                pdf_file.write(f.read())

            return pdf_file.getvalue()

    def save_file(self):
        """ Saves the document to a file. """
        self.document.save(self.filename)

    def save_to_stream(self, stream: BinaryFile):
        """ Saves the document to a binary stream. """
        self.document.save(stream)

    def save_pdf_file(self, filepath: str):
        """ Save the PDF version of the document to a file."""
        if not filepath.endswith('pdf'):
            filepath += '.pdf'
        with open(filepath, 'wb') as f:
            pdf_stream = BinaryFile()
            self.save_pdf_stream(pdf_stream)
            f.write(pdf_stream.getvalue())

    def save_pdf_stream(self, stream: BinaryFile):
        """ Save the PDF version of the document to a binary file stream. """
        pdf_bytes = self.convert_to_pdf()
        stream.write(pdf_bytes)
