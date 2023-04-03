#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Write document file. """
from typing import List

import docx

from dessia_common.files import BinaryFile


class HeaderFooter:
    """ Represents a header or footer in a docx document. """

    def __init__(self, text: str, align: str = 'center'):
        self.text = text
        self.align = align

    def add_to_section(self, section, is_header):
        """ Add the header to the document. """
        if is_header:
            header = section.header
        else:
            footer = section.footer
        paragraph = header.add_paragraph() if is_header else footer.add_paragraph()
        paragraph.text = self.text
        paragraph.alignment = getattr(docx.enum.text.WD_ALIGN_PARAGRAPH, self.align.upper())


class Header(HeaderFooter):
    """ Represents a header in a docx document. """

    def add_to_document(self, document: docx.Document):
        """ Add the header to the document. """
        section = document.sections[-1]
        super().add_to_section(section, is_header=True)


class Footer(HeaderFooter):
    """ Represents a footer in a docx document. """

    def add_to_document(self, document: docx.Document):
        """ Add the footer to the document. """
        section = document.sections[-1]
        super().add_to_section(section, is_header=False)


class Heading:
    """ Represents a heading in a docx document. """

    def __init__(self, text: str, level: int):
        self.text = text
        self.level = level

    def add_to_document(self, document: docx.Document):
        """ Add the heading to the document. """
        document.add_heading(self.text, self.level)


class Paragraph:
    """ Represents a paragraph in a docx document. """

    def __init__(self, text: str):
        self.text = text

    def add_to_document(self, document: docx.Document):
        """ Add paragraph to the document. """
        document.add_paragraph(self.text)


class DocxWriter:
    """ write a docx file. """

    def __init__(self, filename: str, paragraphs: List[Paragraph], headings: List[Heading] = None, footer: Footer = None
                 , header: Header = None):
        self.filename = filename
        self.headings = headings
        self.footer = footer
        self.header = header
        self.paragraphs = paragraphs
        self.document = docx.Document()

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

    def add_header_footer(self, is_header: bool = True) -> 'DocxWriter':
        """ Add a header or footer to the document. """
        section = self.document.sections[-1]
        if is_header and self.header is not None:
            self.header.add_to_section(section=section, is_header=is_header)
        elif not is_header and self.header is not None:
            self.footer.add_to_section(section=section, is_header=is_header)
        else:
            raise ValueError("Cannot add header/footer. Header/Footer object is not defined.")

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

    def add_header_footer_picture(self, image_path: str, is_header: bool = True, width: int = None, height: int = None)\
            -> 'DocxWriter':
        """ Add a picture to the header or footer of the document. """
        section = self.document.sections[-1]
        if is_header:
            header = section.header
            paragraph = header.paragraphs[0] if header.paragraphs else header.add_paragraph()
            run = paragraph.add_run()
            run.add_picture(image_path, width=width, height=height)
        else:
            footer = section.footer
            paragraph = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
            run = paragraph.add_run()
            run.add_picture(image_path, width=width, height=height)
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

    def save_file(self):
        """ Saves the document to a file. """
        self.document.save(self.filename)

    def save_to_stream(self, stream: BinaryFile):
        """ Saves the document to a binary stream. """
        self.document.save(stream)
