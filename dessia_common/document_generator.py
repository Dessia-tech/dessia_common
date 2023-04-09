#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Write document file. """

import re
from typing import List, Union
import docx
import markdown
from bs4 import BeautifulSoup

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
    """ Represents a header in document. """

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

    @classmethod
    def from_markdown(cls, markdown_text: str):
        """ Create a new Heading object from Markdown text. """
        html = markdown.markdown(markdown_text)
        soup = BeautifulSoup(html, 'html.parser')
        heading_tag = soup.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        if heading_tag is not None:
            text = heading_tag.text.strip()
            level = int(heading_tag.name[1])
            return cls(text, level)
        return None


class Paragraph:
    """ Represents a paragraph in the document. """

    def __init__(self, text: str):
        self.text = text

    def add_to_document(self, document: docx.Document):
        """ Add paragraph to the document. """
        document.add_paragraph(self.text)

    @classmethod
    def from_markdown(cls, markdown_text: str):
        """ Create a new Paragraph object from Markdown text. """
        html_text = markdown.markdown(markdown_text)
        soup = BeautifulSoup(html_text, 'html.parser')
        plain_text = soup.get_text("\n", strip=True)
        return cls(plain_text)


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

    def __init__(self, paragraphs: List[Paragraph] = None, section: Section = None, filename: str = None,
                 headings: List[Heading] = None):
        if filename is None:
            filename = "document.docx"
        self.filename = filename
        if headings is None:
            headings = []
        self.headings = headings
        self.section = section
        if paragraphs is None:
            paragraphs = []
        self.paragraphs = paragraphs
        self.document = docx.Document()

        if self.section:
            self.section.add_to_document(document=self.document)

    def add_headings(self, page_break: bool = True) -> 'DocxWriter':
        """ Add a list of headings to the document. """
        for heading in self.headings:
            self.document.add_heading(heading.text, level=heading.level)
        if page_break:
            self.add_page_breaks(num_page_breaks=1)
        return self

    def add_paragraphs(self, add_heading: bool = True) -> 'DocxWriter':
        """ Add a list of paragraphs to the document. """
        document = self.document
        headings = self.headings
        for paragraph in self.paragraphs:
            if headings and add_heading:
                document.add_heading(headings[0].text, headings[0].level)
                headings = headings[1:]
            document.add_paragraph(paragraph.text)
        self.document = document
        self.headings = headings
        return self

    def add_paragraph_as_heading(self, text: str):
        """
        Adds a new heading to the document, using the specified text as the heading text.

        The level of the heading is determined by the number of '#' characters in the text.
        For example, a single '#' character at the beginning of the text will create a level 1 heading.
        """
        self.document.add_heading(text, level=text.count('#'))
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
                if i < len(row):
                    cell.text = row[i]
                else:
                    cell.text = ""
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

    @staticmethod
    def parse_markdown(markdown_text: str):
        """
        Parses the given markdown text and returns a tuple containing a list of headings,
        a list of paragraphs and tables.
        """
        elements, headings = [], []
        # current_paragraph = ''
        table_pattern = re.compile(r'^\|.*\|$')
        horizontal_line_pattern = re.compile(r'^\s*\|?\s*-+\s*\|?\s*(-+\s*\|?)*\s*$')

        for line in markdown_text.split('\n'):
            line = line.strip()
            if line:

                if line.startswith('#'):
                    headings.append(Heading.from_markdown(line))
                    elements.append(Paragraph(text=line))

                elif table_pattern.match(line) and not horizontal_line_pattern.match(line):
                    row = line.strip('|').split('|')
                    if '---' not in line:
                        elements.append(row)

                else:
                    # if '---' not in line:
                    #     current_paragraph += line
                    if '---' not in line:
                        elements.append(Paragraph(text=line))

        # if current_paragraph:
        #     elements.append(Paragraph(current_paragraph, heading=heading))
        return headings, elements

    @classmethod
    def from_markdown(cls, markdown_text: str):
        """ Converts the given markdown text into a DocxWriter instance."""
        headings, elements = cls.parse_markdown(markdown_text=markdown_text)
        docx_writer = cls(headings=headings)
        docx_writer.add_headings()
        for item in elements:
            if isinstance(item, Paragraph):
                docx_writer.paragraphs = [item]
                if item.text.startswith('#'):
                    docx_writer.add_paragraph_as_heading(text=item.text.strip(' # '))
                else:
                    docx_writer.add_paragraphs(add_heading=False)
            if isinstance(item, list):
                docx_writer.add_table([item])

        return docx_writer

    def save_file(self):
        """ Saves the document to a file. """
        if not self.filename.endswith('.docx'):
            self.filename += '.docx'
        self.document.save(self.filename)

    def save_to_stream(self, stream: BinaryFile):
        """ Saves the document to a binary stream. """
        self.document.save(stream)
