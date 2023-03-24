#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Write document file. """
from typing import List, Tuple

import docx


class DocxWriter:
    """ write a docx file. """

    def __init__(self, filename: str):
        self.filename = filename
        self.document = docx.Document()
        self.headings = None

    def add_headings(self, headings: List[Tuple[str, int]]) -> 'DocxWriter':
        """ Add a list of headings to the document. """
        document = self.document
        for heading in headings:
            document.add_heading(heading[0], heading[1])
        writer = DocxWriter(self.filename)
        writer.document = document
        writer.headings = headings
        writer.add_page_breaks(1)
        return writer

    def add_paragraphs(self, paragraphs: List[str]) -> 'DocxWriter':
        """ Add a list of paragraphs to the document. """
        document = self.document
        headings = self.headings
        for paragraph in paragraphs:
            if headings:
                document.add_heading(headings[0][0], headings[0][1])
                headings = headings[1:]
            document.add_paragraph(paragraph)
        writer = DocxWriter(self.filename)
        writer.document = document
        writer.headings = headings
        return writer

    def add_page_breaks(self, num_page_breaks: int):
        """ Add an empty page to the document. """
        for _ in range(num_page_breaks):
            self.document.add_page_break()

    def add_table(self, rows: List[List[str]]) -> 'DocxWriter':
        """ Add table to the document. """
        document = self.document
        table = document.add_table(rows=1, cols=len(rows[0]))
        for i, cell in enumerate(table.rows[0].cells):
            cell.text = rows[0][i]
        for row in rows[1:]:
            row_cells = table.add_row().cells
            for i, cell in enumerate(row_cells):
                cell.text = row[i]
        writer = DocxWriter(self.filename)
        writer.document = document
        writer.headings = self.headings
        return writer

    def add_header_footer(self, text: str, is_header: bool = True, align: str = 'center') -> 'DocxWriter':
        """ Add a header or footer to the document. """
        document = self.document
        section = document.sections[-1]
        if is_header:
            header = section.header
            for paragraph in header.paragraphs:
                paragraph.text = text
                paragraph.alignment = getattr(docx.enum.text.WD_ALIGN_PARAGRAPH, align.upper())
        else:
            footer = section.footer
            for paragraph in footer.paragraphs:
                paragraph.text = text
                paragraph.alignment = getattr(docx.enum.text.WD_ALIGN_PARAGRAPH, align.upper())
        writer = DocxWriter(self.filename)
        writer.document = document
        writer.headings = self.headings
        return writer

    def add_picture(self, image_path: str, width: int = None, height: int = None) -> 'DocxWriter':
        """ Add an image to the document. """
        document = self.document
        document.add_picture(image_path, width=width, height=height)
        writer = DocxWriter(self.filename)
        writer.document = document
        writer.headings = self.headings
        return writer

    def save_file(self):
        """ Saves the document to a file. """
        self.document.save(self.filename)
