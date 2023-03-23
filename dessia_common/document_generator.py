#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Write document file. """
from typing import List, Tuple

import docx


class DocxWriter:
    """ write a docx file. """

    def __init__(self, filename: str):
        self.filename = filename
        self.doc = docx.Document()
        self.headings = None

    def add_headings(self, headings: List[Tuple[str, int]]) -> 'DocxWriter':
        """ Add a list of headings to the document. """
        new_doc = self.doc
        for heading in headings:
            new_doc.add_heading(heading[0], heading[1])
        new_writer = DocxWriter(self.filename)
        new_writer.doc = new_doc
        new_writer.headings = headings
        new_writer.add_page_breaks(1)
        return new_writer

    def add_paragraphs(self, paragraphs: List[str]) -> 'DocxWriter':
        """ Add a list of paragraphs to the document. """
        new_doc = self.doc
        new_headings = self.headings
        for paragraph in paragraphs:
            if new_headings:
                new_doc.add_heading(new_headings[0][0], new_headings[0][1])
                new_headings = new_headings[1:]
            new_doc.add_paragraph(paragraph)
        new_writer = DocxWriter(self.filename)
        new_writer.doc = new_doc
        new_writer.headings = new_headings
        return new_writer

    def add_page_breaks(self, num_page_breaks: int):
        """ Add an empty page to the document. """
        for _ in range(num_page_breaks):
            self.doc.add_page_break()

    def add_table(self, rows: List[List[str]]) -> 'DocxWriter':
        """ Add table to the document. """
        new_doc = self.doc
        table = new_doc.add_table(rows=1, cols=len(rows[0]))
        for i, cell in enumerate(table.rows[0].cells):
            cell.text = rows[0][i]
        for row in rows[1:]:
            row_cells = table.add_row().cells
            for i, cell in enumerate(row_cells):
                cell.text = row[i]
        new_writer = DocxWriter(self.filename)
        new_writer.doc = new_doc
        new_writer.headings = self.headings
        return new_writer

    def add_header_footer(self, text: str, is_header: bool = True, align: str = 'center') -> 'DocxWriter':
        """ Add a header or footer to the document. """
        new_doc = self.doc
        section = new_doc.sections[-1]
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
        new_writer = DocxWriter(self.filename)
        new_writer.doc = new_doc
        new_writer.headings = self.headings
        return new_writer

    def add_picture(self, image_path: str, width: int = None, height: int = None) \
            -> 'DocxWriter':
        """ Add an image to the document. """
        new_doc = self.doc
        new_doc.add_picture(image_path, width=width, height=height)
        new_writer = DocxWriter(self.filename)
        new_writer.doc = new_doc
        new_writer.headings = self.headings
        return new_writer

    def save_file(self):
        self.doc.save(self.filename)
