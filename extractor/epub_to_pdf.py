#!/usr/bin/env python3
"""
EPUB to PDF Structure Generator

This script extracts the table of contents and chapter structure from EPUB files
and generates a PDF document with the hierarchical structure.
"""

import argparse
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
import re
from typing import List, Dict, Tuple, Optional
import sys
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, darkblue


class EPUBToPDFGenerator:
    def __init__(self, epub_path: str):
        self.epub_path = Path(epub_path)
        self.namespace_map = {}

    def extract_structure(self) -> Dict:
        """Extract basic structure from an EPUB file (simplified version)."""
        if not self.epub_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {self.epub_path}")
            
        # Return basic structure with just the title
        return {
            'title': self.epub_path.stem,
            'chapters': []
        }

    def _find_opf_path(self, container_root: ET.Element) -> str:
        """Find the path to the OPF file from container.xml."""
        for rootfile in container_root.findall('.//{urn:oasis:names:tc:opendocument:xmlns:container}rootfile'):
            if rootfile.get('media-type') == 'application/oebps-package+xml':
                return rootfile.get('full-path')
        raise Exception("Could not find OPF file in container.xml")
    
    def _extract_detailed_structure(self) -> Dict:
        """Extract detailed structure including all headings from chapter content."""
        if not self.epub_path.exists():
            raise FileNotFoundError(f"EPUB file not found: {self.epub_path}")
            
        try:
            with zipfile.ZipFile(self.epub_path, 'r') as epub:
                # Get basic structure first
                basic_structure = self.extract_structure()
                
                # Get the container.xml to find the OPF file
                container_content = epub.read('META-INF/container.xml')
                container_root = ET.fromstring(container_content)
                opf_path = self._find_opf_path(container_root)
                opf_content = epub.read(opf_path)
                opf_root = ET.fromstring(opf_content)
                
                # Get all HTML files from the spine
                html_files = self._get_spine_files(epub, opf_root, opf_path)
                
                # Extract detailed structure from each HTML file
                detailed_chapters = []
                for html_file in html_files:
                    chapter_structure = self._extract_chapter_structure(epub, html_file)
                    if chapter_structure:
                        detailed_chapters.append(chapter_structure)
                
                return {
                    'title': basic_structure['title'],
                    'chapters': detailed_chapters
                }
                    
        except Exception as e:
            raise Exception(f"Error processing EPUB file: {str(e)}")

    def _get_spine_files(self, epub: zipfile.ZipFile, opf_root: ET.Element, opf_path: str) -> List[Dict]:
        """Get all HTML files from the spine in order."""
        opf_dir = str(Path(opf_path).parent)
        if opf_dir == '.':
            opf_dir = ''
        
        # Get manifest items
        manifest_items = {}
        for item in opf_root.findall('.//{http://www.idpf.org/2007/opf}item'):
            item_id = item.get('id')
            href = item.get('href')
            if opf_dir:
                full_path = f"{opf_dir}/{href}"
            else:
                full_path = href
            manifest_items[item_id] = {
                'href': href,
                'full_path': full_path,
                'media-type': item.get('media-type')
            }
        
        # Get spine order
        html_files = []
        for itemref in opf_root.findall('.//{http://www.idpf.org/2007/opf}itemref'):
            idref = itemref.get('idref')
            if idref in manifest_items and manifest_items[idref]['media-type'] == 'application/xhtml+xml':
                html_files.append(manifest_items[idref])
        
        return html_files

    def _extract_chapter_structure(self, epub: zipfile.ZipFile, html_file: Dict) -> Dict:
        """Extract structure from a single HTML file."""
        try:
            content = epub.read(html_file['full_path'])
            html_content = content.decode('utf-8', errors='ignore')
            
            # Find all headings in the HTML content
            headings = self._find_all_headings(html_content)
            
            if not headings:
                # If no headings found, use filename as title
                title = Path(html_file['href']).stem.replace('_', ' ').replace('-', ' ').title()
                return {
                    'title': title,
                    'level': 1,
                    'src': html_file['href'],
                    'children': []
                }
            
            # Build hierarchical structure from headings
            return self._build_heading_hierarchy(headings, html_file['href'])
            
        except Exception as e:
            print(f"Warning: Could not process {html_file['href']}: {e}")
            return None

    def _find_all_headings(self, html_content: str) -> List[Dict]:
        """Find all heading tags (h1-h6) in HTML content."""
        headings = []
        
        # Pattern to match heading tags with their content
        heading_pattern = r'<(h[1-6])[^>]*>(.*?)</\1>'
        
        for match in re.finditer(heading_pattern, html_content, re.IGNORECASE | re.DOTALL):
            tag = match.group(1).lower()
            content = match.group(2)
            
            # Clean up the content (remove HTML tags, decode entities)
            clean_content = re.sub(r'<[^>]+>', '', content)
            clean_content = re.sub(r'&[a-zA-Z0-9#]+;', ' ', clean_content)
            clean_content = ' '.join(clean_content.split())  # Normalize whitespace
            
            if clean_content.strip():
                level = int(tag[1])  # Extract number from h1, h2, etc.
                headings.append({
                    'title': clean_content.strip(),
                    'level': level,
                    'position': match.start()
                })
        
        return headings

    def _build_heading_hierarchy(self, headings: List[Dict], src: str) -> Dict:
        """Build hierarchical structure from flat list of headings."""
        if not headings:
            return None
        
        # Start with the first heading as the main chapter
        root = {
            'title': headings[0]['title'],
            'level': 1,  # Always start chapters at level 1
            'src': src,
            'children': []
        }
        
        # Build hierarchy for remaining headings
        if len(headings) > 1:
            root['children'] = self._build_children_hierarchy(headings[1:], 2)
        
        return root

    def _build_children_hierarchy(self, headings: List[Dict], base_level: int) -> List[Dict]:
        """Recursively build hierarchy for child headings."""
        if not headings:
            return []
        
        children = []
        i = 0
        
        while i < len(headings):
            current = headings[i]
            
            # Adjust level relative to base level
            adjusted_level = base_level + (current['level'] - headings[0]['level'])
            
            child = {
                'title': current['title'],
                'level': adjusted_level,
                'src': '',
                'children': []
            }
            
            # Find children for this heading
            j = i + 1
            child_headings = []
            
            while j < len(headings):
                if headings[j]['level'] <= current['level']:
                    break
                child_headings.append(headings[j])
                j += 1
            
            # Recursively build children
            if child_headings:
                child['children'] = self._build_children_hierarchy(child_headings, adjusted_level + 1)
            
            children.append(child)
            i = j
        
        return children

    def generate_pdf(self, output_path: str = None) -> str:
        """Generate a PDF with the EPUB structure."""
        # Extract detailed structure
        structure = self._extract_detailed_structure()
        
        # Set output path
        if not output_path:
            output_path = self.epub_path.stem + "_structure.pdf"
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []
        
        # Set up styles
        styles = getSampleStyleSheet()
        
        # Title style
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Title'],
            fontSize=24,
            textColor=darkblue,
            spaceAfter=30,
            alignment=1  # Center
        )
        
        # Heading styles for different levels
        heading_styles = {}
        for level in range(1, 7):
            heading_styles[level] = ParagraphStyle(
                f'Heading{level}',
                parent=styles['Heading1'],
                fontSize=16 - (level - 1) * 2,
                textColor=blue if level == 1 else black,
                leftIndent=(level - 1) * 20,
                spaceAfter=12,
                spaceBefore=6
            )
        
        # Add title
        story.append(Paragraph(f"Estructura del Libro", title_style))
        story.append(Paragraph(f"{structure['title']}", title_style))
        story.append(Spacer(1, 20))
        
        # Add table of contents header
        toc_style = ParagraphStyle(
            'TOCHeader',
            parent=styles['Heading1'],
            fontSize=18,
            textColor=darkblue,
            spaceAfter=20
        )
        story.append(Paragraph("Índice de Contenidos", toc_style))
        
        # Add chapters with numbering
        def add_chapter_to_story(chapter: Dict, story: List, numbering: List[int]):
            level = min(chapter['level'], 6)  # Limit to 6 levels
            style = heading_styles[level]
            
            # Create numbering string (e.g., "1.2.3")
            number_str = ".".join(map(str, numbering))
            
            # Create the chapter entry with numbering
            chapter_text = f"{number_str} {chapter['title']}"
            story.append(Paragraph(chapter_text, style))
            
            # Add children with incremented numbering
            if chapter['children']:
                for i, child in enumerate(chapter['children'], 1):
                    child_numbering = numbering + [i]
                    add_chapter_to_story(child, story, child_numbering)
        
        if structure['chapters']:
            for i, chapter in enumerate(structure['chapters'], 1):
                add_chapter_to_story(chapter, story, [i])
        else:
            no_chapters_style = ParagraphStyle(
                'NoChapters',
                parent=styles['Normal'],
                fontSize=12,
                textColor=black,
                spaceAfter=12
            )
            story.append(Paragraph("No se encontraron capítulos en la tabla de contenidos.", no_chapters_style))
        
        # Add summary
        story.append(Spacer(1, 30))
        summary_style = ParagraphStyle(
            'Summary',
            parent=styles['Normal'],
            fontSize=10,
            textColor=black,
            spaceAfter=6
        )
        
        total_chapters = self._count_chapters(structure['chapters'])
        story.append(Paragraph(f"Total de capítulos encontrados: {total_chapters}", summary_style))
        story.append(Paragraph(f"Archivo EPUB: {self.epub_path.name}", summary_style))
        
        # Build PDF
        doc.build(story)
        
        return output_path

    def _count_chapters(self, chapters: List[Dict]) -> int:
        """Count total number of chapters recursively."""
        count = len(chapters)
        for chapter in chapters:
            count += self._count_chapters(chapter['children'])
        return count

    def generate_markdown(self, output_path: str = None) -> str:
        """Generate a Markdown file with the EPUB structure."""
        # Extract detailed structure
        structure = self._extract_detailed_structure()
        
        # Set output path
        if not output_path:
            output_path = self.epub_path.stem + "_structure.md"
        
        # Create output directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        markdown_content = []
        
        # Add title
        markdown_content.append(f"# Estructura del Libro")
        markdown_content.append(f"## {structure['title']}")
        markdown_content.append("")
        markdown_content.append("---")
        markdown_content.append("")
        
        # Add table of contents
        markdown_content.append("## Índice de Contenidos")
        markdown_content.append("")
        
        # Function to add chapters recursively with numbering
        def add_chapter_to_markdown(chapter: Dict, content: List[str], numbering: List[int]):
            # Create numbering string (e.g., "1.2.3")
            number_str = ".".join(map(str, numbering))
            
            # Add indentation based on level
            indent = "  " * (len(numbering) - 1)
            
            content.append(f"{indent}{number_str} {chapter['title']}")
            content.append("")
            
            # Add children with incremented numbering
            if chapter['children']:
                for i, child in enumerate(chapter['children'], 1):
                    child_numbering = numbering + [i]
                    add_chapter_to_markdown(child, content, child_numbering)
        
        if structure['chapters']:
            for i, chapter in enumerate(structure['chapters'], 1):
                add_chapter_to_markdown(chapter, markdown_content, [i])
        else:
            markdown_content.append("No se encontraron capítulos en la tabla de contenidos.")
            markdown_content.append("")
        
        # Add summary
        markdown_content.append("---")
        markdown_content.append("")
        markdown_content.append("## Resumen")
        markdown_content.append("")
        total_chapters = self._count_chapters(structure['chapters'])
        markdown_content.append(f"- **Total de capítulos encontrados:** {total_chapters}")
        markdown_content.append(f"- **Archivo EPUB:** `{self.epub_path.name}`")
        markdown_content.append("")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(markdown_content))
        
        return output_path

    def generate_both_outputs(self, output_dir: str = "output") -> Dict[str, str]:
        """Generate both PDF and Markdown outputs in the specified directory."""
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate file paths
        base_name = self.epub_path.stem
        pdf_path = output_path / f"{base_name}_structure.pdf"
        md_path = output_path / f"{base_name}_structure.md"
        
        # Generate both files
        pdf_result = self.generate_pdf(str(pdf_path))
        md_result = self.generate_markdown(str(md_path))
        
        return {
            'pdf': pdf_result,
            'markdown': md_result
        }


def main():
    parser = argparse.ArgumentParser(
        description="Generate PDF and Markdown with EPUB table of contents structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python epub_to_pdf.py book.epub
  python epub_to_pdf.py book.epub --output-dir custom_output
        """
    )
    
    parser.add_argument('epub_file', help='Path to the EPUB file')
    parser.add_argument('--output-dir', '-o', default='output', 
                       help='Output directory for generated files (default: output)')
    
    args = parser.parse_args()
    
    try:
        # Check if reportlab is available
        try:
            import reportlab
        except ImportError:
            print("Error: reportlab library is required to generate PDF files.")
            print("Install it with: pip install reportlab")
            sys.exit(1)
        
        generator = EPUBToPDFGenerator(args.epub_file)
        results = generator.generate_both_outputs(args.output_dir)
        
        print(f"✅ Archivos generados exitosamente:")
        print(f"   📄 PDF: {results['pdf']}")
        print(f"   📝 Markdown: {results['markdown']}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error procesando el archivo EPUB: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()