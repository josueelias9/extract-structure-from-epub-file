from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup
from typing import Dict, Any


class EPUBExtractor:
    """Extractor for EPUB files that creates hierarchical structure"""
    
    def extract_structure(self, epub_path: str) -> Dict[str, Any]:
        """
        Extract hierarchical structure from EPUB file
        
        Args:
            epub_path: Path to EPUB file
            
        Returns:
            Dict with book structure (content and subsections)
        """
        print(f"📖 Reading EPUB file: {epub_path}")
        book = epub.read_epub(epub_path)
        structure = {}

        for item in book.get_items():
            if item.get_type() == ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_body_content(), 'lxml')

                # Extract h1-h6 headers
                headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                if not headers:
                    continue

                # Stack to maintain current hierarchy
                hierarchy = {}  # {level: (title, dict_reference)}
                
                for i, header in enumerate(headers):
                    level = int(header.name[1])
                    title = header.get_text(strip=True)
                    
                    # Capture content until finding ANY header
                    content_parts = []
                    current = header.find_next_sibling()
                    
                    while current:
                        # Stop if current element IS a header
                        if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            break
                        
                        # Stop if element CONTAINS a header
                        if current.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                            break
                        
                        content_parts.append(str(current))
                        current = current.find_next_sibling()
                    
                    content_html = ''.join(content_parts)
                    
                    # Create entry for this header
                    entry = {
                        "content": content_html,
                        "subsections": {}
                    }
                    
                    # Clean hierarchy of equal or greater levels
                    hierarchy = {k: v for k, v in hierarchy.items() if k < level}
                    
                    # Add to structure according to level
                    if level == 1:
                        # h1 is root level
                        structure[title] = entry
                        hierarchy[1] = (title, structure[title])
                    else:
                        # Find parent (immediately superior level)
                        parent_level = level - 1
                        while parent_level > 0:
                            if parent_level in hierarchy:
                                parent_title, parent_dict = hierarchy[parent_level]
                                parent_dict["subsections"][title] = entry
                                hierarchy[level] = (title, parent_dict["subsections"][title])
                                break
                            parent_level -= 1

        print(f"✓ Extraction complete: {len(structure)} main sections found")
        return structure
    
    def print_structure(self, structure: Dict[str, Any], indent: int = 0) -> None:
        """
        Print book structure in readable format
        
        Args:
            structure: Dictionary with structure
            indent: Indentation level
        """
        for title, info in structure.items():
            prefix = "  " * indent
            subsection_count = len(info.get("subsections", {}))
            content_length = len(info.get("content", ""))
            print(f"{prefix}→ {title} ({subsection_count} subsections, {content_length} chars)")
            
            if info.get("subsections"):
                self.print_structure(info["subsections"], indent + 1)


# Usage example
if __name__ == "__main__":
    extractor = EPUBExtractor()
    
    # Extract structure
    epub_path = "my_book.epub"
    structure = extractor.extract_structure(epub_path)
    
    # Print structure
    print("\n" + "="*80)
    print("BOOK STRUCTURE")
    print("="*80 + "\n")
    extractor.print_structure(structure)
    
    # Example: access specific content
    print("\n" + "="*80)
    print("EXAMPLE OF CONTENT ACCESS")
    print("="*80)
    
    for chapter_title, chapter_data in structure.items():
        print(f"\n📖 Chapter: {chapter_title}")
        content_preview = chapter_data["content"][:200] + "..." if len(chapter_data["content"]) > 200 else chapter_data["content"]
        print(f"Content preview: {content_preview}")
        break  # Only show first chapter as example