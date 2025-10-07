from typing import Dict, Any, Optional
import os


class MarpExporter:
    """Exporter that converts book structure to Marp presentation format"""
    
    def __init__(
        self,
        theme: str = "default",
        paginate: bool = True,
        header: Optional[str] = None,
        footer: Optional[str] = None,
        background_color: str = "#fff",
        color: str = "#333"
    ):
        """
        Initialize Marp exporter
        
        Args:
            theme: Marp theme (default, gaia, uncover)
            paginate: Show page numbers
            header: Header text for all slides
            footer: Footer text for all slides
            background_color: Background color
            color: Text color
        """
        self.theme = theme
        self.paginate = paginate
        self.header = header
        self.footer = footer
        self.background_color = background_color
        self.color = color
    
    def export_to_marp(
        self,
        structure: Dict[str, Any],
        output_path: str,
        title: str = "Book Presentation",
        include_summaries: bool = True,
        include_content: bool = False,
        max_depth: int = 3
    ) -> None:
        """
        Export book structure to Marp presentation
        
        Args:
            structure: Book structure dictionary
            output_path: Path to save the Marp markdown file
            title: Presentation title
            include_summaries: Include summary slides
            include_content: Include full content (can make presentation very long)
            max_depth: Maximum heading depth to include
        """
        slides = []
        
        # Add front matter (Marp configuration)
        slides.append(self._generate_front_matter())
        
        # Add title slide
        slides.append(self._generate_title_slide(title))
        
        # Add table of contents
        slides.append(self._generate_toc(structure))
        
        # Generate slides from structure
        self._process_structure_recursive(
            structure,
            slides,
            level=1,
            include_summaries=include_summaries,
            include_content=include_content,
            max_depth=max_depth
        )
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n---\n\n'.join(slides))
        
        print(f"✓ Marp presentation exported to: {output_path}")
        print(f"  Total slides: {len(slides)}")
        print(f"\n💡 To preview: Open {output_path} in VSCode with Marp extension")
    
    def _generate_front_matter(self) -> str:
        """Generate Marp front matter configuration"""
        front_matter = [
            "---",
            "marp: true",
            f"theme: {self.theme}",
            f"paginate: {str(self.paginate).lower()}",
        ]
        
        if self.header:
            front_matter.append(f"header: '{self.header}'")
        
        if self.footer:
            front_matter.append(f"footer: '{self.footer}'")
        
        front_matter.extend([
            f"backgroundColor: {self.background_color}",
            f"color: {self.color}",
            "---"
        ])
        
        return '\n'.join(front_matter)
    
    def _generate_title_slide(self, title: str) -> str:
        """Generate title slide"""
        return f"""<!-- _class: lead -->
<!-- _paginate: false -->

# {title}

### AI-Generated Book Summary

---"""
    
    def _generate_toc(self, structure: Dict[str, Any]) -> str:
        """Generate table of contents slide"""
        toc_lines = [
            "<!-- _class: lead -->",
            "",
            "## 📚 Table of Contents",
            ""
        ]
        
        for i, (title, info) in enumerate(structure.items(), 1):
            subsection_count = len(info.get("subsections", {}))
            toc_lines.append(f"{i}. **{title}** ({subsection_count} sections)")
        
        return '\n'.join(toc_lines)
    
    def _process_structure_recursive(
        self,
        structure: Dict[str, Any],
        slides: list,
        level: int,
        include_summaries: bool,
        include_content: bool,
        max_depth: int,
        parent_number: str = ""
    ) -> None:
        """
        Recursively process structure and generate slides
        
        Args:
            structure: Structure dictionary
            slides: List to append slides to
            level: Current heading level
            include_summaries: Include summary slides
            include_content: Include content slides
            max_depth: Maximum depth to process
            parent_number: Parent section numbering (e.g., "1.2")
        """
        if level > max_depth:
            return
        
        for idx, (title, info) in enumerate(structure.items(), 1):
            # Generate section number
            section_number = f"{parent_number}{idx}." if parent_number else f"{idx}."
            
            # Generate section title slide
            slides.append(self._generate_section_slide(title, level, section_number))
            
            # Add summary slide if available
            if include_summaries and info.get("summary"):
                slides.append(self._generate_summary_slide(title, info["summary"], section_number))
            
            # Add content slide if requested
            if include_content and info.get("content"):
                slides.append(self._generate_content_slide(title, info["content"], section_number))
            
            # Process subsections recursively
            if info.get("subsections"):
                self._process_structure_recursive(
                    info["subsections"],
                    slides,
                    level + 1,
                    include_summaries,
                    include_content,
                    max_depth,
                    section_number
                )
    
    def _generate_section_slide(self, title: str, level: int, section_number: str) -> str:
        """Generate a section title slide"""
        # Use different styles based on level
        if level == 1:
            return f"""<!-- _class: lead -->

# {section_number} {title}

---"""
        elif level == 2:
            return f"""## {section_number} {title}

---"""
        else:
            return f"""### {section_number} {title}

---"""
    
    def _generate_summary_slide(self, title: str, summary: str, section_number: str) -> str:
        """Generate a summary slide"""
        # Split summary into chunks if too long
        max_chars = 800
        
        if len(summary) <= max_chars:
            return f"""### {section_number} Summary: {title}

{summary}

---"""
        else:
            # Split into multiple slides
            words = summary.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                if current_length + len(word) + 1 > max_chars:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += len(word) + 1
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
            
            slides = []
            for i, chunk in enumerate(chunks, 1):
                part_label = f" (Part {i}/{len(chunks)})" if len(chunks) > 1 else ""
                slides.append(f"""### {section_number} Summary: {title}{part_label}

{chunk}

---""")
            
            return '\n\n---\n\n'.join(slides)
    
    def _generate_content_slide(self, title: str, content: str, section_number: str) -> str:
        """Generate a content slide"""
        # Limit content length
        max_chars = 1000
        
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        
        return f"""### {section_number} Content: {title}

{content}

---"""
    
    def export_chapter(
        self,
        chapter_title: str,
        chapter_data: Dict[str, Any],
        output_path: str,
        include_summaries: bool = True,
        include_content: bool = False
    ) -> None:
        """
        Export a single chapter to Marp presentation
        
        Args:
            chapter_title: Chapter title
            chapter_data: Chapter data dictionary
            output_path: Output file path
            include_summaries: Include summaries
            include_content: Include content
        """
        # Create a temporary structure with just this chapter
        temp_structure = {chapter_title: chapter_data}
        
        self.export_to_marp(
            temp_structure,
            output_path,
            title=chapter_title,
            include_summaries=include_summaries,
            include_content=include_content
        )


# Usage example
if __name__ == "__main__":
    # Example structure (normally comes from mediator)
    example_structure = {
        "Chapter 1: Introduction": {
            "content": "This is the introduction content with detailed explanations...",
            "summary": "Introduction to the main concepts and overview of what will be covered.",
            "subsections": {
                "Section 1.1: Getting Started": {
                    "content": "Getting started content...",
                    "summary": "Basic setup and initial configuration steps.",
                    "subsections": {}
                },
                "Section 1.2: Prerequisites": {
                    "content": "Prerequisites content...",
                    "summary": "Required knowledge and tools before starting.",
                    "subsections": {}
                }
            }
        },
        "Chapter 2: Main Content": {
            "content": "Main content here...",
            "summary": "Core concepts and detailed explanations of the main topics.",
            "subsections": {}
        }
    }
    
    # Create exporter with custom configuration
    exporter = MarpExporter(
        theme="default",
        paginate=True,
        footer="Generated by EPUBMediator",
        background_color="#fff",
        color="#333"
    )
    
    # Export full presentation
    exporter.export_to_marp(
        structure=example_structure,
        output_path="presentation.md",
        title="My Book Presentation",
        include_summaries=True,
        include_content=False,  # Set to True to include full content
        max_depth=3
    )
    
    print("\n✓ Done! Open presentation.md in VSCode with Marp extension")
    print("  You can also export to PDF/PPTX from VSCode")