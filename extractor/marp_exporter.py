from typing import Dict, Any, Optional
import os
import json


class MarpExporter:
    """Exporter that converts book structure to Marp presentation format"""
    
    def __init__(self):
        """Initialize Marp exporter with static configuration"""
        pass
    
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

    def export_from_json(
        self,
        json_path: str,
        output_path: str,
        title: Optional[str] = None,
        include_summaries: bool = True,
        include_content: bool = False,
        max_depth: int = 3,
    ) -> None:
        """
        Load a saved book structure (book_with_summaries.json) and export to Marp.

        This method ensures strict slide separation without extra blank lines:
        each slide is separated by a single line containing '---'.

        Args:
            json_path: Path to JSON file containing the book structure with summaries
            output_path: Path to save the Marp markdown file
            title: Presentation title (defaults to first top-level section or a generic title)
            include_summaries: Include summary slides
            include_content: Include full content slides (can be very long)
            max_depth: Maximum heading depth to include
        """
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON structure not found: {json_path}")

        # Ensure output directory exists
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # Load structure from JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            structure: Dict[str, Any] = json.load(f)

        # Derive title if not provided
        if not title:
            title = next(iter(structure.keys()), "Book Summary Presentation")

        # Build slides (without writing yet)
        slides: list[str] = []

        # Title slide and TOC
        slides.append(self._generate_title_slide(title))
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

        # Prepare front matter
        front_matter = self._generate_front_matter()

        # Compose final content: front matter block + strict slide separation
        final_content = front_matter + '\n' + ''.join(slides)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)

        print(f"✓ Marp presentation exported from JSON to: {output_path}")
        print(f"  Source JSON: {json_path}")
        print(f"  Total slides: {len(slides)}")
        print(f"\n💡 To preview: Open {output_path} in VSCode with Marp extension")
    
    def _generate_front_matter(self) -> str:
        """Generate Marp front matter configuration"""
        return """---
marp: true
theme: default
paginate: true
footer: 'Eng. Josué Huamán'
style: |
  section {
    background-image: url('https://upload.wikimedia.org/wikipedia/commons/5/51/Google_Cloud_logo.svg');
    background-size: 250px;
    background-position: 95% 90%; /* esquina inferior derecha */
    background-repeat: no-repeat;
    opacity: 1;
  }

---

![bg](https://upload.wikimedia.org/wikipedia/commons/5/51/Google_Cloud_logo.svg)

---"""
    
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
    
    def _should_skip_title(self, title: str) -> bool:
        """
        Check if a title should be skipped based on exclusion criteria
        
        Args:
            title: The section title to check
            
        Returns:
            True if the title should be skipped, False otherwise
        """
        # Clean the title for comparison
        clean_title = title.strip()
        
        # Skip specific titles
        excluded_titles = {
            "Assessment Test",
            "Review Questions", 
            "Table of Contents",
            "Answers to Assessment Test"
        }
        
        # Check if title matches excluded titles
        if clean_title in excluded_titles:
            return True
        
        # Check if title is a single uppercase letter (A, B, C, etc.)
        if len(clean_title) == 1 and clean_title.isupper() and clean_title.isalpha():
            return True
        
        return False
    
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


            if "Translating Business Use Case" in title:
                print("Found it!")

            # Skip excluded titles
            if self._should_skip_title(title):
                continue
            
            # Generate section number
            section_number = f"{parent_number}{idx}." if parent_number else f"{idx}."
            
            # Generate section title slide
            slides.append(self._generate_section_slide(title, level, section_number))
            
            # Add summary slide if available
            if include_summaries and info.get("summary"):
                slides.append(self._generate_summary_slide(info["summary"]))
            
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
            return f"""# {section_number} {title}

---

"""
        elif level == 2:
            return f"""## {section_number} {title}

"""
        elif level ==3:
            return f"""### {section_number} {title}

"""
        else:
            return f"""#### {section_number} {title}

"""
    
    def _generate_summary_slide(self, summary: str) -> str:
        """Generate a summary slide"""
        # Split summary into chunks if too long
        max_chars = 700
        
        if len(summary) <= max_chars:
            return f"""{summary}

---

"""
        else:
            # Split into multiple slides by lines to preserve formatting
            lines = summary.split('\n')
            chunks = []
            current_chunk_lines = []
            current_length = 0
            
            for line in lines:
                line_length = len(line) + 1  # +1 for the newline character
                
                # If adding this line would exceed max_chars, start a new chunk
                if current_length + line_length > max_chars and current_chunk_lines:
                    chunks.append('\n'.join(current_chunk_lines))
                    current_chunk_lines = [line]
                    current_length = line_length
                else:
                    current_chunk_lines.append(line)
                    current_length += line_length
            
            # Add remaining lines as the last chunk
            if current_chunk_lines:
                chunks.append('\n'.join(current_chunk_lines))
            
            # Generate slides from chunks
            slides = []
            for i, chunk in enumerate(chunks, 1):
                slides.append(f"""{chunk}

---
""")
            
            return '\n'.join(slides)
    
    def _generate_content_slide(self, title: str, content: str, section_number: str) -> str:
        """Generate a content slide"""
        # Limit content length
        max_chars = 1000
        
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        
        return f"""### {section_number} Content: {title}

{content}

"""
    
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
    
    # Create exporter
    exporter = MarpExporter()
    
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