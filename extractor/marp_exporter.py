from typing import Dict, Any, Optional
import os
import json


class MarpExporter:
    """Exporter that converts book structure to Marp presentation format"""
    
    def __init__(self):
        """Initialize Marp exporter with static configuration"""
        pass


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

        # Title slide
        slides.append(self._generate_title_slide(title))

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
        
        # Skip titles containing specific keywords/phrases
        excluded_titles = {
            "Assessment Test",
            "Review Questions", 
            "Table of Contents",
            "Answers to Assessment Test",
            "Index",
            "About the Author",
            "Acknowledgments",
            "About the Technical Editors",
            "Introduction",
            "Book Summary Presentation",
            "AppendixAnswers to Review Questions"
        }
        
        # Check if title contains any of the excluded keywords
                # Check if title matches excluded titles
        if clean_title in excluded_titles:
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
