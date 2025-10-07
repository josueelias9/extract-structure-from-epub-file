from hola import EPUBExtractor
from ia_agent import AIAgent
from typing import Dict, Any
import json


class Mediator:
    """Mediator that connects EPUB Extractor with AI Agent to generate summaries"""
    
    def __init__(self, ollama_host: str = "http://ollama:11434"):
        self.extractor = EPUBExtractor()
        self.ai_agent = AIAgent(ollama_host)
    
    def process_epub(self, epub_path: str, generate_summaries: bool = True) -> Dict[str, Any]:
        """
        Process EPUB file: extract structure and optionally generate summaries
        
        Args:
            epub_path: Path to EPUB file
            generate_summaries: If True, generate AI summaries for all content
            
        Returns:
            Dict with complete book structure including summaries
        """
        # Step 1: Extract structure
        print("="*80)
        print("STEP 1: EXTRACTING STRUCTURE")
        print("="*80)
        structure = self.extractor.extract_structure(epub_path)
        
        # Step 2: Generate summaries if requested
        if generate_summaries:
            print("\n" + "="*80)
            print("STEP 2: GENERATING SUMMARIES WITH AI")
            print("="*80)
            
            # Check AI connection
            if not self.ai_agent.test_connection():
                print("⚠️  Could not connect to Ollama. Continuing without summaries.")
                return structure
            
            # Add summaries recursively
            self._add_summaries_recursive(structure)
            print("\n✓ All summaries generated successfully")
        
        return structure
    
    def _add_summaries_recursive(self, structure: Dict[str, Any], indent: int = 0) -> None:
        """
        Add AI-generated summaries recursively to entire structure
        
        Args:
            structure: Dictionary with book structure
            indent: Indentation level for logging
        """
        for title, info in structure.items():
            prefix = "  " * indent
            print(f"{prefix}📝 Summarizing: {title}")
            
            # Generate summary for content at this level
            if info.get("content"):
                info["summary"] = self.ai_agent.summarize_content(info["content"])
            else:
                info["summary"] = ""
            
            # Process subsections recursively
            if info.get("subsections"):
                self._add_summaries_recursive(info["subsections"], indent + 1)
    
    def print_structure(self, structure: Dict[str, Any], indent: int = 0, show_summaries: bool = True) -> None:
        """
        Print book structure in readable format
        
        Args:
            structure: Dictionary with structure
            indent: Indentation level
            show_summaries: If True, show summaries
        """
        for title, info in structure.items():
            prefix = "  " * indent
            subsection_count = len(info.get("subsections", {}))
            print(f"{prefix}→ {title} ({subsection_count} subsections)")
            
            if show_summaries and info.get("summary"):
                summary_preview = info["summary"][:150] + "..." if len(info["summary"]) > 150 else info["summary"]
                print(f"{prefix}  💡 {summary_preview}")
            
            if info.get("subsections"):
                self.print_structure(info["subsections"], indent + 1, show_summaries)
    
    def save_to_json(self, structure: Dict[str, Any], output_path: str) -> None:
        """
        Save complete structure with summaries to JSON file
        
        Args:
            structure: Book structure with summaries
            output_path: Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(structure, f, ensure_ascii=False, indent=2)
        print(f"✓ Structure saved to: {output_path}")
    
    def get_statistics(self, structure: Dict[str, Any]) -> Dict[str, int]:
        """
        Calculate statistics about the processed structure
        
        Args:
            structure: Book structure
            
        Returns:
            Dict with statistics
        """
        stats = {
            "total_sections": 0,
            "total_content_chars": 0,
            "total_summary_chars": 0,
            "sections_with_summaries": 0
        }
        
        def count_recursive(struct: Dict[str, Any]):
            for title, info in struct.items():
                stats["total_sections"] += 1
                stats["total_content_chars"] += len(info.get("content", ""))
                
                if info.get("summary"):
                    stats["sections_with_summaries"] += 1
                    stats["total_summary_chars"] += len(info["summary"])
                
                if info.get("subsections"):
                    count_recursive(info["subsections"])
        
        count_recursive(structure)
        return stats


# Usage example
if __name__ == "__main__":
    # Create mediator
    mediator = Mediator(ollama_host="http://ollama:11434")
    
    # Process EPUB with summaries
    epub_path = "my_book.epub"
    structure = mediator.process_epub(epub_path, generate_summaries=True)
    
    # Display structure with summaries
    print("\n" + "="*80)
    print("BOOK STRUCTURE WITH SUMMARIES")
    print("="*80 + "\n")
    mediator.print_structure(structure, show_summaries=True)
    
    # Show statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    stats = mediator.get_statistics(structure)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Save to JSON
    output_path = "book_with_summaries.json"
    mediator.save_to_json(structure, output_path)
    
    # Example: Access specific summary
    print("\n" + "="*80)
    print("EXAMPLE OF SUMMARY ACCESS")
    print("="*80)
    
    for chapter_title, chapter_data in structure.items():
        print(f"\n📖 Chapter: {chapter_title}")
        if chapter_data.get("summary"):
            print(f"💡 Summary: {chapter_data['summary']}")
        
        # Show first subsection if exists
        if chapter_data.get("subsections"):
            first_subsection = list(chapter_data["subsections"].keys())[0]
            print(f"\n  📄 Subsection: {first_subsection}")
            print(f"  💡 Summary: {chapter_data['subsections'][first_subsection].get('summary', 'N/A')}")
        
        break  # Only show first chapter as example