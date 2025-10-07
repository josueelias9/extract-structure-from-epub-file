from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

def extract_structure(epub_path):
    book = epub.read_epub(epub_path)
    structure = {}

    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), 'lxml')

            # Extraer encabezados h1-h3
            headers = soup.find_all(['h1', 'h2', 'h3'])
            if not headers:
                continue

            current_chapter = None
            current_h2 = None

            for i, header in enumerate(headers):
                level = int(header.name[1])
                title = header.get_text(strip=True)
                
                # Capturar contenido hasta encontrar CUALQUIER encabezado
                content_parts = []
                current = header.find_next()
                
                while current:
                    # Detener si el elemento actual ES un encabezado
                    if current.name in ['h1', 'h2', 'h3']:
                        break
                    
                    # Detener si el elemento CONTIENE un encabezado
                    if current.find(['h1', 'h2', 'h3']):
                        break
                    
                    content_parts.append(str(current))
                    current = current.find_next()
                
                content_html = ''.join(content_parts)

                if level == 1:
                    structure[title] = {
                        "content": content_html,
                        "subsections": {}
                    }
                    current_chapter = title
                    current_h2 = None

                elif level == 2 and current_chapter:
                    structure[current_chapter]["subsections"][title] = {
                        "content": content_html,
                        "subsections": {}
                    }
                    current_h2 = title

                elif level == 3 and current_chapter and current_h2:
                    structure[current_chapter]["subsections"][current_h2]["subsections"][title] = {
                        "content": content_html,
                        "subsections": {}
                    }

    return structure


# 🔍 Ejemplo de uso:
epub_path = "mi_libro.epub"
data = extract_structure(epub_path)

for chapter, info in data.items():
    print(chapter, "→", len(info["subsections"]), "subsecciones")