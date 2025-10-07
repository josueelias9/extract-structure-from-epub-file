from ebooklib import epub, ITEM_DOCUMENT
from bs4 import BeautifulSoup

def extract_structure(epub_path):
    book = epub.read_epub(epub_path)
    structure = {}

    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_body_content(), 'lxml')

            # Extraer encabezados h1-h6
            headers = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            if not headers:
                continue

            # Stack para mantener la jerarquía actual
            hierarchy = {}  # {level: (title, dict_reference)}
            
            for i, header in enumerate(headers):
                level = int(header.name[1])
                title = header.get_text(strip=True)
                
                # Capturar contenido hasta encontrar CUALQUIER encabezado
                content_parts = []
                current = header.find_next_sibling()  # ← CAMBIO: usar find_next_sibling en vez de find_next
                
                while current:
                    # Detener si el elemento actual ES un encabezado
                    if current.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        break
                    
                    # Detener si el elemento CONTIENE un encabezado
                    if current.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                        break
                    
                    content_parts.append(str(current))
                    current = current.find_next_sibling()  # ← CAMBIO: continuar con hermanos solamente
                
                content_html = ''.join(content_parts)
                
                # Crear la entrada para este encabezado
                entry = {
                    "content": content_html,
                    "subsections": {}
                }
                
                # Limpiar jerarquía de niveles iguales o mayores
                hierarchy = {k: v for k, v in hierarchy.items() if k < level}
                
                # Agregar a la estructura según el nivel
                if level == 1:
                    # h1 es nivel raíz
                    structure[title] = entry
                    hierarchy[1] = (title, structure[title])
                else:
                    # Buscar el padre (el nivel inmediatamente superior)
                    parent_level = level - 1
                    while parent_level > 0:
                        if parent_level in hierarchy:
                            parent_title, parent_dict = hierarchy[parent_level]
                            parent_dict["subsections"][title] = entry
                            hierarchy[level] = (title, parent_dict["subsections"][title])
                            break
                        parent_level -= 1

    return structure


# 🔍 Ejemplo de uso:
epub_path = "mi_libro.epub"
data = extract_structure(epub_path)

def print_structure(data, indent=0):
    """Función auxiliar para visualizar la estructura jerárquica"""
    for title, info in data.items():
        print("  " * indent + f"→ {title} ({len(info['subsections'])} subsecciones)")
        if info['subsections']:
            print_structure(info['subsections'], indent + 1)

print_structure(data)