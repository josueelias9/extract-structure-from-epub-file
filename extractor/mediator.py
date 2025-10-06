#!/usr/bin/env python3
"""
Mediator para comunicación con el servicio Ollama

Este módulo proporciona funciones para interactuar con el servicio Ollama
que está ejecutándose en el contenedor ollama-container.
"""

import requests
import json
import os
import time
from typing import Dict, Any, Optional


class OllamaMediator:
    def __init__(self, host: str = None):
        """
        Inicializa el mediador de Ollama.
        
        Args:
            host: URL del servicio Ollama. Si no se proporciona, usa la variable de entorno OLLAMA_HOST
        """
        self.host = host or os.getenv('OLLAMA_HOST', 'http://ollama:11434')
        self.session = requests.Session()
        self.session.timeout = 30
        
    def check_connection(self) -> bool:
        """
        Verifica si el servicio Ollama está disponible.
        
        Returns:
            bool: True si el servicio está disponible, False en caso contrario
        """
        try:
            response = self.session.get(f"{self.host}/api/tags")
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return False
    
    def list_models(self) -> Optional[Dict[str, Any]]:
        """
        Lista los modelos disponibles en Ollama.
        
        Returns:
            Dict con información de los modelos o None si hay error
        """
        try:
            response = self.session.get(f"{self.host}/api/tags")
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error listing models: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error listing models: {e}")
            return None
    
    def pull_model(self, model_name: str) -> bool:
        """
        Descarga un modelo si no está disponible.
        
        Args:
            model_name: Nombre del modelo a descargar
            
        Returns:
            bool: True si se descargó correctamente, False en caso contrario
        """
        try:
            print(f"Pulling model {model_name}...")
            response = self.session.post(
                f"{self.host}/api/pull",
                json={"name": model_name},
                stream=True
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        if "status" in data:
                            print(f"Status: {data['status']}")
                        if data.get("status") == "success":
                            return True
                return True
            else:
                print(f"Error pulling model: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Error pulling model: {e}")
            return False
    
    def generate_response(self, prompt: str, model: str = "llama3.2", stream: bool = False) -> Optional[str]:
        """
        Genera una respuesta usando el modelo especificado.
        
        Args:
            prompt: La pregunta o prompt a enviar
            model: Nombre del modelo a usar (default: llama3.2)
            stream: Si usar streaming o no
            
        Returns:
            str: La respuesta generada o None si hay error
        """
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream
            }
            
            print(f"Sending request to Ollama with model: {model}")
            print(f"Prompt: {prompt}")
            
            response = self.session.post(
                f"{self.host}/api/generate",
                json=payload,
                stream=stream
            )
            
            if response.status_code == 200:
                if stream:
                    # Manejo de respuesta en streaming
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                full_response += data["response"]
                                print(data["response"], end="", flush=True)
                            if data.get("done", False):
                                break
                    print()  # Nueva línea al final
                    return full_response
                else:
                    # Respuesta completa
                    data = response.json()
                    return data.get("response", "No response received")
            else:
                print(f"Error generating response: {response.status_code}")
                print(f"Response: {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error generating response: {e}")
            return None
    
    def ask_question(self, question: str, model: str = "llama3.2") -> Optional[str]:
        """
        Función simplificada para hacer una pregunta.
        
        Args:
            question: La pregunta a hacer
            model: Modelo a usar (default: llama3.2)
            
        Returns:
            str: La respuesta o None si hay error
        """
        # Verificar conexión primero
        if not self.check_connection():
            print("❌ No se puede conectar al servicio Ollama")
            print("Asegúrate de que el contenedor ollama-container esté ejecutándose")
            return None
        
        # Verificar si el modelo está disponible
        models = self.list_models()
        if models:
            available_models = [m["name"] for m in models.get("models", [])]
            if not any(model in m for m in available_models):
                print(f"Modelo {model} no encontrado. Intentando descargarlo...")
                if not self.pull_model(model):
                    print(f"❌ No se pudo descargar el modelo {model}")
                    return None
        
        # Generar respuesta
        return self.generate_response(question, model)


def test_ollama_connection():
    """
    Función de prueba para verificar la conexión con Ollama.
    """
    print("🔍 Probando conexión con Ollama...")
    
    mediator = OllamaMediator()
    
    # Verificar conexión
    if mediator.check_connection():
        print("✅ Conexión exitosa con Ollama")
        
        # Listar modelos disponibles
        models = mediator.list_models()
        if models:
            print("\n📚 Modelos disponibles:")
            for model in models.get("models", []):
                print(f"  - {model['name']}")
        
        # Hacer una pregunta de prueba
        question = "¿Qué es un archivo EPUB y cómo está estructurado?"
        print(f"\n🤔 Pregunta: {question}")
        print("\n💬 Respuesta:")
        print("-" * 50)
        
        response = mediator.ask_question(question)
        if response:
            print(response)
            print("-" * 50)
            print("✅ Prueba completada exitosamente")
        else:
            print("❌ No se pudo obtener respuesta")
    else:
        print("❌ No se pudo conectar a Ollama")
        print("Verifica que el servicio esté ejecutándose con: docker compose ps")


if __name__ == "__main__":
    test_ollama_connection()