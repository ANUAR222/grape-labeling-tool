# Herramienta de Etiquetado de Uvas

Una aplicación de escritorio desarrollada con Tkinter para el etiquetado manual de uvas visibles y la exportación de anotaciones en formato YOLO.

## Descripción General

Esta herramienta de etiquetado permite a los investigadores:
- Abrir imágenes de racimos de uvas
- Etiquetar manualmente las uvas visibles
- Detectar uvas automáticamente usando el modelo YOLOv8
- Generar archivos de exportación en formato YOLO para entrenar modelos de detección de objetos
- Visualizar tanto las uvas etiquetadas manualmente como las detectadas automáticamente

## Instalación

1. Clona el repositorio:
   ```
   git clone https://github.com/placeholder-username/grape-labeling-tool.git
   cd grape-labeling-tool
   ```

2. Instala las dependencias:
   ```
   pip install -r requirements.txt
   ```

## Requisitos

- Python 3.8+
- Tkinter
- PIL (Pillow)
- OpenCV (cv2)
- Ultralytics (para YOLOv8)

## Uso

Ejecuta la aplicación:
```
python src/app.py
```

### Características de la Interfaz

- **Abrir Imagen**: Carga una imagen para etiquetar
- **Detectar Uvas**: Detecta automáticamente las uvas utilizando un modelo YOLOv8 preentrenado
- **Guardar Recortes**: Guarda recortes individuales de las uvas etiquetadas
- **Exportar YOLO**: Exporta las anotaciones en formato YOLO para entrenar modelos de detección de objetos
- **Configuración Ajustable**: Cambia el tamaño del cuadro delimitador y el umbral de confianza

### Controles

- **Clic Izquierdo**: Coloca un punto/cuadro delimitador
- **Clic Derecho**: Elimina el punto más cercano
- **Ajustar Tamaño de Caja**: Cambia el tamaño de los cuadros delimitadores

## Licencia

Este proyecto está bajo la Licencia MIT - consulta el archivo LICENSE para más detalles.
