import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import csv
import torch
from ultralytics import YOLO


class AplicacionPuntosRecortes:
    def __init__(self, root):
        self.root = root
        self.root.title("Recortador de uvas")
        self.root.geometry("900x700")

        # Variables
        self.imagen_original = None
        self.imagen_actual = None
        self.foto = None
        self.ruta_imagen = None
        self.puntos = []
        self.recuadros = []
        self.directorio_base = "recortes"
        self.factor_escala = 1.0
        self.tamano_recorte = 150  # Tamaño predeterminado del recuadro (50x50 píxeles)
        self.modelo = None
        self.umbral_confianza = 0.5

        # Cargar modelo
        self.cargar_modelo()

        # Crear widgets
        self.crear_interfaz()

    def cargar_modelo(self):
        try:
            ruta_modelo = "modelos_uvas/detector_uvas/weights/best.pt"
            if os.path.exists(ruta_modelo):
                self.modelo = YOLO(ruta_modelo)
                print("Modelo cargado correctamente")
            else:
                print(f"No se encontró el modelo en {ruta_modelo}")
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")

    def crear_interfaz(self):
        # Panel superior (botones)
        panel_botones = tk.Frame(self.root)
        panel_botones.pack(fill=tk.X, padx=10, pady=10)

        btn_abrir = tk.Button(panel_botones, text="Abrir imagen", command=self.abrir_imagen)
        btn_abrir.pack(side=tk.LEFT, padx=5)

        btn_detectar = tk.Button(panel_botones, text="Detectar uvas", command=self.detectar_uvas)
        btn_detectar.pack(side=tk.LEFT, padx=5)

        btn_guardar = tk.Button(panel_botones, text="Guardar recortes", command=self.guardar_recortes)
        btn_guardar.pack(side=tk.LEFT, padx=5)

        btn_exportar_yolo = tk.Button(panel_botones, text="Exportar YOLO", command=self.exportar_formato_yolo)
        btn_exportar_yolo.pack(side=tk.LEFT, padx=5)

        btn_convertir_existentes = tk.Button(panel_botones, text="Convertir CSV a YOLO",
                                             command=self.convertir_csv_a_yolo)
        btn_convertir_existentes.pack(side=tk.LEFT, padx=5)

        # Control del tamaño del recuadro
        tk.Label(panel_botones, text="Tamaño recuadro:").pack(side=tk.LEFT, padx=(20, 5))
        self.entrada_tamano = ttk.Spinbox(panel_botones, from_=10, to=200, width=5,
                                          command=self.actualizar_recuadros)
        self.entrada_tamano.set(self.tamano_recorte)
        self.entrada_tamano.pack(side=tk.LEFT)

        # Control del umbral de confianza
        tk.Label(panel_botones, text="Umbral:").pack(side=tk.LEFT, padx=(20, 5))
        self.entrada_umbral = ttk.Spinbox(panel_botones, from_=0.1, to=1.0, increment=0.05, width=5)
        self.entrada_umbral.set(self.umbral_confianza)
        self.entrada_umbral.pack(side=tk.LEFT)

        # Canvas para la imagen
        self.canvas = tk.Canvas(self.root, bg='gray', cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Eventos del canvas
        self.canvas.bind("<Button-1>", self.colocar_punto)
        self.canvas.bind("<Button-3>", self.eliminar_punto)  # Botón derecho para eliminar

        # Barra de estado
        self.estado = tk.Label(self.root, text="Listo para importar una imagen", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.estado.pack(side=tk.BOTTOM, fill=tk.X)

    def detectar_uvas(self):
        if self.imagen_original is None or self.modelo is None:
            messagebox.showwarning("Aviso", "No se ha cargado una imagen o el modelo no está disponible")
            return

        try:
            # Obtener umbral de confianza
            try:
                umbral = float(self.entrada_umbral.get())
                if 0 < umbral <= 1:
                    self.umbral_confianza = umbral
            except ValueError:
                pass

            # Guardar temporalmente la imagen para procesarla con YOLO
            temp_path = "temp_detection.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(self.imagen_original, cv2.COLOR_RGB2BGR))

            # Realizar detección
            resultados = self.modelo(temp_path, conf=self.umbral_confianza)[0]

            # Limpiar puntos existentes
            self.limpiar_puntos()

            # Procesar resultados
            cajas = resultados.boxes.cpu().numpy()
            num_detecciones = len(cajas)

            for caja in cajas:
                # Obtener coordenadas (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, caja.xyxy[0])

                # Calcular centro
                centro_x = (x1 + x2) // 2
                centro_y = (y1 + y2) // 2

                # Escalar para la visualización en canvas
                x_canvas = int(centro_x * self.factor_escala)
                y_canvas = int(centro_y * self.factor_escala)

                # Crear punto en el canvas
                punto_id = self.canvas.create_oval(
                    x_canvas - 3, y_canvas - 3, x_canvas + 3, y_canvas + 3,
                    fill="red", outline="red", tags="punto"
                )

                # Guardar coordenadas originales
                self.puntos.append((punto_id, (centro_x, centro_y)))

                # Crear recuadro
                self.crear_recuadro(x_canvas, y_canvas)

            # Eliminar archivo temporal
            if os.path.exists(temp_path):
                os.remove(temp_path)

            self.estado.config(text=f"Se detectaron {num_detecciones} uvas con confianza > {self.umbral_confianza}")

        except Exception as e:
            messagebox.showerror("Error", f"Error al detectar uvas: {str(e)}")
            print(f"Error detallado: {str(e)}")

    def abrir_imagen(self):
        self.ruta_imagen = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp")]
        )

        if not self.ruta_imagen:
            return

        # Cargar la imagen
        self.imagen_original = cv2.imread(self.ruta_imagen)
        self.imagen_original = cv2.cvtColor(self.imagen_original, cv2.COLOR_BGR2RGB)
        self.mostrar_imagen()

        # Limpiar puntos y recuadros anteriores
        self.limpiar_puntos()
        self.estado.config(text=f"Imagen cargada: {os.path.basename(self.ruta_imagen)}")

    def mostrar_imagen(self):
        if self.imagen_original is None:
            return

        # Obtener dimensiones del canvas
        ancho_canvas = self.canvas.winfo_width()
        alto_canvas = self.canvas.winfo_height()

        # Si el canvas no tiene tamaño aún, usar valores por defecto
        if ancho_canvas < 10:
            ancho_canvas = 800
            alto_canvas = 600

        # Escalar la imagen para que se ajuste al canvas
        alto, ancho = self.imagen_original.shape[:2]
        self.factor_escala_ancho = ancho_canvas / ancho
        self.factor_escala_alto = alto_canvas / alto
        self.factor_escala = min(self.factor_escala_ancho, self.factor_escala_alto) * 0.95

        nuevo_ancho = int(ancho * self.factor_escala)
        nuevo_alto = int(alto * self.factor_escala)

        # Redimensionar imagen
        self.imagen_actual = cv2.resize(self.imagen_original.copy(), (nuevo_ancho, nuevo_alto))

        # Convertir a formato compatible con tkinter
        self.foto = ImageTk.PhotoImage(image=Image.fromarray(self.imagen_actual))

        # Mostrar en canvas
        self.canvas.config(width=nuevo_ancho, height=nuevo_alto)
        self.canvas.create_image(0, 0, image=self.foto, anchor=tk.NW, tags="imagen")

    def colocar_punto(self, event):
        if self.imagen_original is None:
            return

        # Dibujar un punto (círculo pequeño)
        punto_id = self.canvas.create_oval(
            event.x - 3, event.y - 3, event.x + 3, event.y + 3,
            fill="red", outline="red", tags="punto"
        )

        # Coordenadas ajustadas a la imagen original
        x_orig = int(event.x / self.factor_escala)
        y_orig = int(event.y / self.factor_escala)

        # Guardar punto y crear recuadro
        self.puntos.append((punto_id, (x_orig, y_orig)))
        self.crear_recuadro(event.x, event.y)

        self.estado.config(text=f"Punto añadido: {len(self.puntos)} en total")

    def eliminar_punto(self, event):
        if not self.puntos:
            return

        # Encontrar el punto más cercano al clic
        punto_mas_cercano = None
        distancia_minima = float('inf')
        indice_punto = -1

        for i, (punto_id, (x_orig, y_orig)) in enumerate(self.puntos):
            # Convertir coordenadas originales a coordenadas del canvas
            x_canvas = int(x_orig * self.factor_escala)
            y_canvas = int(y_orig * self.factor_escala)

            # Calcular distancia
            distancia = ((event.x - x_canvas) ** 2 + (event.y - y_canvas) ** 2) ** 0.5

            if distancia < distancia_minima:
                distancia_minima = distancia
                punto_mas_cercano = punto_id
                indice_punto = i

        # Si hay un punto cercano (a menos de 20 píxeles), eliminarlo
        if distancia_minima < 3 and punto_mas_cercano is not None:
            # Eliminar punto y su recuadro asociado
            self.canvas.delete(punto_mas_cercano)
            if indice_punto < len(self.recuadros):
                rect_id, _ = self.recuadros[indice_punto]
                self.canvas.delete(rect_id)

                # Eliminar de las listas
                self.puntos.pop(indice_punto)
                self.recuadros.pop(indice_punto)

                self.estado.config(text=f"Punto eliminado: quedan {len(self.puntos)}")

    def crear_recuadro(self, x_canvas, y_canvas):
        try:
            # Obtener el tamaño del recuadro desde la entrada
            tamano = int(self.entrada_tamano.get())
        except ValueError:
            tamano = self.tamano_recorte

        self.tamano_recorte = tamano
        mitad_tamano = tamano * self.factor_escala / 2

        # Coordenadas del recuadro en el canvas
        x1 = x_canvas - mitad_tamano
        y1 = y_canvas - mitad_tamano
        x2 = x_canvas + mitad_tamano
        y2 = y_canvas + mitad_tamano

        # Crear el recuadro
        rect_id = self.canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="green", width=2
        )

        # Coordenadas del recuadro en la imagen original
        x_orig = int(x_canvas / self.factor_escala)
        y_orig = int(y_canvas / self.factor_escala)
        mitad_orig = tamano // 2

        x1_orig = max(0, x_orig - mitad_orig)
        y1_orig = max(0, y_orig - mitad_orig)
        x2_orig = min(self.imagen_original.shape[1], x_orig + mitad_orig)
        y2_orig = min(self.imagen_original.shape[0], y_orig + mitad_orig)

        # Guardar información del recuadro
        self.recuadros.append((rect_id, (x1_orig, y1_orig, x2_orig, y2_orig)))

    def actualizar_recuadros(self):
        if not self.puntos or self.imagen_original is None:
            return

        # Eliminar recuadros actuales
        for rect_id, _ in self.recuadros:
            self.canvas.delete(rect_id)
        self.recuadros = []

        # Recrear recuadros con el nuevo tamaño
        for _, (x_orig, y_orig) in self.puntos:
            x_canvas = int(x_orig * self.factor_escala)
            y_canvas = int(y_orig * self.factor_escala)
            self.crear_recuadro(x_canvas, y_canvas)

    def limpiar_puntos(self):
        # Eliminar puntos y recuadros del canvas
        self.canvas.delete("punto")
        for rect_id, _ in self.recuadros:
            self.canvas.delete(rect_id)

        self.puntos = []
        self.recuadros = []
        self.estado.config(text="Puntos y recuadros eliminados")

    def exportar_formato_yolo(self):
        if not self.recuadros or self.imagen_original is None:
            messagebox.showwarning("Aviso", "No hay recuadros para exportar")
            return

        # Obtener nombre base de la imagen
        nombre_base = os.path.splitext(os.path.basename(self.ruta_imagen))[0]

        # Crear directorio específico para formato YOLO si no existe
        directorio_yolo = os.path.join(self.directorio_base, "yolo")
        if not os.path.exists(directorio_yolo):
            os.makedirs(directorio_yolo)

        # Crear directorio de imágenes si no existe
        directorio_images = os.path.join(directorio_yolo, "images")
        if not os.path.exists(directorio_images):
            os.makedirs(directorio_images)

        # Crear directorio de etiquetas si no existe
        directorio_labels = os.path.join(directorio_yolo, "labels")
        if not os.path.exists(directorio_labels):
            os.makedirs(directorio_labels)

        # Guardar la imagen original en formato YOLO
        ruta_imagen_yolo = os.path.join(directorio_images, f"{nombre_base}.jpg")
        cv2.imwrite(ruta_imagen_yolo, cv2.cvtColor(self.imagen_original, cv2.COLOR_RGB2BGR))

        # Crear archivo de etiquetas YOLO
        ruta_etiquetas = os.path.join(directorio_labels, f"{nombre_base}.txt")

        # Obtener dimensiones de la imagen original
        alto_img, ancho_img = self.imagen_original.shape[:2]

        with open(ruta_etiquetas, 'w') as archivo_yolo:
            for _, coords in self.recuadros:
                x1, y1, x2, y2 = coords

                # Calcular valores normalizados para formato YOLO
                centro_x = (x1 + x2) / 2.0 / ancho_img
                centro_y = (y1 + y2) / 2.0 / alto_img
                ancho_rect = (x2 - x1) / ancho_img
                alto_rect = (y2 - y1) / alto_img

                # Clase 0 para "uva"
                clase = 0

                # Escribir en formato YOLO: <clase> <centro_x> <centro_y> <ancho> <alto>
                archivo_yolo.write(f"{clase} {centro_x:.6f} {centro_y:.6f} {ancho_rect:.6f} {alto_rect:.6f}\n")

        # Crear o actualizar archivo classes.txt
        ruta_clases = os.path.join(directorio_yolo, "classes.txt")
        if not os.path.exists(ruta_clases):
            with open(ruta_clases, 'w') as archivo_clases:
                archivo_clases.write("uva\n")

        # Crear o actualizar archivo dataset.yaml para YOLOv5
        ruta_yaml = os.path.join(directorio_yolo, "dataset.yaml")
        with open(ruta_yaml, 'w') as archivo_yaml:
            archivo_yaml.write(f"path: {os.path.abspath(directorio_yolo)}\n")
            archivo_yaml.write("train: images/train\n")
            archivo_yaml.write("val: images/val\n")
            archivo_yaml.write("test: images/test\n\n")
            archivo_yaml.write("names:\n  0: uva\n")

        messagebox.showinfo("Éxito",
                            f"Se han exportado {len(self.recuadros)} anotaciones en formato YOLO en '{directorio_yolo}'")

    def convertir_csv_a_yolo(self):
        # Crear directorios YOLO si no existen
        directorio_yolo = os.path.join("recortes", "yolo")
        directorio_images = os.path.join(directorio_yolo, "images")
        directorio_labels = os.path.join(directorio_yolo, "labels")

        for directorio in [directorio_yolo, directorio_images, directorio_labels]:
            if not os.path.exists(directorio):
                os.makedirs(directorio)

        # Recorrer todos los subdirectorios en recortes
        imagenes_procesadas = 0
        for directorio in os.listdir("recortes"):
            ruta_directorio = os.path.join("recortes", directorio)
            if not os.path.isdir(ruta_directorio) or directorio == "yolo":
                continue

            # Buscar archivo CSV
            archivos_csv = [f for f in os.listdir(ruta_directorio) if f.endswith("_coordenadas.csv")]
            for csv_file in archivos_csv:
                ruta_csv = os.path.join(ruta_directorio, csv_file)
                nombre_base = csv_file.replace("_coordenadas.csv", "")

                # Copiar imagen original
                ruta_imagen_orig = f"imagenes/{nombre_base}.jpg"
                if os.path.exists(ruta_imagen_orig):
                    imagen = cv2.imread(ruta_imagen_orig)
                    if imagen is not None:
                        alto_img, ancho_img = imagen.shape[:2]
                        ruta_destino = os.path.join(directorio_images, f"{nombre_base}.jpg")
                        cv2.imwrite(ruta_destino, imagen)

                        # Crear archivo de etiquetas YOLO
                        ruta_etiquetas = os.path.join(directorio_labels, f"{nombre_base}.txt")
                        with open(ruta_csv, 'r') as archivo_csv, open(ruta_etiquetas, 'w') as archivo_yolo:
                            csv_reader = csv.DictReader(archivo_csv)
                            for fila in csv_reader:
                                x1, y1, x2, y2 = map(int, [fila['x1'], fila['y1'], fila['x2'], fila['y2']])

                                # Convertir a formato YOLO
                                centro_x = (x1 + x2) / 2.0 / ancho_img
                                centro_y = (y1 + y2) / 2.0 / alto_img
                                ancho_rect = (x2 - x1) / ancho_img
                                alto_rect = (y2 - y1) / alto_img

                                # Clase 0 para "uva"
                                archivo_yolo.write(
                                    f"0 {centro_x:.6f} {centro_y:.6f} {ancho_rect:.6f} {alto_rect:.6f}\n")

                        imagenes_procesadas += 1

        # Crear archivo classes.txt
        ruta_clases = os.path.join(directorio_yolo, "classes.txt")
        with open(ruta_clases, 'w') as archivo_clases:
            archivo_clases.write("uva\n")

        # Crear archivo dataset.yaml
        ruta_yaml = os.path.join(directorio_yolo, "dataset.yaml")
        with open(ruta_yaml, 'w') as archivo_yaml:
            archivo_yaml.write(f"path: {os.path.abspath(directorio_yolo)}\n")
            archivo_yaml.write("train: images/train\n")
            archivo_yaml.write("val: images/val\n")
            archivo_yaml.write("test: images/test\n\n")
            archivo_yaml.write("names:\n  0: uva\n")

        messagebox.showinfo("Éxito",
                            f"Se han convertido {imagenes_procesadas} imágenes y sus anotaciones al formato YOLO")

    def guardar_recortes(self):
        if not self.recuadros or self.imagen_original is None:
            messagebox.showwarning("Aviso", "No hay recuadros para guardar")
            return

        # Obtener nombre base de la imagen
        nombre_base = os.path.splitext(os.path.basename(self.ruta_imagen))[0]

        # Crear directorio específico para esta imagen
        directorio_salida = os.path.join(self.directorio_base, nombre_base)
        if not os.path.exists(directorio_salida):
            os.makedirs(directorio_salida)

        # Crear archivo CSV para las coordenadas
        ruta_csv = os.path.join(directorio_salida, f"{nombre_base}_coordenadas.csv")
        with open(ruta_csv, 'w', newline='') as archivo_csv:
            escritor = csv.writer(archivo_csv)
            # Escribir cabecera
            escritor.writerow(['nombre_archivo', 'x1', 'y1', 'x2', 'y2', 'centro_x', 'centro_y'])

            # Guardar cada recorte y sus coordenadas
            for i, (_, coords) in enumerate(self.recuadros):
                x1, y1, x2, y2 = coords
                recorte = self.imagen_original[y1:y2, x1:x2]

                # Calcular el centro del recuadro
                centro_x = (x1 + x2) // 2
                centro_y = (y1 + y2) // 2

                # Convertir de RGB a BGR para guardar con cv2
                recorte_bgr = cv2.cvtColor(recorte, cv2.COLOR_RGB2BGR)

                # Nombre del archivo: nombre_imagen_uva_N.png
                nombre_archivo = f"{nombre_base}_uva_{i + 1}.png"
                ruta_salida = os.path.join(directorio_salida, nombre_archivo)

                # Guardar recorte
                cv2.imwrite(ruta_salida, recorte_bgr)

                # Guardar coordenadas en CSV
                escritor.writerow([nombre_archivo, x1, y1, x2, y2, centro_x, centro_y])

        messagebox.showinfo("Éxito",
                            f"Se han guardado {len(self.recuadros)} recortes y sus coordenadas en '{directorio_salida}'")


# Iniciar aplicación
if __name__ == "__main__":
    root = tk.Tk()
    app = AplicacionPuntosRecortes(root)
    root.mainloop()
