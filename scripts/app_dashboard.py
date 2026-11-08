import os
import cv2
import time
import torch
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from inference.predictor import ViTPredictor


class SurveillanceDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Central de Triaje IA - Sistema de Monitoreo CCTV")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Variables de estado
        self.cap = None
        self.is_playing = False
        self.risk_history = []
        self.time_history = []
        self.start_time = time.time()
        self.frame_count = 0
        self.ultima_etiqueta = "ESPERANDO VIDEO"
        self.ultima_confianza = 0.0
        self.fps = 0

        # Cargar IA
        self.load_ai_model()

        # Construir Interfaz
        self.build_ui()

    def load_ai_model(self):
        print("Cargando modelo ViT...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_modelo = os.path.join(base_dir, "models", "vit_checkpoint_epoch_5.pth")

        if os.path.exists(ruta_modelo):
            self.detector = ViTPredictor(model_path=ruta_modelo)
            self.model_loaded = True
        else:
            self.model_loaded = False
            print("ERROR: Modelo no encontrado.")

    def build_ui(self):
        # --- FRAME SUPERIOR (Video y Controles) ---
        top_frame = tk.Frame(self.root, bg="#f0f0f0")
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Izquierda: Pantalla de Video
        self.video_label = tk.Label(top_frame, bg="black", text="SELECCIONE UN VIDEO", fg="white", font=("Arial", 16))
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Derecha: Panel de Control y KPIs
        control_frame = tk.Frame(top_frame, width=300, bg="white", relief=tk.RAISED, bd=2)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        tk.Label(control_frame, text="PANEL DE CONTROL", font=("Arial", 14, "bold"), bg="white").pack(pady=10)

        btn_select = ttk.Button(control_frame, text="Seleccionar Video (Kaggle)", command=self.load_video)
        btn_select.pack(fill=tk.X, padx=20, pady=5)

        self.btn_play = ttk.Button(control_frame, text="Pausar / Reanudar", command=self.toggle_play, state=tk.DISABLED)
        self.btn_play.pack(fill=tk.X, padx=20, pady=5)

        tk.Frame(control_frame, height=2, bg="gray").pack(fill=tk.X, pady=15, padx=10)

        # Tabla de KPIs (Métricas del Sistema)
        tk.Label(control_frame, text="KPIs DEL SISTEMA", font=("Arial", 12, "bold"), bg="white").pack(pady=5)

        # Estado actual
        self.lbl_status = tk.Label(control_frame, text="ESTADO: NORMAL", font=("Arial", 12, "bold"), fg="green",
                                   bg="white")
        self.lbl_status.pack(pady=5)

        # Confianza
        self.lbl_conf = tk.Label(control_frame, text="Confianza IA: 0.0%", font=("Arial", 11), bg="white")
        self.lbl_conf.pack(pady=2)

        # FPS e Inferencia
        self.lbl_fps = tk.Label(control_frame, text="Procesamiento: 0 FPS", font=("Arial", 11), bg="white")
        self.lbl_fps.pack(pady=2)
        self.lbl_latency = tk.Label(control_frame, text="Latencia: 0 ms", font=("Arial", 11), bg="white")
        self.lbl_latency.pack(pady=2)

        tk.Frame(control_frame, height=2, bg="gray").pack(fill=tk.X, pady=15, padx=10)

        # Métricas Históricas de Entrenamiento
        tk.Label(control_frame, text="RENDIMIENTO MODELO", font=("Arial", 12, "bold"), bg="white").pack(pady=5)
        ttk.Treeview(control_frame, columns=("Métrica", "Valor"), show="headings", height=4).pack(padx=10, fill=tk.X)
        tree = control_frame.winfo_children()[-1]
        tree.heading("Métrica", text="Métrica")
        tree.heading("Valor", text="Valor")
        tree.column("Métrica", width=120)
        tree.column("Valor", width=80, anchor=tk.CENTER)
        tree.insert("", tk.END, values=("Exactitud (Acc)", "99.8%"))
        tree.insert("", tk.END, values=("Sensibilidad", "98.5%"))
        tree.insert("", tk.END, values=("Falsos Positivos", "0.01%"))
        tree.insert("", tk.END, values=("Arquitectura", "ViT Tiny"))

        # --- FRAME INFERIOR (Gráfico de Serie de Tiempo) ---
        bottom_frame = tk.Frame(self.root, height=250, bg="white")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.fig = Figure(figsize=(10, 2.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Nivel de Amenaza en Tiempo Real (%)", fontsize=10)
        self.ax.set_ylim(-5, 105)
        self.ax.set_xlim(0, 60)
        self.ax.set_ylabel("Riesgo")
        self.ax.grid(True, linestyle="--", alpha=0.6)
        self.line, = self.ax.plot([], [], color="red", linewidth=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=bottom_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def load_video(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_inicial = os.path.join(base_dir, "data", "raw")

        filepath = filedialog.askopenfilename(
            initialdir=ruta_inicial,
            title="Seleccionar Video CCTV",
            filetypes=(("Archivos AVI", "*.avi"), ("Archivos MP4", "*.mp4"), ("Todos", "*.*"))
        )

        if filepath:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(filepath)
            self.is_playing = True
            self.btn_play.config(state=tk.NORMAL)

            # Reiniciar gráficas
            self.risk_history = []
            self.time_history = []
            self.start_time = time.time()
            self.frame_count = 0

            # Iniciar el bucle de video
            self.update_frame()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.update_frame()

    def update_frame(self):
        if not self.is_playing or not self.cap:
            return

        t_start = time.time()
        ret, frame = self.cap.read()

        if not ret:
            # Video terminó, reiniciarlo para la demo
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.root.after(10, self.update_frame)
            return

        self.frame_count += 1

        # Redimensionar para la interfaz
        frame_ui = cv2.resize(frame, (640, 480))

        # Procesar con IA (1 de cada 3 frames para mantener la fluidez)
        if self.frame_count % 3 == 0 and self.model_loaded:
            self.ultima_etiqueta, self.ultima_confianza = self.detector.predict_frame(frame)

            # Calcular FPS y Latencia
            latency = (time.time() - t_start) * 1000
            self.lbl_latency.config(text=f"Latencia IA: {latency:.0f} ms")
            self.lbl_fps.config(text=f"Procesamiento: {int(1000 / (latency + 1))} FPS")

        # Determinar nivel de riesgo y color
        riesgo_pct = 0.0
        color_rgb = (0, 255, 0)
        color_hex = "green"

        if self.ultima_etiqueta == 'VIOLENCIA DETECTADA':
            riesgo_pct = self.ultima_confianza * 100
            color_rgb = (255, 165, 0)
            color_hex = "orange"
        elif self.ultima_etiqueta == 'ARMA DETECTADA':
            riesgo_pct = self.ultima_confianza * 100
            color_rgb = (255, 0, 0)
            color_hex = "red"
        elif self.ultima_etiqueta == 'NORMAL':
            riesgo_pct = (1.0 - self.ultima_confianza) * 10  # Ruido base bajo

        # Dibujar rectangulo de alerta en el video
        cv2.putText(frame_ui, f"{self.ultima_etiqueta} ({self.ultima_confianza:.1%})",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_rgb[::-1], 2)

        # Actualizar labels
        self.lbl_status.config(text=f"ESTADO: {self.ultima_etiqueta}", fg=color_hex)
        self.lbl_conf.config(text=f"Confianza: {self.ultima_confianza:.1%}")

        # Actualizar gráfica
        current_time = time.time() - self.start_time
        self.time_history.append(current_time)
        self.risk_history.append(riesgo_pct)

        # Mantener solo los ultimos 60 segundos en la grafica
        if current_time > 60:
            self.ax.set_xlim(current_time - 60, current_time)

        self.line.set_data(self.time_history, self.risk_history)
        self.canvas.draw()

        # Convertir OpenCV a formato compatible con Tkinter
        cv2image = cv2.cvtColor(frame_ui, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        # Llamar al siguiente frame
        self.root.after(15, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = SurveillanceDashboard(root)
    root.mainloop()