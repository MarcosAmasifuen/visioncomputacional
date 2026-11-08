import os
import cv2
import time
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from inference.predictor_3d import ViTPredictor3D


class DashboardDoctoral:
    def __init__(self, root):
        self.root = root
        self.root.title("PROTOTIPO DOCTORAL - Análisis Cinematográfico ViT-3D & XAI")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f4f6f9")

        self.cap = None
        self.is_playing = False
        self.risk_history = []
        self.time_history = []
        self.start_time = time.time()
        self.frame_count = 0
        self.ultima_etiqueta = "ESPERANDO VIDEO"
        self.ultima_confianza = 0.0
        self.riesgo_suavizado = 0.0
        self.ultimo_heatmap = None

        # --- NUEVAS VARIABLES PARA LA VENTANA XAI ---
        self.analysis_window = None
        self.analysis_label = None

        self.load_ai_model()
        self.build_ui()

    def load_ai_model(self):
        print("Cargando Motor Cinemático 3D con Explicabilidad...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_modelo = os.path.join(base_dir, "models", "spatiotemporal_vit_epoch_8.pth")

        if os.path.exists(ruta_modelo):
            self.detector = ViTPredictor3D(model_path=ruta_modelo, seq_length=5)
            self.model_loaded = True
        else:
            self.model_loaded = False
            print("ERROR: Modelo Espacio-Temporal no encontrado.")

    def build_ui(self):
        top_frame = tk.Frame(self.root, bg="#f4f6f9")
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.video_label = tk.Label(top_frame, bg="#1e1e1e", text="SISTEMA STANDBY\nRequiere conexión CCTV", fg="white",
                                    font=("Arial", 16))
        self.video_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        control_frame = tk.Frame(top_frame, width=300, bg="white", relief=tk.RIDGE, bd=2)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

        tk.Label(control_frame, text="CONTROL DE MANDO", font=("Arial", 13, "bold"), bg="white", fg="#2c3e50").pack(
            pady=10)

        btn_select = ttk.Button(control_frame, text="Cargar Video (Dataset SCVD)", command=self.load_video)
        btn_select.pack(fill=tk.X, padx=20, pady=5)

        self.btn_play = ttk.Button(control_frame, text="Pausar / Reanudar", command=self.toggle_play, state=tk.DISABLED)
        self.btn_play.pack(fill=tk.X, padx=20, pady=5)

        # --- NUEVO BOTÓN: ABRIR VENTANA XAI ---
        self.btn_xai = ttk.Button(control_frame, text="Ver Análisis XAI (Heatmap)", command=self.open_xai_window,
                                  state=tk.DISABLED)
        self.btn_xai.pack(fill=tk.X, padx=20, pady=5)

        tk.Frame(control_frame, height=2, bg="#e0e0e0").pack(fill=tk.X, pady=15, padx=10)

        tk.Label(control_frame, text="TELEMETRÍA EN VIVO", font=("Arial", 12, "bold"), bg="white", fg="#2c3e50").pack(
            pady=5)

        self.lbl_status = tk.Label(control_frame, text="ESTADO: INACTIVO", font=("Arial", 12, "bold"), fg="gray",
                                   bg="white")
        self.lbl_status.pack(pady=5)

        self.lbl_conf = tk.Label(control_frame, text="Certeza (Softmax): 0.0%", font=("Consolas", 11), bg="white")
        self.lbl_conf.pack(pady=2)

        self.lbl_fps = tk.Label(control_frame, text="FPS de Inferencia: 0", font=("Consolas", 11), bg="white")
        self.lbl_fps.pack(pady=2)

        tk.Frame(control_frame, height=2, bg="#e0e0e0").pack(fill=tk.X, pady=15, padx=10)

        tk.Label(control_frame, text="MÉTRICAS DE ARTEFACTO", font=("Arial", 12, "bold"), bg="white",
                 fg="#2c3e50").pack(pady=5)
        ttk.Treeview(control_frame, columns=("Var", "Val"), show="headings", height=5).pack(padx=10, fill=tk.X)
        tree = control_frame.winfo_children()[-1]
        tree.heading("Var", text="Variable")
        tree.heading("Val", text="Valor")
        tree.column("Var", width=120)
        tree.column("Val", width=80, anchor=tk.CENTER)
        tree.insert("", tk.END, values=("Arquitectura", "ViT 3D (Late-Fusion)"))
        tree.insert("", tk.END, values=("Explicabilidad", "Atención Multiventana"))
        tree.insert("", tk.END, values=("Precisión Validada", "97.99%"))
        tree.insert("", tk.END, values=("Filtro Señal", "EMA Asimétrico"))
        tree.insert("", tk.END, values=("Resolución", "T=5, 224x224"))

        bottom_frame = tk.Frame(self.root, height=250, bg="white")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)

        self.fig = Figure(figsize=(10, 2.5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Varianza de Amenaza Espacio-Temporal (Señal Suavizada)", fontsize=10)
        self.ax.set_ylim(-5, 105)
        self.ax.set_xlim(0, 30)
        self.ax.set_ylabel("Prob. Riesgo (%)")
        self.ax.grid(True, linestyle=":", alpha=0.7)
        self.line, = self.ax.plot([], [], color="#e74c3c", linewidth=2.5)

        self.canvas = FigureCanvasTkAgg(self.fig, master=bottom_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- NUEVA FUNCIÓN: CREAR Y GESTIONAR VENTANA SECUNDARIA ---
    def open_xai_window(self):
        # Si la ventana ya existe, la traemos al frente
        if self.analysis_window is not None and self.analysis_window.winfo_exists():
            self.analysis_window.focus()
            return

        # Crear nueva ventana
        self.analysis_window = tk.Toplevel(self.root)
        self.analysis_window.title("Auditoría XAI - Visión de Atención Neuronal")
        self.analysis_window.geometry("680x520")
        self.analysis_window.configure(bg="#1e1e1e")

        self.analysis_label = tk.Label(self.analysis_window, bg="black", text="ESPERANDO DATOS XAI...", fg="white",
                                       font=("Arial", 14))
        self.analysis_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Función para limpiar variables cuando el usuario cierre esta ventana
        def on_close():
            self.analysis_window.destroy()
            self.analysis_window = None
            self.analysis_label = None

        self.analysis_window.protocol("WM_DELETE_WINDOW", on_close)

    def toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.update_frame()

    def load_video(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ruta_inicial = os.path.join(base_dir, "data", "raw")

        filepath = filedialog.askopenfilename(initialdir=ruta_inicial, filetypes=[("Video Files", "*.avi *.mp4")])
        if filepath:
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(filepath)
            self.is_playing = True

            # Habilitar botones
            self.btn_play.config(state=tk.NORMAL)
            self.btn_xai.config(state=tk.NORMAL)

            self.detector.frame_buffer.clear()
            self.risk_history, self.time_history = [], []
            self.start_time = time.time()
            self.frame_count = 0
            self.riesgo_suavizado = 0.0
            self.ultimo_heatmap = None

            self.update_frame()

    def update_frame(self):
        if not self.is_playing or not self.cap: return

        t_start = time.time()
        ret, frame = self.cap.read()

        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.detector.frame_buffer.clear()
            self.riesgo_suavizado = 0.0
            self.ultimo_heatmap = None
            self.root.after(10, self.update_frame)
            return

        self.frame_count += 1
        frame_ui = cv2.resize(frame, (640, 480))

        # --- PROCESAMIENTO IA ---
        if self.model_loaded and self.frame_count % 3 == 0:
            etiqueta, conf, heatmap = self.detector.predict_frame(frame)
            self.ultima_etiqueta = etiqueta
            self.ultima_confianza = conf
            if heatmap is not None:
                self.ultimo_heatmap = heatmap

            latency = (time.time() - t_start) * 1000
            self.lbl_fps.config(text=f"Procesamiento: {int(1000 / (latency + 1))} FPS")

        # --- FILTRO EMA ASIMÉTRICO ---
        riesgo_raw = 0.0
        if self.ultima_etiqueta == 'VIOLENCIA DETECTADA':
            riesgo_raw = self.ultima_confianza * 100
        elif self.ultima_etiqueta == 'ARMA DETECTADA':
            riesgo_raw = self.ultima_confianza * 100
        elif self.ultima_etiqueta == 'NORMAL':
            riesgo_raw = (1.0 - self.ultima_confianza) * 5

        if self.ultima_etiqueta != 'ANALIZANDO CINEMÁTICA...':
            if riesgo_raw > self.riesgo_suavizado:
                self.riesgo_suavizado = (0.80 * riesgo_raw) + (0.20 * self.riesgo_suavizado)
            else:
                self.riesgo_suavizado = (0.01 * riesgo_raw) + (0.99 * self.riesgo_suavizado)

        # Lógica de colores y estados
        color_rgb = (46, 204, 113)
        color_hex = "#2ecc71"
        etiqueta_display = "NORMAL"

        if self.riesgo_suavizado > 40.0:
            if 'ARMA' in self.ultima_etiqueta:
                etiqueta_display = 'ARMA DETECTADA'
                color_rgb = (231, 76, 60)
                color_hex = "#e74c3c"
            else:
                etiqueta_display = 'VIOLENCIA DETECTADA'
                color_rgb = (243, 156, 18)
                color_hex = "#f39c12"
        elif self.ultima_etiqueta == 'ANALIZANDO CINEMÁTICA...':
            etiqueta_display = self.ultima_etiqueta
            color_rgb = (52, 152, 219)
            color_hex = "#3498db"

        # =========================================================
        # SEPARACIÓN DE CANALES DE VIDEO (Limpio vs XAI)
        # =========================================================

        # 1. CANAL PRINCIPAL (LIMPIO)
        frame_main = frame_ui.copy()
        cv2.putText(frame_main, f"{etiqueta_display}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, color_rgb[::-1], 2)
        if self.riesgo_suavizado > 0:
            cv2.putText(frame_main, f"Riesgo Sostenido: {self.riesgo_suavizado:.1f}%", (20, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, color_rgb[::-1], 1)

        # 2. CANAL SECUNDARIO (XAI / HEATMAP) - Solo se procesa si la ventana está abierta
        if self.analysis_window is not None and self.analysis_window.winfo_exists():
            frame_xai = frame_ui.copy()
            if self.ultimo_heatmap is not None:
                # Fusión agresiva al 60% para que el mapa de calor destaque en esta pantalla dedicada
                frame_xai = cv2.addWeighted(frame_xai, 0.4, self.ultimo_heatmap, 0.6, 0)

            # Etiqueta indicativa para la ventana secundaria
            cv2.putText(frame_xai, "MODO XAI: ATENCION SEMANTICA CLS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            # Enviar al Label secundario
            cv2image_xai = cv2.cvtColor(frame_xai, cv2.COLOR_BGR2RGBA)
            img_xai = Image.fromarray(cv2image_xai)
            imgtk_xai = ImageTk.PhotoImage(image=img_xai)
            self.analysis_label.imgtk = imgtk_xai
            self.analysis_label.configure(image=imgtk_xai)

        # =========================================================
        # ACTUALIZACIÓN DE GRÁFICOS Y TEXTOS
        # =========================================================
        self.lbl_status.config(text=f"ESTADO: {etiqueta_display}", fg=color_hex)
        self.lbl_conf.config(text=f"Certeza (Softmax): {self.ultima_confianza:.1%}")

        current_time = time.time() - self.start_time
        self.time_history.append(current_time)
        self.risk_history.append(self.riesgo_suavizado)

        if current_time > 30:
            self.ax.set_xlim(current_time - 30, current_time)

        self.line.set_data(self.time_history, self.risk_history)
        self.canvas.draw()

        # Enviar video limpio al Label principal
        cv2image = cv2.cvtColor(frame_main, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

        self.root.after(10, self.update_frame)


if __name__ == "__main__":
    root = tk.Tk()
    app = DashboardDoctoral(root)
    root.mainloop()