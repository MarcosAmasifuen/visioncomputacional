import cv2
import os
from inference.predictor import ViTPredictor


def main():
    # 1. Cargar el modelo entrenado
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_modelo = os.path.join(base_dir, "models", "vit_checkpoint_epoch_5.pth")

    if not os.path.exists(ruta_modelo):
        print("Error: No se encontró el modelo entrenado en la ruta:", ruta_modelo)
        return

    print("Cargando modelo de IA Vision Transformer...")
    detector = ViTPredictor(model_path=ruta_modelo)

    # 2. Iniciar captura de video
    # Opcion A: Usar tu cámara web (0)
    # video_source = 0

    # Opcion B: Probar con un video de ARMAS del dataset de Kaggle
    video_source = os.path.join(base_dir, "data", "raw", "Weaponized", "w001_converted.avi")

    cap = cv2.VideoCapture(video_source)

    print("Iniciando monitoreo en tiempo real... (Presiona la tecla 'q' en la ventana de video para salir)")

    # Para no saturar tu computadora, procesaremos 1 de cada 5 frames (aprox 6 FPS, ideal para CCTV)
    frame_skip = 5
    frame_count = 0

    ultima_etiqueta = "Iniciando..."
    ultima_confianza = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            # Si el video termina, reiniciarlo en bucle (útil si pruebas con archivos .avi)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1

        # 3. Pasar el frame por la IA solo cada 'frame_skip' fotogramas
        if frame_count % frame_skip == 0:
            ultima_etiqueta, ultima_confianza = detector.predict_frame(frame)

        # 4. Interfaz Gráfica del Serenazgo (HUD)
        color = (0, 255, 0)  # Verde para NORMAL
        if ultima_etiqueta == 'VIOLENCIA DETECTADA':
            color = (0, 165, 255)  # Naranja (BGR en OpenCV)
        elif ultima_etiqueta == 'ARMA DETECTADA':
            color = (0, 0, 255)  # Rojo

        # Dibujar fondo negro transparente para el texto
        cv2.rectangle(frame, (10, 10), (600, 60), (0, 0, 0), -1)

        # Escribir la alerta y el porcentaje en pantalla
        texto = f"ALERTA: {ultima_etiqueta} ({ultima_confianza:.1%})"
        cv2.putText(frame, texto, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Mostrar la ventana
        cv2.imshow("Central de Serenazgo - Triaje IA", frame)

        # Salir si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()