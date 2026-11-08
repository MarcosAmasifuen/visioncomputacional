import os
import cv2


def extract_frames(raw_dir, processed_dir, frames_per_video=10):
    """
    Lee videos .avi de raw_dir, extrae fotogramas equiespaciados y los guarda como .jpg en processed_dir.
    """
    classes = ['Normal', 'Violence', 'Weaponized']

    for cls in classes:
        raw_class_dir = os.path.join(raw_dir, cls)
        processed_class_dir = os.path.join(processed_dir, cls)

        # Creamos las carpetas de destino si no existen
        os.makedirs(processed_class_dir, exist_ok=True)

        if not os.path.exists(raw_class_dir):
            continue

        # Listar todos los videos .avi
        videos = [v for v in os.listdir(raw_class_dir) if v.endswith('.avi')]
        print(f"Procesando {len(videos)} videos en la clase '{cls}'...")

        for video_name in videos:
            video_path = os.path.join(raw_class_dir, video_name)
            cap = cv2.VideoCapture(video_path)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                continue

            # Calcular intervalos para sacar 'N' frames repartidos en todo el video
            intervals = [int(i * total_frames / frames_per_video) for i in range(frames_per_video)]

            for count, frame_idx in enumerate(intervals):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Guardar el frame como imagen
                    out_name = f"{video_name.split('.')[0]}_frame{count}.jpg"
                    out_path = os.path.join(processed_class_dir, out_name)
                    cv2.imwrite(out_path, frame)

            cap.release()


if __name__ == "__main__":
    print("Iniciando extracción de frames...")

    # Autodetectar la ruta raíz exacta de tu proyecto
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_raw = os.path.join(base_dir, "data", "raw")
    ruta_processed = os.path.join(base_dir, "data", "processed")

    extract_frames(raw_dir=ruta_raw, processed_dir=ruta_processed, frames_per_video=10)
    print("¡Extracción completada exitosamente!")