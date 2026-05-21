import librosa
import librosa.display
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import io


class AudioProcessor:
    @staticmethod
    def _get_audio_duration(file_path):
        try:
            return librosa.get_duration(path=file_path)
        except TypeError:
            return librosa.get_duration(filename=file_path)

    @staticmethod
    def extract_features(file_path, duration=30, n_mels=128, target_shape=(128, 128)):
        """
        Extract Mel Spectrogram from the center of the song.
        Returns (features_array, spectrogram_rgb_image)
        """
        try:
            audio_duration = AudioProcessor._get_audio_duration(file_path)
            offset = max((audio_duration - duration) / 2, 0)
            load_duration = duration if audio_duration > duration else None

            y, sr = librosa.load(file_path, offset=offset, duration=load_duration)
            y, _ = librosa.effects.trim(y)

            S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=512, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)

            fig = plt.figure(figsize=(1.28, 1.28), dpi=100)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            librosa.display.specshow(S_dB, sr=sr, hop_length=512)

            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='png')
            io_buf.seek(0)

            import cv2
            img_arr = np.frombuffer(io_buf.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img.shape[:2] != target_shape:
                img = cv2.resize(img, target_shape)

            plt.close(fig)

            # Save debug image
            try:
                debug_dir = os.path.join(os.getcwd(), "debug_output")
                os.makedirs(debug_dir, exist_ok=True)
                debug_path = os.path.join(debug_dir, "last_processed.png")
                plt.imsave(debug_path, img)
            except Exception:
                pass

            features = img.astype(np.float32)
            features = np.expand_dims(features, axis=0)

            # Also return a larger spectrogram image for display in Streamlit
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            librosa.display.specshow(S_dB, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax2, fmax=8000)
            fig2.colorbar(ax2.collections[0], ax=ax2, format='%+2.0f dB')
            ax2.set_title('Mel Spectrogram')
            fig2.tight_layout()
            display_buf = io.BytesIO()
            fig2.savefig(display_buf, format='png', dpi=100)
            display_buf.seek(0)
            plt.close(fig2)

            return features, display_buf

        except Exception as e:
            print(f"Error processing spectrogram: {e}")
            return None, None
