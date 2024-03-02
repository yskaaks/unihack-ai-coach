import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import json


class LabelingApp:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = cv2.VideoCapture(video_source)
        self.frame_number = 0

        self.canvas = tk.Canvas(
            window,
            width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH),
            height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT),
        )
        self.canvas.pack()

        # Buttons for labeling
        self.buttons = {
            "Narrow": "narrow",
            "Neutral": "neutral",
            "Wide": "wide",
            "Feet Out": "feet_out",
            "Feet In": "feet_in",
            "Leaning Left": "leaning_left",
            "Leaning Right": "leaning_right",
            "Good Form": "good_form",
        }
        for text, label in self.buttons.items():
            btn = tk.Button(
                window, text=text, width=15, command=lambda l=label: self.label_frame(l)
            )
            btn.pack(side=tk.LEFT)

        # Navigation buttons
        self.btn_prev_frame = tk.Button(window, text="<<", command=self.prev_frame)
        self.btn_prev_frame.pack(side=tk.LEFT)
        self.btn_next_frame = tk.Button(window, text=">>", command=self.next_frame)
        self.btn_next_frame.pack(side=tk.LEFT)

        self.btn_save_labels = tk.Button(
            window, text="Save Labels", command=self.save_labels
        )
        self.btn_save_labels.pack(side=tk.LEFT)

        # Dictionary to store labels with frame number
        self.labels = {}

        self.update()

        self.window.mainloop()

    def prev_frame(self):
        self.frame_number = max(0, self.frame_number - 1)
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        self.update()

    def next_frame(self):
        self.frame_number = min(
            self.vid.get(cv2.CAP_PROP_FRAME_COUNT), self.frame_number + 1
        )
        self.update()

    def label_frame(self, label):
        if self.frame_number not in self.labels:
            self.labels[self.frame_number] = []
        self.labels[self.frame_number].append(label)
        print(f"Frame {self.frame_number} labeled as: {label}")

    def update(self):
        self.vid.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        ret, frame = self.vid.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.frame_number += 1

    def save_labels(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            with open(file_path, "w") as file:
                json.dump(self.labels, file, indent=4)
            print(f"Labels saved to {file_path}")


def main():
    try:
        video_path = (
            "/Users/yskakshiyap/Desktop/projects/unihack-ai-coach/videos/squats.mov"
        )
        if video_path:
            root = tk.Tk()
            LabelingApp(root, "Squat Labeling App", video_path)
        else:
            print("No video file provided. Exiting.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
