import sys
#import cv2
import neoapi
from PIL import Image, ImageTk
import numpy as np
import torch
import time
import threading
import tkinter as tk
import os
from tkinter import ttk
import time
#import openpyxl
#from openpyxl import Workbook, load_workbook
import pandas as pd
import matplotlib.pyplot as plt


# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('C:/Users/pc/Downloads/yolov5-master', 'custom', path='D:/Mallesh/stations/ssv/runs/train/exp4/weights/best.pt', source='local')

"""
def yolomodel(img):
    # Run inference
    start_time = time.time()
    results = model(img)
    end_time = time.time()

    processing_time = (end_time - start_time) * 1000
    print("Processing Time: {:.2f} milliseconds".format(processing_time))

    return results

"""
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry("1200x1000")

        title_bar = tk.Frame(self, bg="lightgray")
        title_bar.pack(side="top", fill="x")

        self.navigation_frame = tk.Frame(self, bg="lightblue", width=400)
        self.navigation_frame.pack(side="left", fill="y")
        self.navigation_frame.pack_propagate(False)

        nav_frame_title_label = tk.Label(self.navigation_frame, text="SSV Cosmetic Inspection", font=("Arial", 20, "bold"), bg="lightgray")
        nav_frame_title_label.pack(pady=10)

        self.home_frame = tk.Frame(self, bg="white")
        self.home_frame.pack(side="left", fill="both", expand=True)

        # Add a title label to the home_frame
        home_title = tk.Label(self.home_frame, text="Defect Detection and Classification using Machine Learning", font=("Arial", 20, "bold"), bg="lightgray")
        home_title.pack(fill = "x")

        self.home_button = tk.Button(self.navigation_frame, text="Home", font=("Arial", 16), command=self.show_home_frame)
        self.home_button.pack(fill="x", pady=10)

        self.acquisition_button = tk.Button(self.navigation_frame, text="Start Acquisition", font=("Arial", 16), command=self.start_acquisition)
        self.acquisition_button.pack(fill="y", pady = 10)

        self.stop_button = tk.Button(self.navigation_frame, text="Stop Acquisition", font=("Arial", 16), command=self.stop_acquisition)
        self.stop_button.pack(fill="y", pady=10)

         # Initialize other components

        #self.camera_names = ["Top", "Neck", "Shoulder", "Body", "Bottom", "inside"]
        self.camera_names = ["Shoulder"]
        self.camera_selection_var = tk.StringVar()
        self.camera_selection_var.set("Select Camera")

        # Initialize other components

        # Initialize other components
        self.camera_dropdown = tk.OptionMenu(self.navigation_frame, self.camera_selection_var, *self.camera_names)
        self.camera_dropdown.pack(side="top", fill="y", padx=10, pady=10)

        self.show_defect_button = tk.Button(self.navigation_frame, text="Show Report", font=("Arial", 16), command=self.show_defect_table)
        self.show_defect_button.pack(side="top", fill="y", padx=10, pady=10)

        self.reset_button = tk.Button(self.navigation_frame, text="Reset", font=("Arial", 16), command=self.reset_app, bg="red", fg="white")
        self.reset_button.pack(side = "bottom", fill="y", pady=10)

        self.graph_button = tk.Button(self.navigation_frame, text="Show Graph", font=("Arial", 16), command=self.show_defect_graph)
        self.graph_button.pack(fill="y", pady=10)

        self.stats_frame_position = None    
        self.close_button = None
        self.defect_tree = None
        
        self.total_images_processed = 0
        self.defected_images = 0
        self.no_defect_images = 0
        self.create_image_stat_table()
        self.update_image_stat_table()
        self.create_defect_table()
        self.camera_selection_var.trace("w", lambda *args: self.update_defect_graph())

        #camera_names = ["Top", "Neck", "Shoulder", "Body", "Bottom", "inside"]
        camera_names = ["Shoulder"]
        self.detected_image_labels = {}
        self.detected_photo_list = [None] * len(camera_names)
        self.cameras = {}
        self.defect_counts = {}
        

        self.detected_frame = tk.Frame(self.home_frame, bg="white")
        self.detected_frame.pack(pady=10)

         #Create a frame to hold the detected images
        self.detected_frame = tk.Frame(self.home_frame, bg="white")
        self.detected_frame.pack(side="left", padx=30, pady=10, fill="both", expand=True)

        # Create a grid for detected images
        row_idx = 0
        for idx, camera_name in enumerate(camera_names):
            camera = neoapi.Cam()
            camera.Connect(camera_name)
            camera.f.ExposureTime.Set(600)
            #camera.SetImageBufferCount(1000)
            self.cameras[camera_name] = camera

            if idx % 2 == 0:  # Every even index (0, 2, 4) is for "Top" and "Neck"
                row_idx += 1

            # Add camera name label
            camera_name_label = tk.Label(self.detected_frame, text=f"{camera_name} - Image", font=("Arial", 12, "bold"))
            camera_name_label.grid(row=row_idx * 2, column=idx % 2, padx=10, pady=10, sticky="n")
            #camera_name_label[camera_name].config(anchor="center")
            # Add detected image label
            self.detected_image_labels[camera_name] = tk.Label(self.detected_frame, text=f"{camera_name} - Detected Image")
            self.detected_image_labels[camera_name].grid(row=row_idx * 2 + 1, column=idx % 2, padx=10, pady=10, sticky="n")
            self.detected_image_labels[camera_name].config(anchor="center")

            # Add detected photo label
            label = tk.Label(self.detected_frame, bg="white")
            label.grid(row=row_idx * 2 + 1, column=idx % 2, padx=70, pady=10, sticky="n")
            self.detected_photo_list[idx] = label

        self.is_acquiring = False
        self.detected_image_idx = 0
        self.defect_table_frame = tk.Frame(self.home_frame, bg="white")
        self.defect_tree = None


    def capture_images(self):
        while self.is_acquiring:
            for camera_name, camera in self.cameras.items():
                img = camera.GetImage()

                if not img.IsEmpty():
                    imgarray = img.GetNPArray()
                    imgarray = np.squeeze(imgarray, axis=2)
                    imgarray = np.stack((imgarray,) * 3, axis=-1)
                    start_time = time.time()
                    results = model(imgarray)
                    end_time = time.time()
                    processing_time = (end_time - start_time) * 1000
                    print("Processing Time: {:.2f} milliseconds".format(processing_time))
                    
                    detected_img_array = results.render()[0]
                    detected_img = Image.fromarray(detected_img_array.astype(np.uint8))
                    detected_img = detected_img.resize((500, 400))
                    detected_photo = ImageTk.PhotoImage(detected_img)

                    self.detected_image_labels[camera_name].config(image=detected_photo)
                    self.detected_image_labels[camera_name].image = detected_photo

                    class_indices = results.pred[0][:, -1].tolist()
                    detected_objects = [results.names[i] for i in class_indices]

                    if len(detected_objects) > 0:
                        self.defected_images += 1
                        if camera_name not in self.defect_counts:
                            self.defect_counts[camera_name] = {}
                        for defect in detected_objects:
                            self.defect_counts[camera_name][defect] = self.defect_counts[camera_name].get(defect, 0) + 1
                            self.update_excel_sheet(camera_name, detected_objects)
                            
                    else:
                        self.no_defect_images += 1

                    self.total_images_processed += 1

                    self.update_image_stat_table()

                    #image_name = f"{camera_name}_{time.strftime('%Y%m%d_%H%M%S')}.png"
                    # self.save_defects_to_file(camera_name, image_name, detected_objects)

                    if self.detected_image_idx < len(self.detected_photo_list):
                        self.detected_photo_list[self.detected_image_idx].config(image=detected_photo)
                        self.detected_photo_list[self.detected_image_idx].image = detected_photo
                        self.detected_image_idx += 1
                        if self.detected_image_idx >= len(self.detected_photo_list):
                            self.detected_image_idx = 0
                time.sleep(0.2)
                self.update()


    def update_excel_sheet(self, camera_name, detected_objects):
        excel_file = f"{camera_name}_defects.xlsx"

        if not os.path.exists(excel_file):
            df = pd.DataFrame(columns=["Defect", "Count"])
        else:
            df = pd.read_excel(excel_file)

        for defect in detected_objects:
            if "Defect" in df.columns and defect in df["Defect"].values:
                idx = df.index[df["Defect"] == defect].tolist()[0]
                df.loc[idx, "Count"] += 1
            else:
                new_row = {"Defect": defect, "Count": 1}
                df = df.append(new_row, ignore_index=True)

        df.to_excel(excel_file, index=False)


    def create_image_stat_table(self):
        # Create a frame to hold the table
        self.stats_frame = tk.Frame(self.home_frame, bg="white")
        self.stats_frame.pack(side="right", padx=200, pady=50, fill="y", expand=True)

        num_rows = 8  # Update this value based on your actual number of rows
        row_height = 30  # Update this value to match your row height
        header_height = 10  # Update this value to match your header height

        # Calculate the total height required for the Treeview
        total_height = (num_rows + 1) * row_height + header_height

        self.tree = ttk.Treeview(self.stats_frame, columns=("Processed Images", "Count of Images"), show="headings")
        self.tree.grid(row=0, column=0, sticky="nsew")

        # Set the calculated height for the Treeview
        self.tree.configure(height=num_rows)

        self.tree.heading("Processed Images", text="Processed Images")
        self.tree.heading("Count of Images", text="Count of Images")

        self.total_images_item = self.tree.insert("", "end", values=("Total Images Processed", self.total_images_processed))
        self.no_defect_item = self.tree.insert("", "end", values=("No Defect Images", self.no_defect_images))
        self.defected_item = self.tree.insert("", "end", values=("Defected Images", self.defected_images))
        #self.no_top_defect_item = self.tree.insert("", "end", values=("Top Defected Images", self.no_top_cam_defect_images))
        #self.no_shoulder_defect_item = self.tree.insert("", "end", values=("Shoulder Defected Images", self.no_shoulder_cam_defect_images))
        #self.no_neck_defect_item = self.tree.insert("", "end", values =  ("Neck defected Images", self.no_neck_cam_defect_images))
        #self.no_body_defect_item = self.tree.insert("", "end", values =  ("Body defected Images", self.no_body_cam_defect_images))
        #self.no_bottom_defect_item = self.tree.insert("", "end", values =  ("Bottom defected Images", self.no_bottom_cam_defect_images))


        # Configure row colors to mimic lines between rows
        self.tree.tag_configure("evenrow", background="#f0f0f0")
        
        # Set row height using a custom style
        self.tree_style = ttk.Style()
        self.tree_style.configure("Treeview",
            rowheight=row_height  # Use the calculated row height
        )

        # Apply the custom style to the tree
        self.tree.tag_configure("my_custom_style.Treeview", background="white")
        self.tree_style.map("my_custom_style.Treeview", background=[("selected", "#bfbfbf")])

        # Update grid weights to ensure proper resizing
        self.stats_frame.grid_rowconfigure(1, weight=10)
        self.stats_frame.grid_columnconfigure(1, weight=10)
        # Customize the appearance of the Treeview using ttk.Style
        self.tree_style = ttk.Style()
        self.tree_style.theme_create("my_custom_style", parent="default")
        self.tree_style.theme_use("my_custom_style")
        
        self.tree_style.configure("Treeview",
            background="white",
            fieldbackground="white",
            borderwidth=10,
            spacing=5,
            font=("Arial", 12),
            rowheight=30  # Increase the row height
        )
        self.tree_style.configure("Treeview.Heading",
            font=("Arial", 14, "bold")
        )
        # Configure row colors to mimic lines between rows
        self.tree.tag_configure("evenrow", background="#f0f0f0")
        for idx, item in enumerate(self.tree.get_children()):
            tag = "evenrow" if idx % 2 == 0 else ""
            self.tree.item(item, tags=(tag,))
    
    def update_image_stat_table(self):
        # Update the values in the table
        self.tree.item(self.total_images_item, values=("Total Images Processed", self.total_images_processed))
        self.tree.item(self.no_defect_item, values=("No Defect Images", self.no_defect_images))
        self.tree.item(self.defected_item, values=("Defected Images", self.defected_images))
        #for camera_name in self.camera_names:
        #    self.tree.item(f"{camera_name} Defected Images", values=(f"{camera_name} Defected Images", self.camera_defect_counts[camera_name]))

    def create_defect_table(self):
        self.defect_table_frame = tk.Frame(self.home_frame, bg="white")
        self.defect_tree = None


    def show_defect_table(self):
        selected_camera = self.camera_selection_var.get()
        if selected_camera == "Select Camera":
            return

        if self.defect_tree is None:
            self.defect_tree = ttk.Treeview(self.defect_table_frame, columns=("Defect Names", "Count"), show="headings")
            self.defect_tree.heading("Defect Names", text="Defect Names")
            self.defect_tree.heading("Count", text="Count")

        if selected_camera in self.defect_counts:
            defect_counts = self.defect_counts[selected_camera]
            self.update_defect_treeview(defect_counts)
        else:
            self.clear_defect_treeview()

         # Store the position of the image statistics table
        self.stats_frame_position = self.stats_frame.grid_info()

        # Hide the image statistics table frame
        self.stats_frame.pack_forget()

        # Show the defect table frame
        self.defect_tree.pack(side="top", padx=200, pady=50, fill="both", expand=True)

        # Show or update the close button
        if self.close_button is None:
            self.close_button = tk.Button(self.defect_table_frame, text="Close", font=("Arial", 16), command=self.close_defect_table)
            self.close_button.pack(padx=0, pady=50)
        else:
            self.close_button.pack(padx=0, pady=50)  # Show the existing close button

        # Show the defect table frame
        self.defect_table_frame.pack(fill="both", expand=True)

    def close_defect_table(self):
        # Clear the defect tree and hide the defects table
        self.clear_defect_treeview()
        # Remove the close button
        if self.close_button:
            self.close_button.pack_forget()

         # Hide the close button and defects table frame
        if self.close_button:
            self.close_button.pack_forget()
        self.defect_table_frame.pack_forget()

        # Restore the image statistics table position
        if self.stats_frame_position:
            self.stats_frame.grid(**self.stats_frame_position)
        else:
            self.stats_frame.pack(side="right", padx=10, pady=70, fill="both", expand=True)

    def update_defect_treeview(self, defect_counts):
        # Clear the current defect data table items
        for item in self.defect_tree.get_children():
            self.defect_tree.delete(item)

        # Update the defect data table with defect counts for the selected camera
        for defect, count in defect_counts.items():
            defect_item = self.defect_tree.insert("", "end", values=(defect, count))
            self.defect_tree.item(defect_item, values=(defect, count))

    def clear_defect_treeview(self):
        # Clear the current defect data table items
        for item in self.defect_tree.get_children():
            self.defect_tree.delete(item)


    def stop_acquisition(self):
        if self.is_acquiring:
            self.is_acquiring = False
            self.stop_button.config(state=tk.DISABLED)
            self.acquisition_button.config(state=tk.NORMAL)

    def start_acquisition(self):
        if not self.is_acquiring:
            self.is_acquiring = True
            self.stop_button.config(state=tk.NORMAL)
            self.acquisition_button.config(state=tk.DISABLED)
            acquisition_thread = threading.Thread(target=self.capture_images)
            acquisition_thread.start()

    def update_defect_counts_for_camera(self, event):
        selected_camera = self.camera_selection_var.get()
        if selected_camera in self.defect_counts:
            defect_counts = self.defect_counts[selected_camera]
            self.update_treeview_with_defect_counts(defect_counts)
        else:
            self.clear_treeview()

    def update_treeview_with_defect_counts(self, defect_counts):
        # Clear the current Treeview items
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Update the Treeview with defect counts for the selected camera
        for defect, count in defect_counts.items():
            defect_item = self.tree.insert("", "end", values=(defect, count))
            self.tree.item(defect_item, values=(defect, count))

    def clear_treeview(self):
        # Clear the current Treeview items
        for item in self.tree.get_children():
            self.tree.delete(item)

    def show_home_frame(self):
        self.home_frame.tkraise()


    def reset_app(self):
        if self.defect_tree:
            for item in self.defect_tree.get_children():
                self.defect_tree.delete(item)

        # Reset all values
        self.total_images_processed = 0
        self.defected_images = 0
        self.no_defect_images = 0
        self.defect_counts = {}
        self.update_image_stat_table()

        # Clear defect table
        self.clear_defect_treeview()

        # Clear the Excel sheet
        for camera_name in self.camera_names:
            excel_file = f"{camera_name}_defects.xlsx"
            if os.path.exists(excel_file):
                os.remove(excel_file)

        # Hide the close button and defects table frame
        if self.close_button:
            self.close_button.pack_forget()
        self.defect_table_frame.pack_forget()

        # Restore the image statistics table position
        if self.stats_frame_position:
            self.stats_frame.grid(**self.stats_frame_position)
        else:
            self.stats_frame.pack(side="right", padx=10, pady=70, fill="both", expand=True)

    def show_defect_graph(self):
        selected_camera = self.camera_selection_var.get()
        if selected_camera == "Select Camera":
            return
    
        if selected_camera in self.defect_counts:
            defect_counts = self.defect_counts[selected_camera]
            self.plot_defect_graph(selected_camera, defect_counts)
        else:
            print(f"No defect counts available for {selected_camera}")

    def plot_defect_graph(self, camera_name, defect_counts):
        defects = list(defect_counts.keys())
        counts = list(defect_counts.values())

        plt.figure(figsize=(10, 6))
        plt.bar(defects, counts)
        plt.xlabel("Defects")
        plt.ylabel("Count")
        plt.title(f"{camera_name} Defects")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.show()

    def update_defect_graph(self):
        selected_camera = self.camera_selection_var.get()
        if selected_camera == "Select Camera":
            return

        if selected_camera in self.defect_counts:
            defect_counts = self.defect_counts[selected_camera]
            self.plot_defect_graph(selected_camera, defect_counts)

               
    def show_home_frame(self):
        # Hide the defect table frame if it's currently shown
        if self.defect_table_frame.winfo_ismapped():
            self.close_defect_table()

        # Raise the home frame to the top
        self.home_frame.tkraise()

if __name__ == "__main__":
    app = App()
    app.mainloop()
