#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for the Grape Labeling Tool.
This application provides a GUI for manually labeling grape images and exporting
annotations in YOLO format.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import numpy as np

# Import the main application class
from src.app import AplicacionPuntosRecortes

def main():
    """Main execution function that initializes and runs the Tkinter application."""
    # Create the root Tkinter window
    root = tk.Tk()
    
    # Create an instance of the labeling application
    app = AplicacionPuntosRecortes(root)
    
    # Start the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
