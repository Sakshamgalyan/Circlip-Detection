import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QTextEdit, QGroupBox, 
                            QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget, 
                            QStatusBar, QFileDialog)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QDate
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtMultimedia import QSound
import cv2
from ultralytics import YOLO
import torch
import mysql.connector  
from mysql.connector import errorcode
import rk_mcprotocol as mc
import time
import os
from io import BytesIO
from PIL import Image
import requests
import numpy as np
from datetime import datetime, date
import winsound
import csv
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Disable OpenMP issues for PyTorch
torch.backends.openmp.enabled = False
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
RTSP_URL = "rtsp://admin:Kubota@2025@192.168.180.187:554/Streaming/Channels/101"
CAMERA_IND = 0
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Saksham@1",
    "database": "Circlip",
    "autocommit": True
}
PLC_HOST = '192.168.3.250'
PLC_PORT = 1025
LOGO_URL = "https://upload.wikimedia.org/wikipedia/commons/7/7f/Escorts_Kubota_Limited.jpg"
TITLE_LOGO_URL = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSxODT3mCalzwuNjjG27OI9ya_uPfebLhL7Sg&s"
ALERT_SOUND_FILE = "alert.wav"

class DetectionThread(QThread):
    update_signal = pyqtSignal(str, float, float, float, str, QImage)
    error_signal = pyqtSignal(str)
    log_signal = pyqtSignal(str, str)
    frame_signal = pyqtSignal(QImage)
    alert_signal = pyqtSignal(str, str)
    
    def __init__(self, plc_socket, db_connection):
        super().__init__()
        self.plc_socket = plc_socket
        self.db_connection = db_connection
        self.running = True
        self.frame_count = 0
        self.processing_times = []
        
    def run(self):
        try:
            if not self.db_connection.is_connected():
                self.db_connection.reconnect()
                
            self.log_signal.emit("Loading YOLO model...", "info")
            model = YOLO("yolov8training/exp1/weights/best.pt")
            # model.to("cuda" if torch.cuda.is_available() else "cpu")
            model.to('cpu')

            self.log_signal.emit("Starting detection...", "info")
            cap = cv2.VideoCapture(CAMERA_IND)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                self.error_signal.emit("Could not open camera feed")
                return

            frame_count = single_circlip_frames = multiple_circlips_frames = no_circlip_frames = 0
            start_time = time.time()

            while self.running and time.time() - start_time < 2:              #Detection time
                ret, frame = cap.read()
                if not ret:
                    self.error_signal.emit("Failed to read frame")
                    break

                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                qt_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
                self.frame_signal.emit(qt_image)

                detection_start = time.time()
                results = model.predict(frame, device='cuda', dnn=True)
                self.processing_times.append(time.time() - detection_start)

                for result in results:
                    num_circlips = len(result.boxes)
                    if num_circlips == 1:
                        single_circlip_frames += 1
                    elif num_circlips > 1:
                        multiple_circlips_frames += 1
                        self.alert_signal.emit(f"Multiple circlips detected: {num_circlips}", "warning")
                    else:
                        no_circlip_frames += 1

                frame_count += 1
                # QThread.msleep(30)            # for smooth video

            cap.release()

            if frame_count == 0:
                self.error_signal.emit("No frames processed")
                return

            single_percent = (single_circlip_frames / frame_count) * 100
            multiple_percent = (multiple_circlips_frames / frame_count) * 100
            none_percent = (no_circlip_frames / frame_count) * 100
            result = "YES" if single_percent >= 60 else "NO"

            if result == "NO":
                self.play_error_sound()
                self.log_signal.emit("Circlip missing/incorrect", "error")

            self.store_result(single_percent, multiple_percent, none_percent, result)
            self.send_to_plc(result)

            self.update_signal.emit(
                "Detection complete", 
                single_percent, 
                multiple_percent, 
                none_percent, 
                result,
                qt_image
            )

        except Exception as e:
            self.error_signal.emit(f"Error: {str(e)}")
            self.play_error_sound()

    def store_result(self, single, multiple, none, result):
        try:
            if not self.db_connection.is_connected():
                self.db_connection.reconnect()
                
            cursor = self.db_connection.cursor()
            query = """
                INSERT INTO detection_results (
                    single_circlip_percentage, 
                    multiple_circlips_percentage, 
                    no_circlip_percentage, 
                    result
                ) VALUES (%s, %s, %s, %s)
            """
            values = (single, multiple, none, result)
            cursor.execute(query, values)
            self.db_connection.commit()
            self.log_signal.emit(f"Result stored in database: {result}", "info")
        except mysql.connector.Error as e:
            self.log_signal.emit(f"Database error: {e}", "error")
            try:
                self.db_connection.reconnect()
                cursor = self.db_connection.cursor()
                cursor.execute(query, values)
                self.db_connection.commit()
                self.log_signal.emit("Retry successful - result stored", "info")
            except Exception as retry_error:
                self.log_signal.emit(f"Failed to store after retry: {retry_error}", "error")
        finally:
            if 'cursor' in locals():
                cursor.close()

    def send_to_plc(self, result):
        if self.plc_socket:
            try:
                value = 1 if result == "YES" else 0
                write_status = mc.write_sign_word(self.plc_socket, 'D1', [value], False)
                if write_status:
                    self.log_signal.emit(f"Sent to PLC D1: {value}", "info")
                else:
                    self.log_signal.emit("Failed to write to PLC D1", "error")
            except Exception as e:
                self.log_signal.emit(f"PLC error: {e}", "error")

    def play_error_sound(self):
        try:
            winsound.Beep(1000, 1000)
        except:
            pass

class DetectionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Circlip Detection System")
        self.setMinimumSize(1200, 800)
        
        # Initialize detection_thread as None
        self.detection_thread = None
        
        # Set window icon (logo in title bar)
        self.load_window_icon()
        
        # Initialize UI first
        self.init_ui()
        self.load_logo()
        self.add_alert_system()
        
        # Now initialize connections
        self.db_connection = self.connect_database()
        self.create_table()
        self.plc_socket = self.connect_plc()
        
        # Initial system status
        self.log_message("System initialized", "info")
        if self.db_connection and self.db_connection.is_connected():
            self.load_today_summary()

    def load_window_icon(self):
        """Load company logo as window icon"""
        try:
            response = requests.get(TITLE_LOGO_URL, timeout=5)
            img = Image.open(BytesIO(response.content))
            
            # Convert to bytes
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format='PNG')
            
            # Create QIcon from bytes
            qimg = QImage()
            qimg.loadFromData(img_byte_arr.getvalue())
            pixmap = QPixmap.fromImage(qimg)
            self.setWindowIcon(QIcon(pixmap))
        except Exception as e:
            self.log_message(f"Could not load window icon: {e}", "warning")

    def init_ui(self):
        """Initialize all UI components"""
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Header with logo and title
        header = QHBoxLayout()
        self.logo_label = QLabel()
        self.logo_label.setFixedSize(120, 60)
        header.addWidget(self.logo_label)
        
        title = QLabel("Circlip Detection System")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        header.addWidget(title, 1, Qt.AlignCenter)
        
        header.addSpacing(120)
        main_layout.addLayout(header)
        
        # Main tab interface
        self.tab_widget = QTabWidget()
        
        # Detection tab
        detection_tab = QWidget()
        detection_layout = QVBoxLayout()
        
        # Camera and controls
        content_layout = QHBoxLayout()
        
        # Left panel - Camera
        left_panel = QVBoxLayout()
        camera_group = QGroupBox("Camera Feed")
        self.camera_label = QLabel()
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black; min-height: 400px;")
        camera_group.setLayout(QVBoxLayout())
        camera_group.layout().addWidget(self.camera_label)
        left_panel.addWidget(camera_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        self.detect_button = QPushButton("Start Detection")
        self.detect_button.setStyleSheet("""
            QPushButton {
                font-size: 16px; 
                padding: 10px;
                background-color: #3498db;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.detect_button.clicked.connect(self.start_detection)
        button_layout.addWidget(self.detect_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("""
            QPushButton {
                font-size: 16px; 
                padding: 10px;
                background-color: #e74c3c;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_detection)
        button_layout.addWidget(self.stop_button)
        
        self.try_again_button = QPushButton("Try Again")
        self.try_again_button.setStyleSheet("""
            QPushButton {
                font-size: 16px; 
                padding: 10px;
                background-color: #f39c12;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #d35400;
            }
            QPushButton:disabled {
                background-color: #95a5a6;
            }
        """)
        self.try_again_button.setEnabled(False)
        self.try_again_button.clicked.connect(self.try_again_detection)
        button_layout.addWidget(self.try_again_button)
        
        left_panel.addLayout(button_layout)
        content_layout.addLayout(left_panel, 2)
        
        # Right panel - Results
        self.right_panel = QVBoxLayout()
        
        # Connection status
        connection_group = QGroupBox("Connection Status")
        connection_layout = QVBoxLayout()
        
        # Database status
        self.db_status_label = QLabel("Database: Not connected")
        self.db_reconnect_button = QPushButton("Reconnect")
        self.db_reconnect_button.setStyleSheet("""
            QPushButton {
                padding: 5px;
                background-color: #3498db;
                color: white;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.db_reconnect_button.clicked.connect(self.reconnect_database)
        
        db_layout = QHBoxLayout()
        db_layout.addWidget(self.db_status_label)
        db_layout.addWidget(self.db_reconnect_button)
        connection_layout.addLayout(db_layout)
        
        # PLC status
        self.plc_status_label = QLabel("PLC: Not connected")
        self.plc_reconnect_button = QPushButton("Reconnect")
        self.plc_reconnect_button.setStyleSheet("""
            QPushButton {
                padding: 5px;
                background-color: #3498db;
                color: white;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)
        self.plc_reconnect_button.clicked.connect(self.reconnect_plc)
        
        plc_layout = QHBoxLayout()
        plc_layout.addWidget(self.plc_status_label)
        plc_layout.addWidget(self.plc_reconnect_button)
        connection_layout.addLayout(plc_layout)
        
        connection_group.setLayout(connection_layout)
        self.right_panel.addWidget(connection_group)
        
        # Results display
        results_group = QGroupBox("Detection Results")
        results_layout = QVBoxLayout()
        self.result_label = QLabel("Status: Ready")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        results_layout.addWidget(self.result_label)
        
        self.timestamp_label = QLabel("Last update: Never")
        results_layout.addWidget(self.timestamp_label)
        
        self.fps_label = QLabel("FPS: --")
        results_layout.addWidget(self.fps_label)
        
        self.processing_time_label = QLabel("Processing Time: -- ms")
        results_layout.addWidget(self.processing_time_label)
        
        results_group.setLayout(results_layout)
        self.right_panel.addWidget(results_group)
        
        # System log
        log_group = QGroupBox("System Log")
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet("font-family: Consolas; font-size: 12px;")
        log_group.setLayout(QVBoxLayout())
        log_group.layout().addWidget(self.log_display)
        self.right_panel.addWidget(log_group, 1)
        
        content_layout.addLayout(self.right_panel, 1)
        detection_layout.addLayout(content_layout)
        detection_tab.setLayout(detection_layout)
        
        # Today's summary tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout()
        
        self.date_label = QLabel(f"Date: {QDate.currentDate().toString('dddd, MMMM d, yyyy')}")
        summary_layout.addWidget(self.date_label)
        
        # Statistics table
        stats_group = QGroupBox("Today's Statistics")
        self.stats_table = QTableWidget()
        self.stats_table.setColumnCount(4)
        self.stats_table.setHorizontalHeaderLabels(["Time", "Result", "Single %", "Multiple %"])
        self.stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        stats_group.setLayout(QVBoxLayout())
        stats_group.layout().addWidget(self.stats_table)
        summary_layout.addWidget(stats_group)
        
        # Summary metrics
        metrics_layout = QHBoxLayout()
        
        total_group = QGroupBox("Total")
        self.total_label = QLabel("0")
        total_group.setLayout(QVBoxLayout())
        total_group.layout().addWidget(self.total_label)
        metrics_layout.addWidget(total_group)
        
        passed_group = QGroupBox("Passed")
        self.passed_label = QLabel("0")
        passed_group.setLayout(QVBoxLayout())
        passed_group.layout().addWidget(self.passed_label)
        metrics_layout.addWidget(passed_group)
        
        failed_group = QGroupBox("Failed")
        self.failed_label = QLabel("0")
        failed_group.setLayout(QVBoxLayout())
        failed_group.layout().addWidget(self.failed_label)
        metrics_layout.addWidget(failed_group)
        
        summary_layout.addLayout(metrics_layout)
        
        # Export buttons
        export_layout = QHBoxLayout()
        self.export_csv_button = QPushButton("Export to CSV")
        self.export_csv_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px;
                background-color: #27ae60;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #219653;
            }
        """)
        self.export_csv_button.clicked.connect(self.export_to_csv)
        export_layout.addWidget(self.export_csv_button)
        
        self.export_pdf_button = QPushButton("Export to PDF")
        self.export_pdf_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px;
                background-color: #e74c3c;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.export_pdf_button.clicked.connect(self.export_to_pdf)
        export_layout.addWidget(self.export_pdf_button)
        
        summary_layout.addLayout(export_layout)
        
        # Refresh button
        refresh_button = QPushButton("Refresh Statistics")
        refresh_button.clicked.connect(self.load_today_summary)
        summary_layout.addWidget(refresh_button)
        
        summary_tab.setLayout(summary_layout)
        
        # History tab
        history_tab = QWidget()
        history_layout = QVBoxLayout()
        
        # Date selection
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Select Date:"))
        
        self.date_edit = QTextEdit()
        self.date_edit.setMaximumHeight(30)
        self.date_edit.setPlaceholderText("YYYY-MM-DD")
        date_layout.addWidget(self.date_edit)
        
        search_button = QPushButton("Search")
        search_button.clicked.connect(self.load_history)
        date_layout.addWidget(search_button)
        
        history_layout.addLayout(date_layout)
        
        # History table
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["Timestamp", "Result", "Single %", "Multiple %", "No Circlip %"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.history_table)
        
        # Export buttons for history
        history_export_layout = QHBoxLayout()
        self.history_export_csv_button = QPushButton("Export to CSV")
        self.history_export_csv_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px;
                background-color: #27ae60;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #219653;
            }
        """)
        self.history_export_csv_button.clicked.connect(self.export_history_to_csv)
        history_export_layout.addWidget(self.history_export_csv_button)
        
        self.history_export_pdf_button = QPushButton("Export to PDF")
        self.history_export_pdf_button.setStyleSheet("""
            QPushButton {
                font-size: 14px;
                padding: 8px;
                background-color: #e74c3c;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        self.history_export_pdf_button.clicked.connect(self.export_history_to_pdf)
        history_export_layout.addWidget(self.history_export_pdf_button)
        
        history_layout.addLayout(history_export_layout)
        
        history_tab.setLayout(history_layout)
        
        # Add tabs
        self.tab_widget.addTab(detection_tab, "Detection")
        self.tab_widget.addTab(summary_tab, "Today's Summary")
        self.tab_widget.addTab(history_tab, "History")
        
        main_layout.addWidget(self.tab_widget)
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # Add status bar with developer name
        self.statusBar().showMessage("Developed by SAKSHAM & SPARSH")
        self.statusBar().setStyleSheet("""
            QStatusBar {
                font-size: 12px;
                color: #555;
                border-top: 1px solid #ddd;
                padding: 3px;
            }
        """)

    def start_detection(self):
        """Start detection process"""
        if not self.db_connection or not self.db_connection.is_connected():
            self.db_connection = self.connect_database()
            if not self.db_connection:
                self.log_message("Cannot start detection - no database connection", "error")
                return
                
        self.log_display.clear()
        self.detect_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.try_again_button.setEnabled(False)
        self.result_label.setText("Status: Detecting...")
        
        self.detection_thread = DetectionThread(self.plc_socket, self.db_connection)
        self.detection_thread.update_signal.connect(self.update_display)
        self.detection_thread.error_signal.connect(self.handle_error)
        self.detection_thread.log_signal.connect(self.log_message)
        self.detection_thread.frame_signal.connect(self.update_frame)
        self.detection_thread.alert_signal.connect(self.trigger_alert)
        self.detection_thread.start()

    def stop_detection(self):
        """Stop detection process"""
        if hasattr(self, 'detection_thread') and self.detection_thread is not None:
            self.detection_thread.running = False
            self.detection_thread.wait()
            self.detection_thread = None
            
        self.detect_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_message("Detection stopped", "info")

    def try_again_detection(self):
        """Delete last record and start new detection"""
        if not self.db_connection or not self.db_connection.is_connected():
            self.db_connection = self.connect_database()
            if not self.db_connection:
                self.log_message("Cannot try again - no database connection", "error")
                return
    
        try:
            cursor = self.db_connection.cursor()
            # Get the ID of the last record
            cursor.execute("SELECT id FROM detection_results ORDER BY id DESC LIMIT 1")
            last_id = cursor.fetchone()
            
            if last_id:
                # Delete the last record
                cursor.execute("DELETE FROM detection_results WHERE id = %s", (last_id[0],))
                self.db_connection.commit()
                self.log_message(f"Deleted last record (ID: {last_id[0]})", "info")
                
                # Update the summary display
                self.load_today_summary()
                
                # Start new detection immediately
                self.start_detection()
                
        except mysql.connector.Error as e:
            self.log_message(f"Database error during try again: {e}", "error")
        finally:
            if 'cursor' in locals():
                cursor.close()

    def update_frame(self, image):
        """Update camera display"""
        pixmap = QPixmap.fromImage(image)
        self.camera_label.setPixmap(pixmap.scaled(
            self.camera_label.width(), 
            self.camera_label.height(), 
            Qt.KeepAspectRatio
        ))
        
        # Update FPS
        current_time = time.time()
        if hasattr(self, 'last_frame_time'):
            fps = 1 / (current_time - self.last_frame_time)
            self.fps_label.setText(f"FPS: {fps:.1f}")
        self.last_frame_time = current_time

    def update_display(self, status, single, multiple, none, result, image):
        """Update UI with detection results"""
        if result == "YES":
            self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: green;")
            self.result_label.setText("Status: Circlip detected")
            self.try_again_button.setEnabled(False)
        else:
            self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: red;")
            self.result_label.setText("Status: Circlip missing")
            self.try_again_button.setEnabled(True)
            
        self.timestamp_label.setText(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if hasattr(self.detection_thread, 'processing_times') and self.detection_thread.processing_times:
            avg_time = np.mean(self.detection_thread.processing_times) * 1000
            self.processing_time_label.setText(f"Processing Time: {avg_time:.1f} ms")
        
        self.detect_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.load_today_summary()

    def handle_error(self, error_message):
        """Handle detection errors"""
        self.log_message(error_message, "error")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: bold; color: red;")
        self.result_label.setText("Status: Error occurred")
        self.detect_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.try_again_button.setEnabled(False)

    def trigger_alert(self, message, level="warning"):
        """Display visual alert - modified to suppress 'No circlip' alerts"""
        # Skip showing alerts for "No circlip detected" messages
        if "No circlip detected" in message:
            return
            
        color = {
            "error": "background-color: #ff4444; color: white;",
            "warning": "background-color: #ffbb33; color: black;",
            "info": "background-color: #99cc00; color: white;"
        }.get(level, "")
        
        self.alert_panel.setText(message)
        self.alert_panel.setStyleSheet(f"font-size: 24px; font-weight: bold; padding: 20px; border-radius: 10px; margin: 10px; {color}")
        self.alert_panel.show()
        
        if level in ["error", "warning"] and self.alert_sound:
            self.alert_sound.play()
            
        QTimer.singleShot(5000, self.alert_panel.hide)

    def export_to_csv(self):
        """Export today's results to CSV file"""
        if not self.db_connection or not self.db_connection.is_connected():
            self.log_message("Database not connected", "error")
            return
        
        try:
            today = date.today().strftime('%Y-%m-%d')
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT timestamp, result, 
                       single_circlip_percentage as single, 
                       multiple_circlips_percentage as multiple,
                       no_circlip_percentage as none
                FROM detection_results 
                WHERE DATE(timestamp) = %s
                ORDER BY timestamp DESC
            """, (today,))
            
            detections = cursor.fetchall()
            cursor.close()
            
            if not detections:
                self.log_message("No data to export", "warning")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save CSV File",
                f"circlip_results_{today}.csv",
                "CSV Files (*.csv)"
            )
            
            if not file_path:
                return
            
            with open(file_path, mode='w', newline='') as csv_file:
                fieldnames = ['timestamp', 'result', 'single', 'multiple', 'none']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(detections)
            
            self.log_message(f"Results exported to {file_path}", "info")
            
        except Exception as e:
            self.log_message(f"CSV export failed: {e}", "error")

    def export_to_pdf(self):
        """Export today's results to PDF report"""
        if not self.db_connection or not self.db_connection.is_connected():
            self.log_message("Database not connected", "error")
            return
        
        try:
            today = date.today().strftime('%Y-%m-%d')
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT timestamp, result, 
                       single_circlip_percentage as single, 
                       multiple_circlips_percentage as multiple,
                       no_circlip_percentage as none
                FROM detection_results 
                WHERE DATE(timestamp) = %s
                ORDER BY timestamp DESC
            """, (today,))
            
            detections = cursor.fetchall()
            cursor.close()
            
            if not detections:
                self.log_message("No data to export", "warning")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save PDF Report",
                f"circlip_report_{today}.pdf",
                "PDF Files (*.pdf)"
            )
            
            if not file_path:
                return
            
            # Create PDF document
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            elements = []
            
            # Add title
            styles = getSampleStyleSheet()
            elements.append(Paragraph("Circlip Detection Report", styles['Title']))
            elements.append(Paragraph(f"Date: {today}", styles['Normal']))
            elements.append(Paragraph(" ", styles['Normal']))  # Spacer
            
            # Prepare data for table
            data = [['Timestamp', 'Result', 'Single %', 'Multiple %', 'No Circlip %']]
            for detection in detections:
                data.append([
                    str(detection['timestamp']),
                    detection['result'],
                    f"{detection['single']:.1f}",
                    f"{detection['multiple']:.1f}",
                    f"{detection['none']:.1f}"
                ])
            
            # Create table
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(table)
            
            # Summary statistics
            passed = sum(1 for d in detections if d['result'] == "YES")
            failed = len(detections) - passed
            elements.append(Paragraph(" ", styles['Normal']))  # Spacer
            elements.append(Paragraph(f"Total Detections: {len(detections)}", styles['Normal']))
            elements.append(Paragraph(f"Passed: {passed}", styles['Normal']))
            elements.append(Paragraph(f"Failed: {failed}", styles['Normal']))
            
            # Build PDF
            doc.build(elements)
            self.log_message(f"PDF report saved to {file_path}", "info")
            
        except Exception as e:
            self.log_message(f"PDF export failed: {e}", "error")

    def export_history_to_csv(self):
        """Export historical data to CSV file"""
        if not self.db_connection or not self.db_connection.is_connected():
            self.log_message("Database not connected", "error")
            return
        
        try:
            selected_date = self.date_edit.toPlainText().strip()
            if not selected_date:
                selected_date = date.today().strftime('%Y-%m-%d')
                
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT timestamp, result, 
                       single_circlip_percentage as single, 
                       multiple_circlips_percentage as multiple,
                       no_circlip_percentage as none
                FROM detection_results 
                WHERE DATE(timestamp) = %s
                ORDER BY timestamp DESC
            """, (selected_date,))
            
            detections = cursor.fetchall()
            cursor.close()
            
            if not detections:
                self.log_message("No data to export", "warning")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save CSV File",
                f"circlip_history_{selected_date}.csv",
                "CSV Files (*.csv)"
            )
            
            if not file_path:
                return
            
            with open(file_path, mode='w', newline='') as csv_file:
                fieldnames = ['timestamp', 'result', 'single', 'multiple', 'none']
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(detections)
            
            self.log_message(f"History exported to {file_path}", "info")
            
        except Exception as e:
            self.log_message(f"CSV export failed: {e}", "error")

    def export_history_to_pdf(self):
        """Export historical data to PDF report"""
        if not self.db_connection or not self.db_connection.is_connected():
            self.log_message("Database not connected", "error")
            return
        
        try:
            selected_date = self.date_edit.toPlainText().strip()
            if not selected_date:
                selected_date = date.today().strftime('%Y-%m-%d')
                
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT timestamp, result, 
                       single_circlip_percentage as single, 
                       multiple_circlips_percentage as multiple,
                       no_circlip_percentage as none
                FROM detection_results 
                WHERE DATE(timestamp) = %s
                ORDER BY timestamp DESC
            """, (selected_date,))
            
            detections = cursor.fetchall()
            cursor.close()
            
            if not detections:
                self.log_message("No data to export", "warning")
                return
            
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save PDF Report",
                f"circlip_history_{selected_date}.pdf",
                "PDF Files (*.pdf)"
            )
            
            if not file_path:
                return
            
            # Create PDF document
            doc = SimpleDocTemplate(file_path, pagesize=letter)
            elements = []
            
            # Add title
            styles = getSampleStyleSheet()
            elements.append(Paragraph("Circlip Detection History Report", styles['Title']))
            elements.append(Paragraph(f"Date: {selected_date}", styles['Normal']))
            elements.append(Paragraph(" ", styles['Normal']))  # Spacer
            
            # Prepare data for table
            data = [['Timestamp', 'Result', 'Single %', 'Multiple %', 'No Circlip %']]
            for detection in detections:
                data.append([
                    str(detection['timestamp']),
                    detection['result'],
                    f"{detection['single']:.1f}",
                    f"{detection['multiple']:.1f}",
                    f"{detection['none']:.1f}"
                ])
            
            # Create table
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(table)
            
            # Summary statistics
            passed = sum(1 for d in detections if d['result'] == "YES")
            failed = len(detections) - passed
            elements.append(Paragraph(" ", styles['Normal']))  # Spacer
            elements.append(Paragraph(f"Total Detections: {len(detections)}", styles['Normal']))
            elements.append(Paragraph(f"Passed: {passed}", styles['Normal']))
            elements.append(Paragraph(f"Failed: {failed}", styles['Normal']))
            
            # Build PDF
            doc.build(elements)
            self.log_message(f"PDF history report saved to {file_path}", "info")
            
        except Exception as e:
            self.log_message(f"PDF export failed: {e}", "error")

    def load_today_summary(self):
        """Load today's detection summary"""
        if not self.db_connection or not self.db_connection.is_connected():
            return
            
        try:
            today = date.today().strftime('%Y-%m-%d')
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT TIME(timestamp) as time, result, 
                       single_circlip_percentage as single, 
                       multiple_circlips_percentage as multiple
                FROM detection_results 
                WHERE DATE(timestamp) = %s
                ORDER BY timestamp DESC
            """, (today,))
            
            detections = cursor.fetchall()
            cursor.close()
            
            self.stats_table.setRowCount(len(detections))
            for row, detection in enumerate(detections):
                self.stats_table.setItem(row, 0, QTableWidgetItem(str(detection['time'])))
                self.stats_table.setItem(row, 1, QTableWidgetItem(detection['result']))
                self.stats_table.setItem(row, 2, QTableWidgetItem(f"{detection['single']:.1f}%"))
                self.stats_table.setItem(row, 3, QTableWidgetItem(f"{detection['multiple']:.1f}%"))
                
                # Color coding
                color = Qt.green if detection['result'] == "YES" else Qt.red
                for col in range(4):
                    self.stats_table.item(row, col).setBackground(color)
            
            # Update summary counts
            self.total_label.setText(str(len(detections)))
            passed = sum(1 for d in detections if d['result'] == "YES")
            self.passed_label.setText(str(passed))
            self.failed_label.setText(str(len(detections) - passed))
            
        except mysql.connector.Error as e:
            self.log_message(f"Database error loading summary: {e}", "error")

    def load_history(self):
        """Load historical detection data"""
        if not self.db_connection or not self.db_connection.is_connected():
            return
            
        try:
            selected_date = self.date_edit.toPlainText().strip()
            if not selected_date:
                selected_date = date.today().strftime('%Y-%m-%d')
                
            cursor = self.db_connection.cursor(dictionary=True)
            cursor.execute("""
                SELECT timestamp, result, 
                       single_circlip_percentage as single, 
                       multiple_circlips_percentage as multiple,
                       no_circlip_percentage as none
                FROM detection_results 
                WHERE DATE(timestamp) = %s
                ORDER BY timestamp DESC
            """, (selected_date,))
            
            detections = cursor.fetchall()
            cursor.close()
            
            self.history_table.setRowCount(len(detections))
            for row, detection in enumerate(detections):
                self.history_table.setItem(row, 0, QTableWidgetItem(str(detection['timestamp'])))
                self.history_table.setItem(row, 1, QTableWidgetItem(detection['result']))
                self.history_table.setItem(row, 2, QTableWidgetItem(f"{detection['single']:.1f}%"))
                self.history_table.setItem(row, 3, QTableWidgetItem(f"{detection['multiple']:.1f}%"))
                self.history_table.setItem(row, 4, QTableWidgetItem(f"{detection['none']:.1f}%"))
                
                # Color coding
                color = Qt.green if detection['result'] == "YES" else Qt.red
                for col in range(5):
                    self.history_table.item(row, col).setBackground(color)
                    
        except mysql.connector.Error as e:
            self.log_message(f"Database error loading history: {e}", "error")
        except ValueError:
            self.log_message("Invalid date format. Use YYYY-MM-DD", "warning")

    def connect_database(self):
        """Establish database connection with proper error handling"""
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            if conn.is_connected():
                self.db_status_label.setText("Database: Connected")
                self.db_status_label.setStyleSheet("color: green;")
                self.log_message("Database connected successfully", "info")
                return conn
        except mysql.connector.Error as e:
            if e.errno == errorcode.ER_BAD_DB_ERROR:
                try:
                    # Create database if it doesn't exist
                    temp_conn = mysql.connector.connect(
                        host=DB_CONFIG["host"],
                        user=DB_CONFIG["user"],
                        password=DB_CONFIG["password"]
                    )
                    cursor = temp_conn.cursor()
                    cursor.execute(f"CREATE DATABASE {DB_CONFIG['database']}")
                    temp_conn.close()
                    # Retry connection
                    return self.connect_database()
                except mysql.connector.Error as e:
                    self.db_status_label.setText("Database: Not connected")
                    self.db_status_label.setStyleSheet("color: red;")
                    self.log_message(f"Failed to create database: {e}", "error")
            else:
                self.db_status_label.setText("Database: Not connected")
                self.db_status_label.setStyleSheet("color: red;")
                self.log_message(f"Database connection failed: {e}", "error")
        return None

    def reconnect_database(self):
        """Reconnect to database"""
        self.log_message("Attempting to reconnect to database...", "info")
        if hasattr(self, 'db_connection') and self.db_connection and self.db_connection.is_connected():
            self.db_connection.close()
            
        self.db_connection = self.connect_database()
        if self.db_connection and self.db_connection.is_connected():
            self.load_today_summary()

    def create_table(self):
        """Create the detection_results table if it doesn't exist"""
        if self.db_connection and self.db_connection.is_connected():
            try:
                cursor = self.db_connection.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS detection_results (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        single_circlip_percentage FLOAT NOT NULL,
                        multiple_circlips_percentage FLOAT NOT NULL,
                        no_circlip_percentage FLOAT NOT NULL,
                        result VARCHAR(10) NOT NULL
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """)
                self.db_connection.commit()
                self.log_message("Database table verified/created", "info")
            except mysql.connector.Error as e:
                self.log_message(f"Error creating table: {e}", "error")
            finally:
                cursor.close()

    def connect_plc(self):
        """Establish connection to PLC"""
        try:
            s = mc.open_socket(PLC_HOST, PLC_PORT)
            if s:
                self.plc_status_label.setText("PLC: Connected")
                self.plc_status_label.setStyleSheet("color: green;")
                self.log_message("PLC connected successfully", "info")
                return s
            self.log_message("PLC connection failed", "error")
        except Exception as e:
            self.log_message(f"PLC connection error: {e}", "error")
        self.plc_status_label.setText("PLC: Not connected")
        self.plc_status_label.setStyleSheet("color: red;")
        return None

    def reconnect_plc(self):
        """Reconnect to PLC"""
        self.log_message("Attempting to reconnect to PLC...", "info")
        if hasattr(self, 'plc_socket') and self.plc_socket:
            self.plc_socket.close()
            
        self.plc_socket = self.connect_plc()

    def load_logo(self):
        """Load company logo"""
        try:
            response = requests.get(LOGO_URL, timeout=5)
            img = Image.open(BytesIO(response.content))
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, 'JPEG')
            qimg = QImage()
            qimg.loadFromData(img_byte_arr.getvalue())
            pixmap = QPixmap.fromImage(qimg)
            self.logo_label.setPixmap(pixmap.scaled(120, 60, Qt.KeepAspectRatio))
        except Exception as e:
            self.log_message(f"Could not load logo: {e}", "warning")
            self.logo_label.setText("Logo")

    def add_alert_system(self):
        """Initialize alert system"""
        if os.path.exists(ALERT_SOUND_FILE):
            try:
                self.alert_sound = QSound(ALERT_SOUND_FILE)
            except Exception as e:
                self.alert_sound = None
                self.log_message(f"Alert sound error: {e}", "warning")
        else:
            self.alert_sound = None
            
        self.alert_panel = QLabel()
        self.alert_panel.setAlignment(Qt.AlignCenter)
        self.alert_panel.hide()
        self.right_panel.insertWidget(2, self.alert_panel)

    def log_message(self, message, level="info"):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        colors = {
            "error": "red",
            "warning": "orange",
            "info": "black"
        }
        self.log_display.append(f'<font color="{colors.get(level, "black")}">[{timestamp}] {message}</font>')
        self.log_display.verticalScrollBar().setValue(self.log_display.verticalScrollBar().maximum())

    def closeEvent(self, event):
        """Clean up resources when closing"""
        self.stop_detection()
        
        if hasattr(self, 'db_connection') and self.db_connection and self.db_connection.is_connected():
            self.db_connection.close()
            self.log_message("Database connection closed", "info")
            
        if hasattr(self, 'plc_socket') and self.plc_socket:
            self.plc_socket.close()
            self.log_message("PLC connection closed", "info")
            
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = DetectionUI()
    window.show()
    sys.exit(app.exec_())