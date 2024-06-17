import os
import tkinter as tk
from tkinter import filedialog, messagebox
import PyPDF2

class PDFProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Processor")
        self.create_widgets()

    def create_widgets(self):
        self.load_button = tk.Button(self.root, text="Load PDFs", command=self.load_pdfs)
        self.load_button.pack(pady=10)

        self.process_button = tk.Button(self.root, text="Process PDFs", command=self.process_pdfs)
        self.process_button.pack(pady=10)

        self.save_button = tk.Button(self.root, text="Save as Text", command=self.save_as_text)
        self.save_button.pack(pady=10)

        self.status_label = tk.Label(self.root, text="", fg="green")
        self.status_label.pack(pady=10)

        self.pdf_files = []
        self.processed_text = ""

    def load_pdfs(self):
        self.pdf_files = filedialog.askopenfilenames(filetypes=[("PDF files", "*.pdf")])
        if self.pdf_files:
            self.status_label.config(text=f"{len(self.pdf_files)} PDFs loaded.")
        else:
            self.status_label.config(text="No PDFs loaded.")

    def process_pdfs(self):
        self.processed_text = ""
        for pdf_file in self.pdf_files:
            with open(pdf_file, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = len(reader.pages)
                for page_num in range(15, num_pages):  # Skip the first 15 pages
                    page = reader.pages[page_num]
                    self.processed_text += page.extract_text()
        self.status_label.config(text="PDFs processed successfully.")

    def save_as_text(self):
        if self.processed_text:
            save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
            if save_path:
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(self.processed_text)
                messagebox.showinfo("Success", "Text file saved successfully.")
        else:
            messagebox.showwarning("Warning", "No text to save. Please process PDFs first.")

if __name__ == "__main__":
    root = tk.Tk()
    app = PDFProcessorApp(root)
    root.mainloop()