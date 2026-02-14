

import tkinter as tk
from tkinter import ttk, messagebox

import sys
import os

# Allow importing from src/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from inference_engine import CreditRiskEngine


class FinancialRiskApp:

    def __init__(self, root):
        self.root = root
        self.root.title("Financial Risk Dashboard")
        self.root.geometry("900x500")
        self.root.configure(bg="#1e1e2f")

        self.engine = CreditRiskEngine()

        self._build_styles()
        self._build_layout()

    # ---------------------------
    # Styling
    # ---------------------------

    def _build_styles(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(
            "TLabel",
            background="#1e1e2f",
            foreground="white",
            font=("Segoe UI", 11)
        )

        style.configure(
            "Header.TLabel",
            font=("Segoe UI", 18, "bold"),
            foreground="#4cc9f0"
        )

        style.configure(
            "TButton",
            font=("Segoe UI", 11, "bold"),
            padding=6
        )

        style.configure(
            "Result.TLabel",
            font=("Segoe UI", 12, "bold"),
            foreground="#f1fa8c"
        )

    # ---------------------------
    # Layout
    # ---------------------------

    def _build_layout(self):

        header = ttk.Label(
            self.root,
            text="Financial Risk Intelligence Dashboard",
            style="Header.TLabel"
        )
        header.pack(pady=15)

        main_frame = tk.Frame(self.root, bg="#1e1e2f")
        main_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Left Panel (Inputs)
        input_frame = tk.Frame(main_frame, bg="#2a2a40")
        input_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        tk.Label(
            input_frame,
            text="Applicant Information",
            bg="#2a2a40",
            fg="#ffffff",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=10)

        self.entries = {}
        fields = [
            ("LIMIT_BAL", "Credit Limit (Total Approved Amount)"),
            ("AGE", "Applicant Age"),
            ("PAY_0", "Recent Payment Delay (Last Month)"),
            ("PAY_2", "Payment Delay (2 Months Ago)"),
            ("PAY_3", "Payment Delay (3 Months Ago)"),
            ("BILL_AMT1", "Latest Credit Card Bill Amount"),
            ("PAY_AMT1", "Latest Payment Made")
        ]

        for field_key, field_label in fields:
            frame = tk.Frame(input_frame, bg="#2a2a40")
            frame.pack(fill="x", pady=5, padx=20)

            label = tk.Label(
                frame,
                text=field_label,
                bg="#2a2a40",
                fg="white",
                anchor="w"
            )
            label.pack(side="left")

            entry = tk.Entry(frame)
            entry.pack(side="right", fill="x", expand=True)

            self.entries[field_key] = entry

        evaluate_btn = ttk.Button(
            input_frame,
            text="Evaluate Applicant",
            command=self.evaluate_applicant
        )
        evaluate_btn.pack(pady=20)

        # Right Panel (Results)
        result_frame = tk.Frame(main_frame, bg="#2a2a40")
        result_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        tk.Label(
            result_frame,
            text="Risk Assessment Results",
            bg="#2a2a40",
            fg="#ffffff",
            font=("Segoe UI", 14, "bold")
        ).pack(pady=10)

        self.result_labels = {}

        result_fields = [
            "probability_of_default",
            "credit_score",
            "risk_category",
            "predicted_loss",
            "recommended_premium",
            "decision"
        ]

        for field in result_fields:
            frame = tk.Frame(result_frame, bg="#2a2a40")
            frame.pack(fill="x", pady=5, padx=20)

            label = tk.Label(
                frame,
                text=field.replace("_", " ").title(),
                bg="#2a2a40",
                fg="white",
                width=20,
                anchor="w"
            )
            label.pack(side="left")

            value_label = tk.Label(
                frame,
                text="—",
                bg="#2a2a40",
                fg="#f1fa8c",
                font=("Segoe UI", 12, "bold")
            )
            value_label.pack(side="right")

            self.result_labels[field] = value_label

    # ---------------------------
    # Evaluation Logic
    # ---------------------------

    def evaluate_applicant(self):

        try:
            user_input = {}

            for field, entry in self.entries.items():
                value = float(entry.get())
                user_input[field] = value

            result = self.engine.evaluate_applicant(user_input)

            for key, value in result.items():
                self.result_labels[key].config(text=str(value))

            self._color_decision(result["decision"])

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")

    # ---------------------------
    # Dynamic Coloring
    # ---------------------------

    def _color_decision(self, decision):

        if decision == "Approve":
            color = "#4caf50"
        elif decision == "Manual Review":
            color = "#ff9800"
        else:
            color = "#f44336"

        self.result_labels["decision"].config(fg=color)


# ---------------------------
# Run Application
# ---------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = FinancialRiskApp(root)
    root.mainloop()