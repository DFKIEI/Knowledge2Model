import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QLabel, QMessageBox, QDialog, \
    QRadioButton
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon

import RDF_GPT_TOOL.main
from RDF_GPT_TOOL.main import *


class ModelSelector(QDialog):
    def __init__(self, model_list, parent=None):
        super(ModelSelector, self).__init__(parent)
        self._selected_index = -1
        self.radio_buttons = []
        self.initUI(model_list)

    def initUI(self, model_list):
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Model Selector')

        layout = QVBoxLayout()

        # Instruction label for user
        instruction_label = QLabel("Please choose one model:")
        layout.addWidget(instruction_label)

        for index, model in enumerate(model_list):
            radio_button = QRadioButton(model)
            self.radio_buttons.append(radio_button)
            layout.addWidget(radio_button)

        self.btn = QPushButton('Confirm Selection', self)
        self.btn.clicked.connect(self.confirmSelection)
        layout.addWidget(self.btn)

        self.setLayout(layout)

    def confirmSelection(self):
        for index, radio_button in enumerate(self.radio_buttons):
            if radio_button.isChecked():
                self._selected_index = index
                break
        self.accept()

    def get_selected_index(self):
        return self._selected_index


def select_model(model_list):
    selector = ModelSelector(model_list)
    result = selector.exec_()
    if result == QDialog.Accepted:
        return selector.get_selected_index()
    return -1


def generate_code():
    prompt = text_input.text()
    model_list, model_names, model_info = RDF_GPT_TOOL.main.get_possibel_models(prompt)
    selected_index = select_model(model_list)
    # chosen_model = model_names[0] if len(model_list) == 1 else model_names[
    #    RDF_GPT_TOOL.model_selection.select_model(models)]
    if selected_index != -1:
        print(f"Selected Model Index: {selected_index}")
        QMessageBox.information(window, 'Code Generation', 'Your code will be generated.')
        RDF_GPT_TOOL.main.generate_code(model_names[selected_index], prompt, model_info)
    else:
        QMessageBox.information(window, 'Code Generation', 'No model selected.')

    window.close()


def process_prompt(prompt):
    print("Processing prompt:", prompt)
    # Add your machine learning code generation logic here


app = QApplication(sys.argv)

# Styling the application
app.setStyleSheet("QPushButton { font-size: 16px; }"
                  "QLabel { font-size: 14px; }"
                  "QLineEdit { font-size: 14px; }")

window = QWidget()
window.setWindowTitle("ML Code Generator")
window.setWindowIcon(QIcon('path_to_icon.png'))  # Set the path to your icon

layout = QVBoxLayout()

instruction_label = QLabel("Enter a prompt to generate machine learning Python code:")
layout.addWidget(instruction_label)

text_input = QLineEdit()
text_input.setFont(QFont('Arial', 12))
layout.addWidget(text_input)

generate_button = QPushButton("Generate Code")
generate_button.setFont(QFont('Arial', 14))
generate_button.clicked.connect(generate_code)
layout.addWidget(generate_button)

# Adjusting layout spacing
layout.setSpacing(10)
layout.setAlignment(Qt.AlignTop)

window.setLayout(layout)
window.setGeometry(300, 300, 400, 200)  # Adjust size as needed

window.show()

sys.exit(app.exec_())
