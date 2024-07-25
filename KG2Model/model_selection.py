import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QRadioButton

def select_model(model_list):
    # Check if a QApplication already exists
    app = QApplication.instance()
    if not app:  # Create a new instance if not
        app = QApplication(sys.argv)
    selector = ModelSelector(model_list)
    selector.show()
    app.exec_()
    return selector.get_selected_index()

class ModelSelector(QWidget):
    def __init__(self, model_list):
        super().__init__()
        self._selected_index = -1
        self.radio_buttons = []
        self.initUI(model_list)

    def initUI(self, model_list):
        self.setGeometry(300, 300, 300, 200)
        self.setWindowTitle('Model Selector')

        layout = QVBoxLayout()

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
        self.close()

    def get_selected_index(self):
        return self._selected_index

# Example usage
if __name__ == '__main__':
    model_list = ["Model 1", "Model 2", "Model 3", "Model 4"]  # Replace with your actual model list
    selected_index = select_model(model_list)
    print(f"Selected Model Index: {selected_index}")
