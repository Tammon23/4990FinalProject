import functools

from PyQt6.QtGui import QIcon, QIntValidator
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QApplication, QVBoxLayout, QWidget, QMessageBox

from Train import Train, Preproccesser
from theme import DarkTheme, LightTheme
from PyQt6.QtCore import QThreadPool
from PyQt6 import uic

from gensim.models import KeyedVectors, Word2Vec, FastText, Doc2Vec
from enum import IntEnum, auto
import traceback
import ntpath
import sys

from TabContent import TabTemplate
from Util import ErrorMessageBox, Resources
from multithread import Worker


class ModelTypes(IntEnum):
    def _generate_next_value_(self, start, count, last_values):
        return count

    GENSIM_W2C_MODEL = auto()
    GENSIM_W2C_WV = auto()
    GENSIM_FASTTEXT_MODEL = auto()
    GENSIM_FASTTEXT_WV = auto()
    GENSIM_D2C_MODEL = auto()
    GENSIM_D2C_WV = auto()


class Ui(QMainWindow):
    def __init__(self):
        super(Ui, self).__init__()
        uic.loadUi('main.ui', self)

        self.setWindowTitle("NN Display")
        self.setWindowIcon(QIcon(Resources.LOGO.value))

        self.current_file = None
        self.current_training_file = None
        self.isLoadingModel = False
        self.isTrainingModel = False
        self.TrainedModel = None

        # general
        self.pushButtonOpenModel.clicked.connect(self.open_model_file_dialog)  # noqa
        self.pushButtonLoadModel.clicked.connect(self.on_push_button_load_model)  # noqa
        self.tabWidgetMainTabs.tabCloseRequested.connect(self.close_tab)  # noqa

        # convenience
        self.toolButtonChangeTheme.clicked.connect(self.toggle_theme)
        self.light_mode = True

        # training a new model
        self.pushButtonStartTraining.clicked.connect(self.on_push_button_start_training)
        self.pushButtonOpenFileForTraining.clicked.connect(self.open_file_for_training_dialog)
        self.lineEditTrainGensimBatchSize.setValidator(QIntValidator(1, 3000000))
        self.lineEditTrainGensimEmbeddingSize.setValidator(QIntValidator(1, 1000))
        self.lineEditTrainGensimWindowSize.setValidator(QIntValidator(1, 500))
        self.lineEditTrainGensimMinWordCount.setValidator(QIntValidator(1, 10000))
        self.lineEditTrainGensimNumberOfIterations.setValidator(QIntValidator(1, 1000))
        self.lineEditTrainGensimNumberOfWorkers.setValidator(QIntValidator(-1, 100))

        self.progressBarLoadingModel.hide()
        self.progressBarTraining.hide()
        self.labelCurrentModel.hide()

        self.show()  # Show the GUI

    def toggle_theme(self):
        self.light_mode = not self.light_mode
        app = QApplication.instance()

        if self.light_mode:
            app.setStyle(LightTheme.STYLE)
            app.setPalette(LightTheme())
            self.toolButtonChangeTheme.setIcon(QIcon(Resources.MOON_ICON.value))
        else:
            app.setStyle(DarkTheme.STYLE)
            app.setPalette(DarkTheme())
            self.toolButtonChangeTheme.setIcon(QIcon(Resources.SUN_ICON.value))

    @staticmethod
    def load_model_type(filename, model_type, name):
        try:
            if model_type == ModelTypes.GENSIM_W2C_MODEL:
                return Word2Vec.load(filename), name
            elif model_type in [ModelTypes.GENSIM_W2C_WV, ModelTypes.GENSIM_FASTTEXT_WV, ModelTypes.GENSIM_D2C_WV]:
                return KeyedVectors.load(filename, mmap='r'), name
            elif model_type == ModelTypes.GENSIM_FASTTEXT_MODEL:
                return FastText.load(filename), name
            elif model_type == ModelTypes.GENSIM_D2C_MODEL:
                return Doc2Vec.load(filename), name
            else:
                return None, None

        except:
            return None, None

    @staticmethod
    def get_path(path):
        # Gotten from https://stackoverflow.com/a/8384788/19236048
        head, tail = ntpath.split(path)
        return tail or ntpath.basename(head)

    def close_tab(self, index):
        widget = self.tabWidgetMainTabs.widget(index)
        if widget is not None:
            widget.deleteLater()
        self.tabWidgetMainTabs.removeTab(index)
        self.labelTitle.setText(f"Number of open tabs: {self.tabWidgetMainTabs.count()}")

    # general
    def open_model_file_dialog(self):
        """ Allows user to open a file form the file dialog"""

        dialog = QFileDialog(self)
        dialog.setOption(QFileDialog.Option.ReadOnly)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setViewMode(QFileDialog.ViewMode.Detail)

        if dialog.exec():
            self.labelCurrentModel.show()
            self.current_file = dialog.selectedFiles()[0]
            self.labelCurrentModel.setText(f"Current Selected File: {Ui.get_path(self.current_file)}")

    def on_push_button_load_model(self):
        if self.isLoadingModel:
            ErrorMessageBox("Currently loading model, cannot load another!", warnIcon=True)

        elif self.comboBoxModelType.currentIndex() == -1:
            ErrorMessageBox("Please select a model type first!", warnIcon=True)

        elif self.current_file is None:
            ErrorMessageBox("Please open a model first!", warnIcon=True)

        elif "" == self.lineEditTabName.text():
            ErrorMessageBox("Please enter a tab name", warnIcon=True)

        else:
            self.progressBarLoadingModel.show()
            self.isLoadingModel = True
            # Pass the function to execute
            worker = Worker(Ui.load_model_type, self.current_file, self.comboBoxModelType.currentIndex(),
                            self.lineEditTabName.text())
            worker.signals.result.connect(self.create_new_tab)

            # Execute
            pool = QThreadPool.globalInstance()
            pool.start(worker)

    def create_new_tab(self, ret):
        model, name = ret
        if model is None:
            ErrorMessageBox("Unable to open model file!", warnIcon=True)
            return

        tab = QWidget()
        self.tabWidgetMainTabs.addTab(tab, name)

        tabLayout = QVBoxLayout(tab)
        tabLayout.addWidget(
            TabTemplate(model, self.comboBoxModelType.currentIndex(), self.comboBoxModelType.currentText(),
                        self.current_file))

        # clean up
        self.labelTitle.setText(f"Number of open tabs: {self.tabWidgetMainTabs.count()}")
        self.progressBarLoadingModel.hide()
        self.labelCurrentModel.hide()
        self.current_file = None
        self.lineEditTabName.setText("")
        self.isLoadingModel = False

    def open_file_for_training_dialog(self):
        """ Allows user to open a file form the file dialog"""
        dialog = QFileDialog(self)
        dialog.setOption(QFileDialog.Option.ReadOnly)
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setViewMode(QFileDialog.ViewMode.Detail)

        if dialog.exec():
            self.current_training_file = dialog.selectedFiles()[0]
            self.labelTrainOpenFileName.setText(Ui.get_path(self.current_training_file))

    @staticmethod
    def train_model_type(file: str, filename: str, batch_size: int, embedding_size: int, window_size: int, min_word_count: int,
                         number_of_workers: int, number_of_epochs: int, useSG: bool, model_type: int, save_model: bool):

        # from file get sentences to pass into train
        sentences = list(Preproccesser.preprocess_doc(file, batch_size))
        trainer = Train(sentences, embedding_size, window_size, min_word_count, number_of_workers, number_of_epochs,
                        useSG)
        trainer.train(model_type)

        return trainer, save_model, filename

    def on_push_button_start_training(self):
        if self.current_training_file is None:
            ErrorMessageBox("Please select a corpus file to train first!", warnIcon=True)
            return

        if self.isTrainingModel:
            ErrorMessageBox("Currently training a model, please wait for it to finish first", warnIcon=True)
            return

        batch_size = self.lineEditTrainGensimBatchSize.text()
        embedding_size = self.lineEditTrainGensimEmbeddingSize.text()
        min_word_count = self.lineEditTrainGensimMinWordCount.text()
        nEpochs = self.lineEditTrainGensimNumberOfIterations.text()
        nWorkers = self.lineEditTrainGensimNumberOfWorkers.text()
        window_size = self.lineEditTrainGensimWindowSize.text()
        useSG = self.radioButtonTrainGensimSG.isChecked()
        save_model = self.checkBoxSaveTrainedModel.isChecked()
        filename = self.lineEditSaveTrainedModelFilename.text()
        comboBoxSelection = self.comboBoxTrainModelType.currentIndex()

        if comboBoxSelection == -1:
            ErrorMessageBox("Please select a model type first!", warnIcon=True)
            return

        if len(batch_size) == 0:
            ErrorMessageBox("Please enter a non-empty Training Batch Size", warnIcon=True)
            return

        if len(embedding_size) == 0:
            ErrorMessageBox("Please enter a non-empty Embedding Size", warnIcon=True)
            return

        if len(min_word_count) == 0:
            ErrorMessageBox("Please enter a non-empty Min Word Count", warnIcon=True)
            return

        if len(nEpochs) == 0:
            ErrorMessageBox("Please enter a non-empty Number of Iterations", warnIcon=True)
            return

        if len(nWorkers) == 0:
            ErrorMessageBox("Please enter a non-empty Number of Epochs", warnIcon=True)
            return

        if len(window_size) == 0:
            ErrorMessageBox("Please enter a non-empty Window Size", warnIcon=True)
            return

        filename = filename if "" != filename else "NN-Display-Default.model"
        batch_size, embedding_size, min_word_count, nEpochs, nWorkers, window_size = \
            map(int, [batch_size, embedding_size, min_word_count, nEpochs, nWorkers, window_size])

        self.labelTrainOpenFileName.setText("")
        self.progressBarTraining.show()
        self.isTrainingModel = True

        worker = Worker(Ui.train_model_type, self.current_training_file, filename, batch_size, embedding_size, window_size,
                        min_word_count, nWorkers, nEpochs, useSG, self.comboBoxTrainModelType.currentIndex(),
                        save_model)

        self.current_training_file = None

        worker.signals.result.connect(self.on_result_finished_trained_model)
        worker.signals.finished.connect(self.on_finish_trained_model)

        pool = QThreadPool.globalInstance()
        pool.start(worker)

    def on_finish_trained_model(self):
        self.isTrainingModel = False
        self.progressBarTraining.hide()

    def on_result_finished_trained_model(self, ret):
        model, save_model, filename = ret
        self.TrainedModel = model
        if save_model:
            print("here")
            self.accept_save(None, filename)

            dlg = QMessageBox(self)
            dlg.setWindowTitle("Success!")
            dlg.setText(f"Successfully created and saved model using name {filename}")
            dlg.exec()
            return

        dlg = QMessageBox(self)
        dlg.setWindowTitle("Success!")
        dlg.setText(f"Successfully created model, do you want to save?")
        dlg.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        dlg.buttonClicked.connect(functools.partial(self.accept_save, filename=filename))
        dlg.exec()

    def accept_save(self, button, filename):
        if button is None or button.text() == "OK":
            r, msg = self.TrainedModel.save(filename)
            if not r:
                ErrorMessageBox(msg)


def excepthook(exc_type, exc_value, exc_tb):
    print("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))
    QApplication.quit()


if __name__ == "__main__":
    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    window = Ui()
    ret = app.exec()
    sys.exit(ret)
