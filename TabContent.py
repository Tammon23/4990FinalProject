from PyQt6.QtWidgets import QButtonGroup
from PyQt6.QtCore import QThreadPool
from PyQt6 import uic

from matplotlib.backends.backend_qt import NavigationToolbar2QT

from Util import ErrorMessageBox
from multithread import Worker
from AI import PointPlotter

form, base = uic.loadUiType('tab_template.ui')


class TabTemplate(base, form):
    def __init__(self, model, model_type_value, model_type, file_location):
        super(base, self).__init__()
        self.setupUi(self)

        self.model = model
        self.model_type = model_type
        self.model_type_value = model_type_value
        self.file_location = file_location
        self.displayWordSubset = set()
        self.isGeneratingImage = False

        self.progressBarPlotModel.hide()
        self.listWidgetWordSubset.hide()
        self.comboBoxAddedWords.hide()
        self.pushButtonAddWord.hide()
        self.pushButtonRemoveWord.hide()
        self.pushButtonRemoveAllWords.hide()
        self.label4.hide()
        self.spinBoxNumWordsSubset.hide()

        self.labelFileLocation.setText(f"File Path: {file_location}")
        self.labelModelType.setText(f"Model Type: {model_type}")

        # Compute most similarities group box
        self.pushButtonComputeSimilarWords.clicked.connect(self.on_push_button_compute_similar_words)

        # Plot model group box
        self.plot_model_button_group = QButtonGroup()
        for i, bt in enumerate(
                [self.rb_random_sample, self.rb_first_n_words, self.rb_last_n_words, self.rb_word_subset]):
            self.plot_model_button_group.addButton(bt, i)
            bt.clicked.connect(self.on_radio_button_clicked)

        self.horizontalSliderPerplexity.valueChanged.connect(self.spinBoxPerplexity.setValue)
        self.spinBoxPerplexity.valueChanged.connect(self.horizontalSliderPerplexity.setValue)
        self.pushButtonGenerate.clicked.connect(self.on_push_button_generate_clicked)

        self.sc = PointPlotter(self, width=6, height=6, dpi=100)
        toolbar = NavigationToolbar2QT(self.sc, self)
        self.verticalLayoutGeneratedImage.addWidget(toolbar)
        self.verticalLayoutGeneratedImage.addWidget(self.sc)

        self.addedWords = set()
        if self.model_type_value in [0, 2, 4]:
            self.addedWords = set(self.model.wv.key_to_index)
        elif self.model_type_value in [1, 3, 5]:
            self.addedWords = set(self.model.key_to_index)
        else:
            ErrorMessageBox(f"Unhandled model type {self.model_type_value}-{self.model_type}")
            exit(1)

        textWidth = self.fontMetrics().horizontalAdvance(max(self.addedWords))
        self.comboBoxWordsCompute.setMinimumWidth(textWidth + 20)
        self.comboBoxAddedWords.setMinimumWidth(textWidth + 20)

        # doing some magic to reduce initial lag spike for combobox with a lot of words
        self.comboBoxAddedWords.addItems(self.addedWords)
        self.comboBoxWordsCompute.addItems(self.addedWords)

        self.pushButtonAddWord.clicked.connect(self.on_push_button_add_word)
        self.pushButtonRemoveWord.clicked.connect(self.on_push_button_remove_word)
        self.pushButtonRemoveAllWords.clicked.connect(self.on_push_button_remove_all_words)

    # Compute most similarities group box
    def on_push_button_compute_similar_words(self):
        word = self.comboBoxWordsCompute.currentText()
        depth = self.spinBoxNumWords.value()
        show_probabilities = self.checkBoxShowEmbeddingProbs.isChecked()
        radio_button_id = self.model_type_value

        if radio_button_id in [0, 2, 4]:
            result = self.model.wv.most_similar(word, topn=depth)
        elif radio_button_id in [1, 3, 5]:
            result = self.model.most_similar(word, topn=depth)
        else:
            ErrorMessageBox("Unknown model type contact author!")
            return

        if word not in self.addedWords:
            ErrorMessageBox("Unknown word, please choose a word that is in the model!", warnIcon=True)
            return

        self.plainTextEdit.clear()

        if show_probabilities:
            longest_word = 9
            for word, _ in result:
                longest_word = max(len(word), longest_word)

            longest_word += 10
            self.plainTextEdit.appendPlainText(f"{'--Words--':<{longest_word}} {'--Probabilities--':<{longest_word}}")
            for word, prob in result:
                self.plainTextEdit.appendPlainText(str(f"{word:<{longest_word}} {prob:<{longest_word}}"))

        else:
            self.plainTextEdit.setPlainText("--Words--\n" + "\n".join([pair[0] for pair in result]))

    # plot model group box
    def on_radio_button_clicked(self):
        if self.plot_model_button_group.checkedId() == 3:
            self.listWidgetWordSubset.show()
            self.comboBoxAddedWords.show()
            self.pushButtonAddWord.show()
            self.pushButtonRemoveWord.show()
            self.pushButtonRemoveAllWords.show()
            self.label4.show()
            self.spinBoxNumWordsSubset.show()
            self.spinBoxNWordsToDisplay.setEnabled(False)

        else:
            self.listWidgetWordSubset.hide()
            self.comboBoxAddedWords.hide()
            self.pushButtonAddWord.hide()
            self.pushButtonRemoveWord.hide()
            self.pushButtonRemoveAllWords.hide()
            self.label4.hide()
            self.spinBoxNumWordsSubset.hide()

            self.spinBoxNWordsToDisplay.setEnabled(True)

    @staticmethod
    def generate_new_image(sc, perplexity, nWords, mode, words):
        if mode == 0:
            sc.TSNEPlot(perplexity=perplexity, num_tokens_to_plot=nWords, use_random_sample=True)
        elif mode == 1:
            sc.TSNEPlot(perplexity=perplexity, num_tokens_to_plot=nWords, use_random_sample=False, useLastN=False)
        elif mode == 2:
            sc.TSNEPlot(perplexity=perplexity, num_tokens_to_plot=nWords, use_random_sample=False, useLastN=True)
        else:
            sc.TSNEPlot(perplexity=perplexity, num_tokens_to_plot=nWords, words=words)

    def on_push_button_generate_clicked(self):

        if self.isGeneratingImage:
            ErrorMessageBox("Cannot generate new image since an image is already generating",
                            info="Please wait for that to finish first!", warnIcon=True)
            return

        else:
            self.isGeneratingImage = True

        radio_button_id = self.plot_model_button_group.checkedId()
        perplexity = self.spinBoxPerplexity.value()
        nWords = self.spinBoxNWordsToDisplay.value()

        words = None if radio_button_id != 3 else self.displayWordSubset

        self.progressBarPlotModel.show()
        self.sc.set_model(self.model if self.model_type_value in [1, 3, 5] else self.model.wv)
        worker = Worker(TabTemplate.generate_new_image, self.sc, perplexity, nWords, radio_button_id, words)
        worker.signals.finished.connect(self.clean_up_image_generator)
        worker.signals.error.connect(ErrorMessageBox)

        # Execute
        pool = QThreadPool.globalInstance()
        pool.start(worker)

    def clean_up_image_generator(self):
        self.isGeneratingImage = False
        self.progressBarPlotModel.hide()

    def on_push_button_add_word(self):
        itemIndex = self.comboBoxAddedWords.currentIndex()
        itemText = self.comboBoxAddedWords.currentText()

        if itemText in self.displayWordSubset:
            ErrorMessageBox(f"Word {itemText} already selected", warnIcon=True)

        elif itemText == "":
            ErrorMessageBox("Word cannot be empty!", warnIcon=True)

        elif itemText != self.comboBoxAddedWords.itemText(itemIndex):
            ErrorMessageBox("Word must exist in model!", warnIcon=True)

        else:
            self.listWidgetWordSubset.addItem(itemText)
            self.displayWordSubset.add(itemText)
            self.spinBoxNumWordsSubset.setValue(self.spinBoxNumWordsSubset.value() + 1)

    def on_push_button_remove_word(self):
        AllItems = self.listWidgetWordSubset.selectedItems()
        if not AllItems:
            ErrorMessageBox("Please select one or more items to remove first", warnIcon=True)
            return
        for item in AllItems:
            self.listWidgetWordSubset.takeItem(self.listWidgetWordSubset.row(item))
            self.displayWordSubset.remove(item.text())
            self.spinBoxNumWordsSubset.setValue(self.spinBoxNumWordsSubset.value() - 1)

    def on_push_button_remove_all_words(self):
        self.spinBoxNumWordsSubset.setValue(0)
        self.listWidgetWordSubset.clear()
        self.displayWordSubset.clear()
