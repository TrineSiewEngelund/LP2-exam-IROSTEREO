# “Interesting 🤦🏾‍♀️”
### - An Emoji-Based Approach to Profiling Irony and Stereotype Spreaders on Twitter
---

### Exam project in Language Processing 2, University of Copenhagen

In this repository, you will found our report and code for the exam project in Lanugage Processing 2 at the University of Copenhagen.

The exam was based on a PAN Shared Task, [*Profiling Irony and Stereotype Spreaders on Twitter (IROSTEREO) 2022*](https://pan.webis.de/clef22/pan22-web/author-profiling.html).

---



**Group members 👩🏼👩🏼‍🦰👩🏽**
  - [Caroline Amalie Ørum-Hansen](https://github.com/caorumhansen)
  - [Maja Mittag](https://github.com/MajaMittag)
  - [Trine K. M. Siewartz Engelund](https://github.com/TrineSiewEngelund)

---

 **File overview 📁**


*   Notebook with pipeline and gridsearch:

    `train_models.ipynb`

*   Notebook with pipeline of the best model:

    `best_model.ipynb`

*   Transformers to extract features:

    `feature_transformers.py`

*   Dataset:

    `pan22-author-profiling-training-2022-03-29`
    
*   Function to read dataset:

    `read_files.py`

*   Project report:

    `MittagEngelundOerumhansen.pdf`


To see how we did the gridsearch, please open `train_models.ipynb`.

If your only interested in seeing the pipeline and results from the best performing model, please open `best_model.ipynb`.

---

 **Abstract 📄**

Irony and stereotype spreading is not limited to spoken language – it exists on social media as well. Detecting people who spread this kind of information can be relevant for moderation or security purposes as well as for human-computer interaction. This paper presents an emoji-based approach to the IROSTEREO shared task on detecting irony and stereotype spreaders on Twitter at CLEF 2022. The goal of this author profiling task is to predict a binary label for each author. Our approach focuses on digital-specific phenomena such as emoji use as well as a special writing style based on a 2017-meme used to convey a mocking tone on text. We extracted 18 different surface-level features in three categories (stylometric, sentiment, and lexical), and built a pipeline with cross-validation and grid search that tested three classifiers (Support Vector Machine, Ran- dom Forest, and Logistic Regression). The resulting best model (Random Forest) obtained an accuracy of 0.892. An analysis of feature importance as evaluated by the Random Forest found that stylometric features were among the highest-ranked features.
