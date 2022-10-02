# Segmentation of a laser trace using Deep Convolutional Neural Networks
A Degree Thesis Submitted to the Faculty of the Escola Tècnica d'Enginyeria de Telecomunicació de Barcelona Universitat Politècnica de Catalunya. 
In partial fulfilment of the requirements for the degree in Bachelor’s degree in Telecommunications Technologies and Services Engineering.
In collaboration with the department of Biomedical Imaging Algorithms of the Czech technical university (CTU), September 2022. 

In this file presents the practical information on the project implementation and how to run it.
- Project structure
- Dataset
- requirements
- Runing the code
  - Preprocess
  - Train model
  - Evaluate
  
 ## Project structure
 The project has the next directory structure:
 - `data/`: This directory handdles all the input data. There are two subdirectories 
    - `raw_data/`: In this directory is all the data ready to be preprocessed. **Masks** and **Images** can be preprocessed. `preprocess.py` is the file where this task will be developed.
        - `Images/`: Contains all the `raw_Images`. 
        - `Masks/`: Contains all the `raw_Masks`.
    - `fit_model/`: In this directory is all the data ready to train. If no preprocess is needed, **save the input data directly to here**.
        - `Images/`: Contains all the images ready to train. If `preprocess.py´ is used, all the precessed images will be created in this directory automatically.
        - `Masks/`: Contains all the masks ready to train. If `preprocess.py´ is used, all the precessed masks will be created in this directory automatically.
