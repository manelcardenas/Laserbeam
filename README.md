# Segmentation of a laser trace using Deep Convolutional Neural Networks
A Degree Thesis Submitted to the Faculty of the Escola Tècnica d'Enginyeria de Telecomunicació de Barcelona Universitat Politècnica de Catalunya. 
In partial fulfilment of the requirements for the degree in Bachelor’s degree in Telecommunications Technologies and Services Engineering.
In collaboration with the department of Biomedical Imaging Algorithms of the Czech technical university (CTU), September 2022. 

In this file presents the practical information on the project implementation and how to run it.
- Project structure
- Dataset
- Requirements
- [Runing the code](##-Runing-the-code)
  - Preprocess
  - Train model
  - Evaluate
  
 ## Project structure
 The project has the next directory structure:
 - `data/`: This directory handdles all the input data. There are two subdirectories 
    - `raw_data/`: In this directory is all the data ready to be preprocessed. **Masks** and **Images** can be preprocessed. `src/preprocess.py` is the file where this task will be developed.
        - `Images/`: Contains all the `raw_Images`. 
        - `Masks/`: Contains all the `raw_Masks`.
    - `fit_model/`: In this directory is all the data ready to train. If no preprocess is needed, **save the input data directly to here**.
        - `Images/`: Contains all the images ready to train. If `src/preprocess.py´ is used, all the precessed images will be created in this directory automatically.
        - `Masks/`: Contains all the masks ready to train. If `src/preprocess.py´ is used, all the precessed masks will be created in this directory automatically.
 -  `data_vis/`: This directory contains `plots.py`, this file is in charge of all the plot of the project.
 -  `metrics/`: This directory contains `metrics.py`, in this file all the metrics of evaluation and loss functions used to train the model are defined.
 -  `report/`: Directory containing the reports of the thesis; `Biomedical_Imaging_Algorithms_report.pdf` the report by the Biomedical Imaging Algorithms of the Czech technical university and the `report.pdf` this is the one refering to this thesis.
 -  `results/`: After using `src/train_model.py` to train the model(CAMBIAR ESTO SI AL FINAL UTILIZO EL EVALUATE.PY)...it contains three sub directories.
    -  `equalized_images/`: This directory contains the images after the equalization of the histogram is made. This is just a sanity check in order to see how the ecualization afects to the images. This process is made in `src/preprocess.py` and ploted with `data_vis/plots.py`.
    -  `predicted_images/`: This directory contains the resulting images predicted by the model. (PONER ALGO MAAAAAAAAS, SI VIENEN DE EVALUATION O DE TRAINING.
    -  `Sanity_check/`: Directory containing the images exactly as they will arrive to the model to train it. This images are presented after normalization so even though it is tried to plot them as in `fit_model/Images/`, their will present some diferences in chrominance. It is just a sanity check, his process is made in `src/train_model.py` and ploted with `data_vis/plots.py`.
 - `simulaciones/`: In this directory we can find all the simulations results explaines in the `reprot.pdf`
 - `src/`: Directory with the .Python files with primary purpose of the code. 
    - `preprocess.py`: Script used to do all the preprocessing of the images. Mandatory to obtain the best results. 
    - `train_model.py`: Functions that invoke the `unet_model.py` and develops the training of the model.
    - `unet_model.py`: Scrpt that contains the U-net model used in this project.

## Dataset
## Requirements
## Runing the code

    
