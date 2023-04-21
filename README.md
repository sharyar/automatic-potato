# automatic-potato
ECE 720 - ML Engineering Project - Winter 2023

## How to start the webapp
1. Clone this repo! `git clone https://github.com/sharyar/automatic-potato`
2. Navigate into the repo directory from your terminal `cd automatic-potato`
3. First create a new python env using conda or pyenv or your favorite python tool! 
   1. I use conda and used `conda create --name potato-env`
   2. `conda activate potato-env`
   3. `conda install python=3.9.16`
4. pip install all the requirements in the requirements.txt file
   1. `pip install -f requirements.txt`
5. Run the training pipelines for the adult dataset. 
   1. Open `01_adult_dataset_final.ipynb`
   2. Run all the cells. This will generate the model and encoder pickle files that streamlit needs for the last step. 
6. Run the streamlit server by using `streamlit run adult_inference_app.py`
   1. You can upload a drifted dataset to it (I have added one in the data directory for convenience) to see the drift detection in action and the inference results. 
7. Sadly, the ui only works for uploading files, not for manual input (didn't have enough time!)