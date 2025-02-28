# deforestation-detection

## Setup

1. **Create a Conda Environment**:
   
   Run the following command to create a new Conda environment using the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```

2. **Activate the Environment**:
   
   Activate the newly created environment with:
   ```bash
   conda activate deforestation-detection
   ```

3. **Download and Place the Dataset**:
   
   Download the dataset from [this link](https://stanfordmlgroup.github.io/projects/forestnet/) and place it in the `data/raw` directory. You can also update the configuration file to point to a different location if preferred. Make sure that the 'ForestNetDataset' folder is being pointed to.

4. **Train the Model**:
   
   Run the `train.py` script to start training the model:
   ```bash
   python train.py
   ```

5. **Evaluate the Model**:
   
   After training, evaluate the model using the `evaluate.py` script. Provide the path to the saved model:
   ```bash
   python evaluate.py --model-path path/to/saved/model
   ```