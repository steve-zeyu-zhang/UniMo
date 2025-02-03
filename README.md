# PCMG
Official Pytorch implementation of the paper "PCMG:3D point cloud human motion generation based on self-attention and transformer".
![image](https://github.com/user-attachments/assets/74b10cc4-8820-46ae-adeb-dafc43e1a5b0)
# quick start
### Quick Start with `quick_start.py`

If you want to quickly get started and experience the functionality without going through the detailed steps of data processing and training, you can directly run the `quick_start.py` script. This script is designed to provide a streamlined experience for users who want to test or evaluate the system immediately.

1. **Run the Quick Start Script**
   Execute the following command in your terminal:

   ```bash
   python quick_start.py
   ```

2. **Next Steps**
   Once you’ve experienced the quick start, you can dive deeper into the system by:
   - Exploring the data processing pipeline (`humanact12_data_process.py`).
   - Training your own model (`train.py`).
   - Modifying the code to suit your specific needs.
# Start
 HumanAct Dataset Processing

1. **Download the HumanAct Dataset**
   First, download the HumanAct dataset to your local machine.

2. **Data Processing**
   After downloading the dataset, run the `humanact12_data_process.py` script to process the data. This script will preprocess the raw data and generate the format required for training and evaluation.

3. **Running the Script**
   Execute the following command in the terminal to run the data processing script:

   ```bash
   python humanact12_data_process.py --input_dir ./data/humanact --output_dir ./processed_data
   ```

   - `--input_dir`: Specifies the directory of the raw HumanAct dataset.
   - `--output_dir`: Specifies the directory where the processed data will be saved.

   Once the processing is complete, the generated data will be saved in the `output_dir`, ready for subsequent training and evaluation.
# train
### Training with `train.py`

After processing the HumanAct dataset, you can proceed to train your model using the `train.py` script. Here’s how to do it:

1. **Run the Training Script**
   Execute the following command in your terminal to start the training process:

   ```bash
   python train.py --data_dir ./processed_data --output_model_dir ./models
   ```

   - `--data_dir`: Specifies the directory where the processed data is stored (e.g., `./processed_data`).
   - `--output_model_dir`: Specifies the directory where the trained model will be saved (e.g., `./models`).

2. **Additional Arguments (Optional)**
   Depending on your specific implementation, you may have additional arguments to customize the training process. For example:
   - `--batch_size`: Set the batch size for training.
   - `--epochs`: Define the number of training epochs.
   - `--learning_rate`: Adjust the learning rate for the optimizer.

   Example with additional arguments:
   ```bash
   python train.py --data_dir ./processed_data --output_model_dir ./models --batch_size 32 --epochs 50 --learning_rate 0.001
   ```

3. **Monitor Training Progress**
   The script will display training progress, including metrics like loss and accuracy (if applicable). You can also use tools like TensorBoard to visualize the training process if supported.

4. **Save the Trained Model**
   Once training is complete, the trained model will be saved in the specified `output_model_dir`. You can use this model for evaluation or inference.
