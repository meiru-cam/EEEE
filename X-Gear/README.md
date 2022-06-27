## X-Gear: Multilingual Generative Language Models for Zero-Shot Cross-Lingual Event Argument Extraction

Code for our ACL-2022 paper [Multilingual Generative Language Models for Zero-Shot Cross-Lingual Event Argument Extraction](https://arxiv.org/abs/2203.08308).


### Setup 

  - Python=3.7.10
  ```
  $ conda env create -f environment.yml
  ```

### Data and Preprocessing

- Go into the folder `./preprocessing/`
- If you follow the instruction in the README.md, then you can get your data in the folder `./processed_data/`

### Training

- Run `./scripts/generate_data_ace05.sh` and `./scripts/generate_data_ere.sh` to generate training examples of different languages for X-Gear. 
  The generated training data will be saved in `./finetuned_data/`.
- Run `./scripts/train_ace05.sh` or `./scripts/train_ere.sh` to train X-Gear. Alternatively, you can run the following command.

  ```
  python ./xgear/train.py -c ./config/config_ace05_mT5copy-base_en.json
  ```
  
  This trains X-Gear with mT5-base + copy mechanisim for ACE-05 English. The model will be saved in `./output/`.
  You can modify the arguments in the config file or replace the config file with other files in `./config/`.
  
### Evaluating

- Run the following script to evaluate the performance for ACE-05 English, Arabic, and Chinese.

  ```
  ./scripts/eval_ace05.sh [model_path] [prediction_dir]
  ```
  
  If you want to test X-Gear with mT5-large, remember to modify the config file in `./scripts/eval_ace05.sh`.
  
- Run the following script to evaluate the performance for ERE English and Spanish.

  ```
  ./scripts/eval_ere.sh [model_path] [prediction_dir]
  ```
  
  If you want to test X-Gear with mT5-large, remember to modify the config file in `./scripts/eval_ere.sh`.
  
We provide our pre-trained models and show their performances as follows.

