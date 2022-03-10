# CSRS: Code Search with Relevance Matching and Semantic Matching

## Dependency
* Python 3.6
* Keras 2.3.1
* Tensorflow 1.14


## Usage

   ### Dataset
  To train and evaluate our model:

  1) Download and unzip dataset from [Google Drive](https://drive.google.com/drive/folders/1GZYLT_lzhlVczXjD6dgwVUvDDPHMB6L7?usp=sharing) or [Baidu Pan](https://pan.baidu.com/s/1U_MtFXqq0C-Qh8WUFAWGvg) for Chinese users.
  
  2) Add this dataset folder to `/data/github`.

   ### Configuration

   The hyper-parameters and settings can be edited in `config.py`

   ### Train and Evaluate

   ```bash
python main.py --mode train
   ```

### Evaluate

```bash
python main.py --mode eval
```



