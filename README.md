# Bird Classifier API

<p align="center">
  <img src="img\EX_BIRD.jpg" />
</p>

The aim of this project is to create a deep learning model for bird classification, that will eventually be built into an API. It leverages pre-trained models like ResNet to classify bird species from a given dataset.

## Installation

1) Clone the repository:

    ```bash
    git clone https://github.com/johnhangen/Bird_API
    cd Bird_API
    ```

2) Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3) Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4) Make sure to download the [dataset](https://paperswithcode.com/dataset/nabirds) and place it in the data/ folder.

## Usage

To train the model, run:

```bash
python main.py
```

## TODO

-tracking with wandb

-create API

-create test func

-need to use trained model for segmentation

-refine hyperparams
