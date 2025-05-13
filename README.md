# QuickMasker: SAM-Powered Interactive Segmentation Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) QuickMasker is a tool for simple and efficient semi-automatic image segmentation using Meta AI's Segment Anything Model (SAM). Annotate objects in a folder of images by providing positive and negative point prompts, and save the generated masks in the standard COCO JSON format.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/QuickMasker.git](https://github.com/tolesch/QuickMasker.git)
    cd QuickMasker
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install PyTorch:** Follow the official instructions at [pytorch.org](https://pytorch.org/get-started/locally/) to install PyTorch and Torchvision suitable for your system (CPU or specific CUDA version).

4.  **Install Dependencies:**
    * Using `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```
    * Or, install core dependencies and `segment-anything` separately:
        ```bash
        pip install opencv-python numpy PyYAML Pillow pycocotools tqdm
        pip install git+[https://github.com/facebookresearch/segment-anything.git](https://github.com/facebookresearch/segment-anything.git)
        ```
    * **Install QuickMasker (Editable Mode for Development):**
        From the project root directory (`QuickMasker/`):
        ```bash
        pip install -e .
        ```
        This uses the `setup.py` file to install the package so you can run it using the `quickmasker` command.

## Configuration (`config.yaml`)

Before running, you need to configure QuickMasker by editing the `config.yaml` file located in the project root

**Edit `config.yaml`:**
    * **`paths`**:
        * `image_folder`: **(Required)** Path to your directory of images.
        * `output_coco_file`: Path to save the final COCO annotations.
    * **`sam_model`**:
        * `model_type`: Choose `"vit_h"`, `"vit_l"`, or `"vit_b"`.
        * `device`: Set to `"cuda"` or `"cpu"`.
    * **`annotation`**: Define default COCO category ID and name.
    * **`ui`**: Customize window size, font sizes (for Pillow), colors for text, panels, points, and mask overlays.

## Usage

1.  **Prepare Images:** Place all images you want to annotate into the folder specified by `image_folder` in your `config.yaml`.
3.  **Run the Tool:**
    * Activate your virtual environment (e.g., `source .venv/bin/activate`).
    * If you installed with `pip install -e .`, you can run from any directory:
        ```bash
        quickmasker
        ```
    * Alternatively, from the project root directory (`QuickMasker/`):
        ```bash
        python -m quickmasker.run_quickmasker
        ```
    * You can also specify a different config file:
        ```bash
        quickmasker -c path/to/your/other_config.yaml
        ```

4.  **First Run:** The tool will download the specified SAM model checkpoint file (e.g., `sam_vit_b_01ec64.pth`) if it's not already present.

5.  **Annotation Window:** An OpenCV window will open.

6.  **Keybindings:**
    * **Left Mouse Button:** Add a **positive** point.
    * **Right Mouse Button:** Add a **negative** point.
        * *The mask updates automatically after each click.*
    * **`s` key:** **Save** the currently displayed mask as a COCO annotation.
    * **`c` key:** **Clear** all points and the current mask overlay for the current image.
    * **`n` key:** Go to the **next** image (progress is saved).
    * **`p` key:** Go to the **previous** image (progress is saved).
    * **`d` key:** Initiate **delete** for the current image. A confirmation prompt will appear.
        * Press **`y`** to confirm deletion.
        * Press **`n`** or **`ESC`** to cancel deletion.
    * **`q` key / `ESC` key:** **Quit** the application (progress is saved).

## Output Files

* **`output_annotations.json`** (or configured name): The main output file containing all saved annotations in COCO format.
* **`annotation_state.json`** (or configured name): Stores the last viewed image index and the full COCO data to allow resuming sessions.
* **`models/`** (or configured name): Contains the downloaded SAM model checkpoint (`.pth` file).
* **`fonts/`** (or configured name): Suggested location for your custom font files.

## Resuming Sessions

Simply run the application again using the same `config.yaml` (or by pointing to the same state file implicitly). The tool will load the `annotation_state.json` file and automatically resume from the image you were last working on, including all previously saved annotations and reflecting any deleted images.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to discuss potential changes or features.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
